"""CerebroCortex - The main coordinator that wires all brain regions together.

This is the primary API for the memory system. It orchestrates:
- Thalamus (gating) -> decides what to store
- Amygdala (affect) -> emotional processing
- Temporal (semantic) -> concept extraction and semantic links
- Association (links) -> associative network management
- Hippocampus (episodes) -> temporal sequence management
- Cerebellum (procedural) -> strategy/workflow management
- Prefrontal (executive) -> priorities, promotions, ranking
- Neocortex (schemas) -> abstraction and generalization

The recall pipeline:
1. ChromaDB vector search -> seed results
2. Spreading activation through graph -> expand results
3. ACT-R + FSRS scoring -> rank results
4. Hebbian strengthening -> learn from recall
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

from cerebro.activation.strength import record_access
from cerebro.config import (
    ALL_COLLECTIONS,
    CHROMA_DIR,
    COLLECTION_KNOWLEDGE,
    COLLECTION_MEMORIES,
    COLLECTION_SESSIONS,
    DATA_DIR,
    DEFAULT_AGENT_ID,
    SQLITE_DB,
)
from cerebro.engines.amygdala import AffectEngine
from cerebro.engines.association import LinkEngine
from cerebro.engines.cerebellum import ProceduralEngine
from cerebro.engines.hippocampus import EpisodicEngine
from cerebro.engines.neocortex import SchemaEngine
from cerebro.engines.prefrontal import ExecutiveEngine
from cerebro.engines.temporal import SemanticEngine
from cerebro.engines.thalamus import GatingEngine
from cerebro.models.episode import Episode
from cerebro.models.memory import MemoryNode
from cerebro.storage.chroma_store import ChromaStore
from cerebro.storage.graph_store import GraphStore
from cerebro.types import EmotionalValence, LinkType, MemoryLayer, MemoryType, Visibility

logger = logging.getLogger(__name__)


def _scope_sql(agent_id: Optional[str] = None, conversation_thread: Optional[str] = None) -> tuple[str, list]:
    """Build a SQL WHERE clause fragment for visibility scope filtering.

    Returns (clause_string, params_list). clause_string is empty when no filter needed.
    """
    if not agent_id:
        return "", []
    params = [agent_id, agent_id]
    if conversation_thread:
        clause = (
            " AND (visibility='shared' OR (visibility='private' AND agent_id=?)"
            " OR (visibility='thread' AND (agent_id=? OR conversation_thread=?)))"
        )
        params.append(conversation_thread)
    else:
        clause = (
            " AND (visibility='shared' OR (visibility='private' AND agent_id=?)"
            " OR (visibility='thread' AND agent_id=?))"
        )
    return clause, params


class CerebroCortex:
    """The brain. Wires all engines together into a unified memory system."""

    def __init__(
        self,
        db_path: Optional[Path] = None,
        chroma_dir: Optional[Path] = None,
    ):
        self._db_path = db_path or SQLITE_DB
        self._chroma_dir = chroma_dir or CHROMA_DIR
        self._initialized = False

        # Storage
        self._graph: Optional[GraphStore] = None
        self._chroma: Optional[ChromaStore] = None

        # Engines (initialized in .initialize())
        self.links: Optional[LinkEngine] = None
        self.gating: Optional[GatingEngine] = None
        self.affect: Optional[AffectEngine] = None
        self.semantic: Optional[SemanticEngine] = None
        self.episodes: Optional[EpisodicEngine] = None
        self.procedural: Optional[ProceduralEngine] = None
        self.executive: Optional[ExecutiveEngine] = None
        self.schemas: Optional[SchemaEngine] = None

    @property
    def graph(self) -> GraphStore:
        if self._graph is None:
            raise RuntimeError("CerebroCortex not initialized. Call initialize() first.")
        return self._graph

    @property
    def vector(self) -> ChromaStore:
        if self._chroma is None:
            raise RuntimeError("CerebroCortex not initialized. Call initialize() first.")
        return self._chroma

    def initialize(self) -> None:
        """Initialize all storage backends and engines."""
        # Ensure data directory exists
        DATA_DIR.mkdir(parents=True, exist_ok=True)

        # Initialize graph store (SQLite + igraph)
        self._graph = GraphStore(self._db_path)
        self._graph.initialize()

        # Initialize vector store (ChromaDB)
        self._chroma = ChromaStore(self._chroma_dir)
        self._chroma.initialize()

        # Initialize engines
        self.links = LinkEngine(self._graph)
        self.gating = GatingEngine(self._graph)
        self.affect = AffectEngine(self._graph)
        self.semantic = SemanticEngine(self._graph)
        self.episodes = EpisodicEngine(self._graph)
        self.procedural = ProceduralEngine(self._graph, vector_store=self._chroma)
        self.executive = ExecutiveEngine(self._graph, vector_store=self._chroma)
        self.schemas = SchemaEngine(self._graph, vector_store=self._chroma)

        self._initialized = True

    def close(self) -> None:
        """Shut down all backends."""
        if self._graph:
            self._graph.close()
        # ChromaDB PersistentClient auto-flushes on shutdown
        self._initialized = False

    # =========================================================================
    # Collection routing
    # =========================================================================

    @staticmethod
    def _collection_for_type(memory_type: MemoryType) -> str:
        """Determine which ChromaDB collection a memory belongs in."""
        if memory_type in (MemoryType.SEMANTIC, MemoryType.SCHEMATIC):
            return COLLECTION_KNOWLEDGE
        elif memory_type == MemoryType.EPISODIC:
            return COLLECTION_SESSIONS
        else:  # PROCEDURAL, PROSPECTIVE, AFFECTIVE
            return COLLECTION_MEMORIES

    @staticmethod
    def _can_access(
        node: MemoryNode,
        agent_id: Optional[str] = None,
        conversation_thread: Optional[str] = None,
    ) -> bool:
        """Check if a given agent can access this memory based on visibility.

        Rules:
        - No agent_id filter = see everything (backwards compat)
        - SHARED: visible to all agents
        - PRIVATE: visible only to the owning agent
        - THREAD: visible if conversation_thread matches, else owner-only
        """
        if agent_id is None:
            return True
        vis = node.metadata.visibility
        if vis == Visibility.SHARED:
            return True
        if vis == Visibility.PRIVATE:
            return node.metadata.agent_id == agent_id
        if vis == Visibility.THREAD:
            if conversation_thread and node.metadata.conversation_thread:
                return node.metadata.conversation_thread == conversation_thread
            return node.metadata.agent_id == agent_id
        return False

    @staticmethod
    def _build_where_filter(
        memory_types: Optional[list[MemoryType]] = None,
        agent_id: Optional[str] = None,
        min_salience: float = 0.0,
        conversation_thread: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        """Build a ChromaDB where clause for metadata filtering."""
        clauses = []

        if memory_types:
            type_values = [mt.value for mt in memory_types]
            if len(type_values) == 1:
                clauses.append({"memory_type": type_values[0]})
            else:
                clauses.append({"memory_type": {"$in": type_values}})

        if agent_id:
            vis_clauses = [
                {"visibility": "shared"},
                {"$and": [{"visibility": "private"}, {"agent_id": agent_id}]},
            ]
            if conversation_thread:
                vis_clauses.append(
                    {"$and": [{"visibility": "thread"}, {"conversation_thread": conversation_thread}]}
                )
            else:
                vis_clauses.append(
                    {"$and": [{"visibility": "thread"}, {"agent_id": agent_id}]}
                )
            clauses.append({"$or": vis_clauses})

        if min_salience > 0.0:
            clauses.append({"salience": {"$gte": min_salience}})

        if not clauses:
            return None
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}

    # =========================================================================
    # REMEMBER - Store a new memory
    # =========================================================================

    def remember(
        self,
        content: str,
        memory_type: Optional[MemoryType] = None,
        tags: Optional[list[str]] = None,
        salience: Optional[float] = None,
        agent_id: str = DEFAULT_AGENT_ID,
        session_id: Optional[str] = None,
        visibility: Visibility = Visibility.SHARED,
        context_ids: Optional[list[str]] = None,
    ) -> Optional[MemoryNode]:
        """Store a new memory through the full encoding pipeline.

        Pipeline:
        1. Thalamus gating (dedup, noise filter, layer assignment)
        2. Semantic engine (concept extraction)
        3. Amygdala (emotional analysis, salience adjustment)
        4. Graph store + Vector store (persist node)
        5. Link engine (auto-link to related memories)
        6. Episode tracking (add to current episode if active)

        Args:
            content: The memory content
            memory_type: Override automatic type classification
            tags: Tags for categorization
            salience: Override automatic salience estimation
            agent_id: Agent storing this memory
            session_id: Current session ID
            visibility: Sharing scope
            context_ids: IDs of memories active in current context

        Returns:
            The stored MemoryNode, or None if gated out (duplicate/noise).
        """
        # 1. Thalamus: should we remember this?
        node = self.gating.evaluate_input(
            content=content,
            memory_type=memory_type,
            tags=tags,
            salience=salience,
            agent_id=agent_id,
            session_id=session_id,
            visibility=visibility,
        )
        if node is None:
            return None

        # 2. Semantic: extract concepts
        node = self.semantic.enrich_node(node)

        # 3. Amygdala: emotional analysis
        node = self.affect.apply_emotion(node)

        # 4. Record first access
        now = time.time()
        node = MemoryNode(
            id=node.id,
            content=node.content,
            metadata=node.metadata,
            strength=record_access(node.strength, now),
            created_at=node.created_at,
        )

        # 5. Persist to graph store
        self._graph.add_node(node)

        # 6. Persist to vector store (ChromaDB)
        coll = self._collection_for_type(node.metadata.memory_type)
        self._chroma.add_node(coll, node)

        # 7. Auto-link
        self.links.auto_link_on_store(node, context_ids=context_ids)
        self.semantic.create_semantic_links(node)
        self.affect.create_affective_links(node)

        # 8. Episode tracking
        if session_id and self.episodes:
            self.episodes.add_to_current_episode(session_id, node.id)

        return node

    # =========================================================================
    # SEND MESSAGE - Direct agent-to-agent communication
    # =========================================================================

    def send_message(
        self,
        to: str,
        content: str,
        agent_id: str = DEFAULT_AGENT_ID,
        in_reply_to: Optional[str] = None,
        tags: Optional[list[str]] = None,
        session_id: Optional[str] = None,
    ) -> MemoryNode:
        """Send a message to another agent, bypassing the gating engine.

        Messages are always stored (no dedup/noise filter), auto-tagged with
        from/to, and stored as SEMANTIC in cerebro_knowledge so all agents
        can find them via recall too.

        Args:
            to: Recipient agent ID, or "all" for broadcast.
            content: Message content.
            agent_id: Sender agent ID.
            in_reply_to: Memory ID of message being replied to (creates link).
            tags: Additional tags.
            session_id: Current session ID.

        Returns:
            The stored MemoryNode.
        """
        from cerebro.models.memory import MemoryMetadata, MemoryNode as MN

        # Auto-tags
        auto_tags = ["msg", f"from:{agent_id}", f"to:{to}"]
        if tags:
            auto_tags.extend(tags)

        # Build node directly — bypass thalamus gating
        node = MN(
            content=content,
            metadata=MemoryMetadata(
                agent_id=agent_id,
                visibility=Visibility.SHARED,
                layer=MemoryLayer.WORKING,
                memory_type=MemoryType.SEMANTIC,
                tags=auto_tags,
                source="agent_message",
                recipient=to,
                salience=0.8,
                session_id=session_id,
                related_agents=[to],
                responding_to=[in_reply_to] if in_reply_to else [],
            ),
        )

        # Semantic enrichment + affect analysis (but no gating)
        node = self.semantic.enrich_node(node)
        node = self.affect.apply_emotion(node)

        # Record first access
        now = time.time()
        node = MN(
            id=node.id,
            content=node.content,
            metadata=node.metadata,
            strength=record_access(node.strength, now),
            created_at=node.created_at,
        )

        # Dual-write to graph + vector store
        self._graph.add_node(node)
        coll = self._collection_for_type(node.metadata.memory_type)
        self._chroma.add_node(coll, node)

        # Auto-link
        self.links.auto_link_on_store(node)
        self.semantic.create_semantic_links(node)

        # Reply link
        if in_reply_to and self._graph.get_node(in_reply_to):
            self._graph.ensure_link(
                source_id=node.id,
                target_id=in_reply_to,
                link_type=LinkType.SUPPORTS,
                weight=0.8,
                source="agent_message",
                evidence=f"Reply from {agent_id} to {to}",
            )

        return node

    # =========================================================================
    # CHECK INBOX - Retrieve messages addressed to this agent
    # =========================================================================

    def check_inbox(
        self,
        agent_id: str = DEFAULT_AGENT_ID,
        from_agent: Optional[str] = None,
        limit: int = 10,
        since: Optional[str] = None,
    ) -> list[MemoryNode]:
        """Check for messages addressed to this agent.

        Queries SQLite directly by the indexed recipient column.

        Args:
            agent_id: The agent checking their inbox.
            from_agent: Only show messages from this sender.
            limit: Max messages to return.
            since: Only messages after this ISO timestamp.

        Returns:
            List of MemoryNodes, newest first.
        """
        query = (
            "SELECT * FROM memory_nodes "
            "WHERE source='agent_message' AND (recipient=? OR recipient='all')"
        )
        params: list = [agent_id]

        if from_agent:
            query += " AND agent_id=?"
            params.append(from_agent)

        if since:
            query += " AND created_at>=?"
            params.append(since)

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        rows = self._graph.conn.execute(query, params).fetchall()
        return [self._graph._row_to_memory_node(r) for r in rows]

    # =========================================================================
    # RECALL - Search and retrieve memories
    # =========================================================================

    def recall(
        self,
        query: str,
        top_k: int = 10,
        memory_types: Optional[list[MemoryType]] = None,
        agent_id: Optional[str] = None,
        min_salience: float = 0.0,
        context_ids: Optional[list[str]] = None,
        conversation_thread: Optional[str] = None,
    ) -> list[tuple[MemoryNode, float]]:
        """Recall memories through the full retrieval pipeline.

        Pipeline:
        1. Vector search via ChromaDB (semantic seeds)
        2. Spreading activation from seeds + context
        3. ACT-R + FSRS combined scoring
        4. Hebbian strengthening of recalled paths
        5. Update access timestamps

        Args:
            query: Search query text
            top_k: Maximum results to return
            memory_types: Filter by memory type
            agent_id: Filter by agent
            min_salience: Minimum salience threshold
            context_ids: Memory IDs to use as activation seeds
            conversation_thread: Thread ID for THREAD-visibility matching

        Returns:
            List of (MemoryNode, score) tuples, sorted by score descending.
        """
        # 1. Vector search across all ChromaDB collections
        where_filter = self._build_where_filter(
            memory_types, agent_id, min_salience, conversation_thread,
        )
        vector_results: dict[str, float] = {}
        per_collection = max(top_k, 10)

        for coll_name in ALL_COLLECTIONS:
            hits = self._chroma.search(
                collection=coll_name,
                query=query,
                n_results=per_collection,
                where=where_filter,
            )
            for hit in hits:
                doc_id = hit["id"]
                similarity = hit.get("similarity") or 0.0
                # Keep best similarity if same ID appears in multiple collections
                if doc_id not in vector_results or similarity > vector_results[doc_id]:
                    vector_results[doc_id] = similarity

        # 2. Spreading activation from vector seeds + context_ids
        seed_ids = list(vector_results.keys())
        seed_weights = [vector_results[sid] for sid in seed_ids]

        # Add explicit context seeds
        if context_ids:
            for cid in context_ids:
                if cid not in vector_results:
                    seed_ids.append(cid)
                    seed_weights.append(0.8)

        activated: dict[str, float] = {}
        if seed_ids:
            activated = self.links.spread_activation(
                seed_ids=seed_ids,
                seed_weights=seed_weights,
                agent_id=agent_id,
                conversation_thread=conversation_thread,
            )

        # 3. Merge candidates: vector hits + activated nodes
        candidate_ids = list(set(vector_results.keys()) | set(activated.keys()))

        # Filter by type/scope/salience (belt-and-suspenders with ChromaDB where)
        filtered_ids = []
        for mid in candidate_ids:
            node = self._graph.get_node(mid)
            if not node:
                continue
            if memory_types and node.metadata.memory_type not in memory_types:
                continue
            if not self._can_access(node, agent_id, conversation_thread):
                continue
            if node.metadata.salience < min_salience:
                continue
            filtered_ids.append(mid)

        # 4. Rank using executive engine (with vector similarities)
        ranked = self.executive.rank_results(
            memory_ids=filtered_ids,
            vector_similarities=vector_results,
            associative_scores=activated,
        )

        # Take top_k
        top_results = ranked[:top_k]

        # 5. Hebbian strengthening of co-activated memories
        result_ids = [mid for mid, _ in top_results]
        if len(result_ids) > 1:
            self.links.strengthen_co_activated(result_ids)

        # 6. Update access timestamps for recalled memories
        now = time.time()
        results = []
        for mid, score in top_results:
            node = self._graph.get_node(mid)
            if node:
                new_strength = record_access(node.strength, now)
                self._graph.update_node_strength(mid, new_strength)
                results.append((node, score))

        return results

    # =========================================================================
    # ASSOCIATE - Create explicit links
    # =========================================================================

    def associate(
        self,
        source_id: str,
        target_id: str,
        link_type: LinkType,
        weight: float = 0.5,
        evidence: Optional[str] = None,
    ) -> Optional[str]:
        """Create an explicit associative link between memories."""
        if not self._graph.get_node(source_id) or not self._graph.get_node(target_id):
            return None
        return self.links.create_link(
            source_id, target_id, link_type,
            weight=weight, source="user", evidence=evidence,
        )

    # =========================================================================
    # GET / DELETE / UPDATE single memory
    # =========================================================================

    def get_memory(
        self,
        memory_id: str,
        agent_id: Optional[str] = None,
    ) -> Optional[MemoryNode]:
        """Get a single memory by ID.

        If agent_id is provided, returns None when the agent lacks access.
        """
        node = self._graph.get_node(memory_id)
        if not node:
            return None
        if not self._can_access(node, agent_id):
            return None
        return node

    def delete_memory(
        self,
        memory_id: str,
        agent_id: Optional[str] = None,
    ) -> bool:
        """Delete a memory from both GraphStore and ChromaDB.

        If agent_id is provided, only the owner can delete.
        Returns True if found and deleted.
        """
        node = self._graph.get_node(memory_id)
        if not node:
            return False
        if not self._can_access(node, agent_id):
            return False

        # Delete from graph store (SQLite + igraph rebuild)
        self._graph.delete_node(memory_id)

        # Delete from ChromaDB
        coll = self._collection_for_type(node.metadata.memory_type)
        self._chroma.delete(coll, [memory_id])

        return True

    def update_memory(
        self,
        memory_id: str,
        content: Optional[str] = None,
        tags: Optional[list[str]] = None,
        salience: Optional[float] = None,
        visibility: Optional[Visibility] = None,
        agent_id: Optional[str] = None,
    ) -> Optional[MemoryNode]:
        """Update a memory's content and/or metadata.

        If agent_id is provided, checks access before updating.
        If content is provided, re-embeds in ChromaDB.
        Returns the updated MemoryNode, or None if not found or access denied.
        """
        node = self._graph.get_node(memory_id)
        if not node:
            return None
        if not self._can_access(node, agent_id):
            return None

        # Build metadata updates
        meta_updates = {}
        if tags is not None:
            meta_updates["tags_json"] = json.dumps(tags)
        if salience is not None:
            meta_updates["salience"] = salience
        if visibility is not None:
            meta_updates["visibility"] = visibility.value

        # Update content in SQLite (separate from metadata)
        if content is not None:
            content_hash = self._graph._content_hash(content)
            self._graph.conn.execute(
                "UPDATE memory_nodes SET content = ?, content_hash = ? WHERE id = ?",
                (content, content_hash, memory_id),
            )
            self._graph.conn.commit()

        if meta_updates:
            self._graph.update_node_metadata(memory_id, **meta_updates)

        # Re-fetch updated node
        updated = self._graph.get_node(memory_id)

        # Sync to ChromaDB (re-embeds if content changed)
        coll = self._collection_for_type(updated.metadata.memory_type)
        self._chroma.update_node(coll, updated)

        # Prune cross-agent links when visibility changes to PRIVATE
        if visibility == Visibility.PRIVATE:
            self._prune_cross_agent_links(memory_id, updated.metadata.agent_id)

        return updated

    def _prune_cross_agent_links(self, memory_id: str, owner_agent_id: str) -> int:
        """Remove links where the PRIVATE memory crosses agent boundaries.

        When a memory becomes PRIVATE, links to/from memories owned by other
        agents are pruned to maintain isolation. Same-agent links are preserved.

        Returns the number of links pruned.
        """
        rows = self._graph.conn.execute(
            "SELECT id, source_id, target_id FROM associative_links "
            "WHERE source_id = ? OR target_id = ?",
            (memory_id, memory_id),
        ).fetchall()

        pruned = 0
        for row in rows:
            other_id = row["target_id"] if row["source_id"] == memory_id else row["source_id"]
            other = self._graph.get_node(other_id)
            if other and other.metadata.agent_id != owner_agent_id:
                self._graph.conn.execute(
                    "DELETE FROM associative_links WHERE id = ?", (row["id"],)
                )
                pruned += 1

        if pruned:
            self._graph.conn.commit()
            self._graph.resync_igraph()
        return pruned

    def share_memory(
        self,
        memory_id: str,
        new_visibility: Visibility,
        agent_id: Optional[str] = None,
    ) -> Optional[MemoryNode]:
        """Change a memory's visibility. Only the owner can change visibility.

        Args:
            memory_id: Memory to change
            new_visibility: Target visibility level
            agent_id: Requesting agent (must be owner if provided)

        Returns:
            Updated MemoryNode, or None if not found or not authorized.
        """
        node = self._graph.get_node(memory_id)
        if not node:
            return None
        # Only the owner can change visibility
        if agent_id and node.metadata.agent_id != agent_id:
            return None
        updated = self.update_memory(memory_id, visibility=new_visibility)
        # Prune cross-agent links when going PRIVATE
        if updated and new_visibility == Visibility.PRIVATE:
            self._prune_cross_agent_links(memory_id, node.metadata.agent_id)
        return updated

    # =========================================================================
    # Episode management (delegates to hippocampus)
    # =========================================================================

    def episode_start(
        self,
        title: Optional[str] = None,
        session_id: Optional[str] = None,
        agent_id: str = DEFAULT_AGENT_ID,
    ) -> Episode:
        """Start recording a new episode."""
        return self.episodes.start_episode(
            title=title, session_id=session_id, agent_id=agent_id,
        )

    def episode_end(
        self,
        episode_id: str,
        summary: Optional[str] = None,
        valence: EmotionalValence = EmotionalValence.NEUTRAL,
    ) -> Optional[Episode]:
        """End the current episode."""
        return self.episodes.end_episode(
            episode_id, summary=summary, valence=valence,
        )

    def list_episodes(
        self,
        limit: int = 10,
        agent_id: Optional[str] = None,
    ) -> list[Episode]:
        """Get recent episodes."""
        return self.episodes.get_recent_episodes(limit=limit, agent_id=agent_id)

    def get_episode(self, episode_id: str, agent_id: Optional[str] = None) -> Optional[Episode]:
        """Get an episode by ID with all its steps.

        If agent_id is provided, only returns the episode if it belongs to that agent.
        """
        ep = self._graph.get_episode(episode_id)
        if ep is None:
            return None
        if agent_id and ep.agent_id != agent_id:
            return None
        return ep

    def get_episode_memories(self, episode_id: str, agent_id: Optional[str] = None) -> list[MemoryNode]:
        """Get all memories in an episode, ordered by position.

        If agent_id is provided, verifies the episode belongs to that agent first.
        """
        ep = self._graph.get_episode(episode_id)
        if ep is None:
            return []
        if agent_id and ep.agent_id != agent_id:
            return []
        memory_ids = self.episodes.get_episode_memories(episode_id)
        return [n for mid in memory_ids if (n := self._graph.get_node(mid))]

    # =========================================================================
    # Intentions (prospective memory)
    # =========================================================================

    def store_intention(
        self,
        content: str,
        tags: Optional[list[str]] = None,
        agent_id: str = DEFAULT_AGENT_ID,
        salience: float = 0.7,
    ) -> MemoryNode:
        """Store a prospective memory (future intention / TODO)."""
        return self.executive.store_intention(
            content=content, tags=tags, agent_id=agent_id, salience=salience,
        )

    def list_intentions(
        self,
        agent_id: Optional[str] = None,
        min_salience: float = 0.3,
        conversation_thread: Optional[str] = None,
    ) -> list[MemoryNode]:
        """Get all pending intentions."""
        return self.executive.get_pending_intentions(
            agent_id=agent_id, min_salience=min_salience,
            conversation_thread=conversation_thread,
        )

    def resolve_intention(self, memory_id: str) -> bool:
        """Mark an intention as resolved (lowers salience)."""
        return self.executive.resolve_intention(memory_id)

    # =========================================================================
    # Graph exploration (LinkEngine)
    # =========================================================================

    def find_path(self, source_id: str, target_id: str) -> Optional[list[str]]:
        """Find shortest path between two memories in the graph."""
        return self.links.find_path(source_id, target_id)

    def get_common_neighbors(self, id_a: str, id_b: str) -> list[str]:
        """Find memories connected to both A and B."""
        return self.links.get_common_neighbors(id_a, id_b)

    # =========================================================================
    # Schemas (SchemaEngine / Neocortex)
    # =========================================================================

    def create_schema(
        self,
        content: str,
        source_ids: list[str],
        tags: Optional[list[str]] = None,
        agent_id: str = DEFAULT_AGENT_ID,
    ) -> MemoryNode:
        """Create an abstract schema from source memories."""
        return self.schemas.create_schema(
            content=content, source_ids=source_ids, tags=tags, agent_id=agent_id,
        )

    def list_schemas(
        self,
        agent_id: Optional[str] = None,
        conversation_thread: Optional[str] = None,
    ) -> list[MemoryNode]:
        """Get all schematic memories."""
        return self.schemas.get_all_schemas(
            agent_id=agent_id, conversation_thread=conversation_thread,
        )

    def find_matching_schemas(
        self,
        tags: Optional[list[str]] = None,
        concepts: Optional[list[str]] = None,
        agent_id: Optional[str] = None,
        conversation_thread: Optional[str] = None,
    ) -> list[MemoryNode]:
        """Find schemas matching tags or concepts."""
        return self.schemas.find_matching_schemas(
            tags=tags, concepts=concepts,
            agent_id=agent_id, conversation_thread=conversation_thread,
        )

    def get_schema_sources(self, schema_id: str) -> list[str]:
        """Get source memory IDs for a schema."""
        return self.schemas.get_schema_sources(schema_id)

    # =========================================================================
    # Procedures (ProceduralEngine / Cerebellum)
    # =========================================================================

    def store_procedure(
        self,
        content: str,
        tags: Optional[list[str]] = None,
        derived_from: Optional[list[str]] = None,
        agent_id: str = DEFAULT_AGENT_ID,
    ) -> MemoryNode:
        """Store a procedural memory (strategy/workflow)."""
        return self.procedural.store_procedure(
            content=content, tags=tags, derived_from=derived_from, agent_id=agent_id,
        )

    def list_procedures(
        self,
        agent_id: Optional[str] = None,
        min_salience: float = 0.0,
        conversation_thread: Optional[str] = None,
    ) -> list[MemoryNode]:
        """Get all procedural memories."""
        return self.procedural.get_all_procedures(
            agent_id=agent_id, min_salience=min_salience,
            conversation_thread=conversation_thread,
        )

    def find_relevant_procedures(
        self,
        tags: Optional[list[str]] = None,
        concepts: Optional[list[str]] = None,
        agent_id: Optional[str] = None,
        conversation_thread: Optional[str] = None,
    ) -> list[MemoryNode]:
        """Find procedures matching tags or concepts."""
        return self.procedural.find_relevant_procedures(
            tags=tags, concepts=concepts,
            agent_id=agent_id, conversation_thread=conversation_thread,
        )

    def record_procedure_outcome(self, procedure_id: str, success: bool) -> bool:
        """Record success/failure of a procedure."""
        return self.procedural.record_outcome(procedure_id, success)

    # =========================================================================
    # Emotional summary (AffectEngine / Amygdala)
    # =========================================================================

    def get_emotional_summary(
        self,
        agent_id: Optional[str] = None,
        conversation_thread: Optional[str] = None,
    ) -> dict[str, int]:
        """Get breakdown of memories by emotional valence."""
        return self.affect.get_emotional_summary(
            agent_id=agent_id, conversation_thread=conversation_thread,
        )

    # =========================================================================
    # Stats
    # =========================================================================

    def stats(self, agent_id: Optional[str] = None) -> dict:
        """Get comprehensive system statistics.

        Args:
            agent_id: If provided, scope counts to this agent's visible memories.
        """
        graph_stats = self._graph.stats(agent_id=agent_id)
        return {
            **graph_stats,
            "vector_store": self._chroma.count_all() if self._chroma else {},
            "schemas": self.schemas.count_schemas(),
            "initialized": self._initialized,
        }

    def backfill_vector_store(self) -> dict[str, int]:
        """Backfill ChromaDB from GraphStore for memories missing from vector search.

        Reads all nodes from SQLite, checks which are missing from ChromaDB,
        and inserts them. Returns counts by collection.
        """
        if not self._initialized:
            raise RuntimeError("CerebroCortex not initialized. Call initialize() first.")
        all_ids = self._graph.get_all_node_ids()
        existing: set[str] = set()
        for coll_name in ALL_COLLECTIONS:
            for rec in self._chroma.get(coll_name, all_ids):
                existing.add(rec["id"])

        missing_ids = [nid for nid in all_ids if nid not in existing]
        if not missing_ids:
            logger.info("Backfill: all memories already in vector store")
            return {"total": 0}

        counts: dict[str, int] = {}
        errors = 0
        for nid in missing_ids:
            node = self._graph.get_node(nid)
            if node is None:
                continue
            coll = self._collection_for_type(node.metadata.memory_type)
            try:
                self._chroma.add_node(coll, node)
                counts[coll] = counts.get(coll, 0) + 1
            except Exception as e:
                logger.error(f"Backfill failed for {nid}: {e}")
                errors += 1

        total = sum(counts.values())
        logger.info(f"Backfill complete: {total} memories added, {errors} errors")
        return {**counts, "total": total, "errors": errors}
