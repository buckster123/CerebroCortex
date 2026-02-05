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

import time
from pathlib import Path
from typing import Optional

from cerebro.activation.strength import record_access
from cerebro.config import CHROMA_DIR, DATA_DIR, SQLITE_DB
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
from cerebro.storage.graph_store import GraphStore
from cerebro.types import EmotionalValence, LinkType, MemoryType, Visibility


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

    def initialize(self) -> None:
        """Initialize all storage backends and engines."""
        # Ensure data directory exists
        DATA_DIR.mkdir(parents=True, exist_ok=True)

        # Initialize graph store (SQLite + igraph)
        self._graph = GraphStore(self._db_path)
        self._graph.initialize()

        # Initialize engines
        self.links = LinkEngine(self._graph)
        self.gating = GatingEngine(self._graph)
        self.affect = AffectEngine(self._graph)
        self.semantic = SemanticEngine(self._graph)
        self.episodes = EpisodicEngine(self._graph)
        self.procedural = ProceduralEngine(self._graph)
        self.executive = ExecutiveEngine(self._graph)
        self.schemas = SchemaEngine(self._graph)

        self._initialized = True

    def close(self) -> None:
        """Shut down all backends."""
        if self._graph:
            self._graph.close()
        self._initialized = False

    # =========================================================================
    # REMEMBER - Store a new memory
    # =========================================================================

    def remember(
        self,
        content: str,
        memory_type: Optional[MemoryType] = None,
        tags: Optional[list[str]] = None,
        salience: Optional[float] = None,
        agent_id: str = "CLAUDE",
        session_id: Optional[str] = None,
        visibility: Visibility = Visibility.SHARED,
        context_ids: Optional[list[str]] = None,
    ) -> Optional[MemoryNode]:
        """Store a new memory through the full encoding pipeline.

        Pipeline:
        1. Thalamus gating (dedup, noise filter, layer assignment)
        2. Semantic engine (concept extraction)
        3. Amygdala (emotional analysis, salience adjustment)
        4. Graph store (persist node)
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

        # 6. Auto-link
        self.links.auto_link_on_store(node, context_ids=context_ids)
        self.semantic.create_semantic_links(node)
        self.affect.create_affective_links(node)

        # 7. Episode tracking
        if session_id and self.episodes:
            self.episodes.add_to_current_episode(session_id, node.id)

        return node

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
    ) -> list[tuple[MemoryNode, float]]:
        """Recall memories through the full retrieval pipeline.

        Pipeline:
        1. Vector search via ChromaDB (seeds)  [TODO: requires ChromaStore]
        2. Spreading activation from seeds
        3. ACT-R + FSRS combined scoring
        4. Hebbian strengthening of recalled paths
        5. Update access timestamps

        For now (without ChromaStore integration), uses graph-only recall
        with spreading activation from context_ids.

        Args:
            query: Search query text
            top_k: Maximum results to return
            memory_types: Filter by memory type
            agent_id: Filter by agent
            min_salience: Minimum salience threshold
            context_ids: Memory IDs to use as activation seeds

        Returns:
            List of (MemoryNode, score) tuples, sorted by score descending.
        """
        seed_ids = context_ids or []
        seed_weights = [0.8] * len(seed_ids)

        # Spreading activation from seeds
        activated = {}
        if seed_ids:
            activated = self.links.spread_activation(
                seed_ids=seed_ids,
                seed_weights=seed_weights,
            )

        # Get candidate memory IDs
        candidate_ids = list(activated.keys()) if activated else self._graph.get_all_node_ids()

        # Filter by type/agent/salience
        filtered_ids = []
        for mid in candidate_ids:
            node = self._graph.get_node(mid)
            if not node:
                continue
            if memory_types and node.metadata.memory_type not in memory_types:
                continue
            if agent_id and node.metadata.agent_id != agent_id:
                continue
            if node.metadata.salience < min_salience:
                continue
            filtered_ids.append(mid)

        # Rank using executive engine
        ranked = self.executive.rank_results(
            memory_ids=filtered_ids,
            associative_scores=activated,
        )

        # Take top_k
        top_results = ranked[:top_k]

        # Hebbian strengthening of co-activated memories
        result_ids = [mid for mid, _ in top_results]
        if len(result_ids) > 1:
            self.links.strengthen_co_activated(result_ids)

        # Update access timestamps for recalled memories
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
    # Episode management (delegates to hippocampus)
    # =========================================================================

    def episode_start(
        self,
        title: Optional[str] = None,
        session_id: Optional[str] = None,
        agent_id: str = "CLAUDE",
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

    # =========================================================================
    # Stats
    # =========================================================================

    def stats(self) -> dict:
        """Get comprehensive system statistics."""
        graph_stats = self._graph.stats()
        return {
            **graph_stats,
            "schemas": self.schemas.count_schemas(),
            "initialized": self._initialized,
        }
