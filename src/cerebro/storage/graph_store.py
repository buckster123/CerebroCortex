"""Graph store: SQLite persistence + igraph in-memory traversal.

This is the most novel component of CerebroCortex. It bridges:
- SQLite: canonical persistent storage for nodes, edges, episodes
- igraph: in-memory C-speed graph for spreading activation and traversal

Node IDs in the graph correspond to document IDs in ChromaDB.
"""

import hashlib
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

import igraph  # type: ignore

from cerebro.models.agent import AgentProfile
from cerebro.models.episode import Episode, EpisodeStep
from cerebro.models.link import AssociativeLink
from cerebro.models.memory import MemoryMetadata, MemoryNode, StrengthState
from cerebro.storage.sqlite_schema import initialize_database
from cerebro.types import LinkType, MemoryLayer, MemoryType


class GraphStore:
    """Manages the SQLite persistence and igraph in-memory graph."""

    def __init__(self, db_path: Path):
        self._db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._graph: Optional[igraph.Graph] = None
        self._id_to_vertex: dict[str, int] = {}
        self._vertex_to_id: dict[int, str] = {}

    def initialize(self) -> None:
        """Create/open database and load graph into memory."""
        self._conn = initialize_database(self._db_path)
        self._rebuild_igraph()

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("GraphStore not initialized. Call initialize() first.")
        return self._conn

    @property
    def graph(self) -> igraph.Graph:
        if self._graph is None:
            raise RuntimeError("GraphStore not initialized. Call initialize() first.")
        return self._graph

    # =========================================================================
    # igraph management
    # =========================================================================

    def _rebuild_igraph(self) -> None:
        """Load the full graph from SQLite into igraph."""
        g = igraph.Graph(directed=True)
        self._id_to_vertex = {}
        self._vertex_to_id = {}

        # Load all node IDs
        rows = self.conn.execute("SELECT id FROM memory_nodes").fetchall()
        if rows:
            ids = [r["id"] for r in rows]
            g.add_vertices(len(ids))
            for idx, node_id in enumerate(ids):
                g.vs[idx]["node_id"] = node_id
                self._id_to_vertex[node_id] = idx
                self._vertex_to_id[idx] = node_id

        # Load all edges
        edges = self.conn.execute(
            "SELECT source_id, target_id, link_type, weight FROM associative_links"
        ).fetchall()
        if edges:
            edge_list = []
            edge_types = []
            edge_weights = []
            for e in edges:
                src_idx = self._id_to_vertex.get(e["source_id"])
                tgt_idx = self._id_to_vertex.get(e["target_id"])
                if src_idx is not None and tgt_idx is not None:
                    edge_list.append((src_idx, tgt_idx))
                    edge_types.append(e["link_type"])
                    edge_weights.append(e["weight"])
            if edge_list:
                g.add_edges(edge_list)
                g.es["link_type"] = edge_types
                g.es["weight"] = edge_weights

        self._graph = g

    def resync_igraph(self) -> None:
        """Full re-sync from SQLite (called after dream engine, etc.)."""
        self._rebuild_igraph()

    # =========================================================================
    # Memory node CRUD
    # =========================================================================

    @staticmethod
    def _content_hash(content: str) -> str:
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def add_node(self, node: MemoryNode) -> str:
        """Add a memory node to SQLite and igraph."""
        content_hash = self._content_hash(node.content)
        meta = node.metadata
        strength = node.strength

        self.conn.execute(
            """INSERT INTO memory_nodes (
                id, content, content_hash, memory_type, layer, agent_id, visibility,
                stability, difficulty, access_count, access_timestamps_json,
                compressed_count, compressed_avg_interval,
                last_retrievability, last_activation, last_computed_at,
                valence, arousal, salience,
                episode_id, session_id, conversation_thread,
                tags_json, concepts_json, responding_to_json, related_agents_json,
                source, derived_from_json, metadata_json,
                created_at, last_accessed_at, promoted_at
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?,
                ?, ?, ?,
                ?, ?, ?,
                ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?,
                ?, ?, ?
            )""",
            (
                node.id, node.content, content_hash, meta.memory_type.value, meta.layer.value,
                meta.agent_id, meta.visibility.value,
                strength.stability, strength.difficulty, strength.access_count,
                json.dumps(strength.access_timestamps),
                strength.compressed_count, strength.compressed_avg_interval,
                strength.last_retrievability, strength.last_activation,
                strength.last_computed_at,
                meta.valence.value, meta.arousal, meta.salience,
                meta.episode_id, meta.session_id, meta.conversation_thread,
                json.dumps(meta.tags), json.dumps(meta.concepts),
                json.dumps(meta.responding_to), json.dumps(meta.related_agents),
                meta.source, json.dumps(meta.derived_from), "{}",
                node.created_at.isoformat(),
                node.last_accessed_at.isoformat() if node.last_accessed_at else None,
                node.promoted_at.isoformat() if node.promoted_at else None,
            ),
        )
        self.conn.commit()

        # Add to igraph
        idx = self.graph.vcount()
        self.graph.add_vertices(1)
        self.graph.vs[idx]["node_id"] = node.id
        self._id_to_vertex[node.id] = idx
        self._vertex_to_id[idx] = node.id

        return node.id

    def get_node(self, node_id: str) -> Optional[MemoryNode]:
        """Get a memory node from SQLite by ID."""
        row = self.conn.execute(
            "SELECT * FROM memory_nodes WHERE id = ?", (node_id,)
        ).fetchone()
        if not row:
            return None
        return self._row_to_memory_node(row)

    def delete_node(self, node_id: str) -> bool:
        """Delete a node from SQLite (cascade deletes links). Rebuild igraph."""
        cursor = self.conn.execute("DELETE FROM memory_nodes WHERE id = ?", (node_id,))
        self.conn.commit()
        if cursor.rowcount > 0:
            # igraph doesn't support efficient single-vertex deletion;
            # for occasional deletes, rebuild is simplest and safest
            self._rebuild_igraph()
            return True
        return False

    def update_node_strength(self, node_id: str, strength: StrengthState) -> bool:
        """Update only the strength parameters for a node."""
        cursor = self.conn.execute(
            """UPDATE memory_nodes SET
                stability = ?, difficulty = ?, access_count = ?,
                access_timestamps_json = ?,
                compressed_count = ?, compressed_avg_interval = ?,
                last_retrievability = ?, last_activation = ?,
                last_computed_at = ?, last_accessed_at = datetime('now')
            WHERE id = ?""",
            (
                strength.stability, strength.difficulty, strength.access_count,
                json.dumps(strength.access_timestamps),
                strength.compressed_count, strength.compressed_avg_interval,
                strength.last_retrievability, strength.last_activation,
                strength.last_computed_at,
                node_id,
            ),
        )
        self.conn.commit()
        return cursor.rowcount > 0

    def update_node_metadata(self, node_id: str, **kwargs) -> bool:
        """Update specific metadata fields for a node."""
        allowed = {
            "layer", "valence", "arousal", "salience", "episode_id",
            "session_id", "tags_json", "concepts_json", "promoted_at",
            "visibility",
        }
        updates = []
        values = []
        for key, val in kwargs.items():
            if key in allowed:
                updates.append(f"{key} = ?")
                values.append(val)
        if not updates:
            return False
        values.append(node_id)
        cursor = self.conn.execute(
            f"UPDATE memory_nodes SET {', '.join(updates)} WHERE id = ?",
            values,
        )
        self.conn.commit()
        return cursor.rowcount > 0

    def get_nodes_since(self, since: datetime) -> list[MemoryNode]:
        """Get all nodes created since a given datetime."""
        rows = self.conn.execute(
            "SELECT * FROM memory_nodes WHERE created_at >= ? ORDER BY created_at",
            (since.isoformat(),),
        ).fetchall()
        return [self._row_to_memory_node(r) for r in rows]

    def get_all_node_ids(self) -> list[str]:
        """Get all memory node IDs."""
        rows = self.conn.execute("SELECT id FROM memory_nodes").fetchall()
        return [r["id"] for r in rows]

    def count_nodes(self) -> int:
        """Count total memory nodes."""
        return self.conn.execute("SELECT COUNT(*) as c FROM memory_nodes").fetchone()["c"]

    def find_duplicate_content(self, content: str) -> Optional[str]:
        """Check if content already exists (by hash). Returns existing ID or None."""
        h = self._content_hash(content)
        row = self.conn.execute(
            "SELECT id FROM memory_nodes WHERE content_hash = ?", (h,)
        ).fetchone()
        return row["id"] if row else None

    # =========================================================================
    # Associative link CRUD
    # =========================================================================

    def add_link(self, link: AssociativeLink) -> str:
        """Add an associative link to SQLite and igraph."""
        try:
            self.conn.execute(
                """INSERT INTO associative_links (
                    id, source_id, target_id, link_type, weight,
                    activation_count, created_at, last_activated,
                    source_reason, evidence
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    link.id, link.source_id, link.target_id,
                    link.link_type.value, link.weight,
                    link.activation_count, link.created_at.isoformat(),
                    link.last_activated.isoformat() if link.last_activated else None,
                    link.source, link.evidence,
                ),
            )
            self.conn.commit()
        except sqlite3.IntegrityError:
            # Unique constraint: update weight instead
            self.conn.execute(
                """UPDATE associative_links SET weight = MAX(weight, ?),
                   last_activated = datetime('now'), activation_count = activation_count + 1
                WHERE source_id = ? AND target_id = ? AND link_type = ?""",
                (link.weight, link.source_id, link.target_id, link.link_type.value),
            )
            self.conn.commit()
            row = self.conn.execute(
                "SELECT id FROM associative_links WHERE source_id=? AND target_id=? AND link_type=?",
                (link.source_id, link.target_id, link.link_type.value),
            ).fetchone()
            if row:
                link.id = row["id"]

        # Add edge to igraph
        src_idx = self._id_to_vertex.get(link.source_id)
        tgt_idx = self._id_to_vertex.get(link.target_id)
        if src_idx is not None and tgt_idx is not None:
            # Check if edge already exists in igraph
            eid = self.graph.get_eid(src_idx, tgt_idx, error=False)
            if eid == -1:
                self.graph.add_edges([(src_idx, tgt_idx)])
                new_eid = self.graph.ecount() - 1
                self.graph.es[new_eid]["link_type"] = link.link_type.value
                self.graph.es[new_eid]["weight"] = link.weight
            else:
                self.graph.es[eid]["weight"] = max(self.graph.es[eid]["weight"], link.weight)

        return link.id

    def ensure_link(
        self,
        source_id: str,
        target_id: str,
        link_type: LinkType,
        weight: float = 0.5,
        source: str = "system",
        evidence: Optional[str] = None,
    ) -> str:
        """Create a link if it doesn't exist, or strengthen if it does."""
        link = AssociativeLink(
            source_id=source_id,
            target_id=target_id,
            link_type=link_type,
            weight=weight,
            source=source,
            evidence=evidence,
        )
        return self.add_link(link)

    def get_link(
        self, source_id: str, target_id: str, link_type: Optional[LinkType] = None
    ) -> Optional[AssociativeLink]:
        """Get a specific link between two nodes."""
        if link_type:
            row = self.conn.execute(
                "SELECT * FROM associative_links WHERE source_id=? AND target_id=? AND link_type=?",
                (source_id, target_id, link_type.value),
            ).fetchone()
        else:
            row = self.conn.execute(
                "SELECT * FROM associative_links WHERE source_id=? AND target_id=?",
                (source_id, target_id),
            ).fetchone()
        if not row:
            return None
        return self._row_to_link(row)

    def has_link(self, source_id: str, target_id: str) -> bool:
        """Check if any link exists between two nodes."""
        row = self.conn.execute(
            "SELECT 1 FROM associative_links WHERE source_id=? AND target_id=?",
            (source_id, target_id),
        ).fetchone()
        return row is not None

    def update_link_weight(self, link_id: str, weight: float) -> bool:
        """Update a link's weight."""
        cursor = self.conn.execute(
            "UPDATE associative_links SET weight = ?, last_activated = datetime('now'), "
            "activation_count = activation_count + 1 WHERE id = ?",
            (weight, link_id),
        )
        self.conn.commit()
        return cursor.rowcount > 0

    def strengthen_link(self, source_id: str, target_id: str, boost: float = 0.1) -> None:
        """Hebbian learning: strengthen a link that was traversed."""
        self.conn.execute(
            """UPDATE associative_links
            SET weight = MIN(weight + ?, 1.0),
                last_activated = datetime('now'),
                activation_count = activation_count + 1
            WHERE source_id = ? AND target_id = ?""",
            (boost, source_id, target_id),
        )
        self.conn.commit()

        # Update igraph too
        src_idx = self._id_to_vertex.get(source_id)
        tgt_idx = self._id_to_vertex.get(target_id)
        if src_idx is not None and tgt_idx is not None:
            eid = self.graph.get_eid(src_idx, tgt_idx, error=False)
            if eid != -1:
                self.graph.es[eid]["weight"] = min(self.graph.es[eid]["weight"] + boost, 1.0)

    def get_neighbors(
        self,
        node_id: str,
        link_types: Optional[list[LinkType]] = None,
        min_weight: float = 0.0,
        direction: str = "all",
    ) -> list[tuple[str, float, str]]:
        """Get neighbors of a node via igraph (fast C-speed).

        Returns: [(neighbor_id, weight, link_type), ...]
        """
        vertex_idx = self._id_to_vertex.get(node_id)
        if vertex_idx is None:
            return []

        mode_map = {"all": "all", "outgoing": "out", "incoming": "in"}
        mode = mode_map.get(direction, "all")

        neighbors = []
        try:
            edges = self.graph.incident(vertex_idx, mode=mode)
        except Exception:
            return []

        for eid in edges:
            edge = self.graph.es[eid]
            weight = edge["weight"]
            link_type = edge["link_type"]

            if weight < min_weight:
                continue
            if link_types and link_type not in [lt.value for lt in link_types]:
                continue

            # Determine neighbor
            source_v = edge.source
            target_v = edge.target
            neighbor_v = target_v if source_v == vertex_idx else source_v
            neighbor_id = self._vertex_to_id.get(neighbor_v)
            if neighbor_id:
                neighbors.append((neighbor_id, weight, link_type))

        return neighbors

    def get_degree(self, node_id: str) -> int:
        """Get the number of links for a node."""
        vertex_idx = self._id_to_vertex.get(node_id)
        if vertex_idx is None:
            return 0
        return self.graph.degree(vertex_idx, mode="all")

    def get_links_by_type(self, link_type: LinkType) -> list[AssociativeLink]:
        """Get all links of a given type from SQLite."""
        rows = self.conn.execute(
            "SELECT * FROM associative_links WHERE link_type = ?", (link_type.value,)
        ).fetchall()
        return [self._row_to_link(r) for r in rows]

    def count_links(self) -> int:
        """Count total associative links."""
        return self.conn.execute("SELECT COUNT(*) as c FROM associative_links").fetchone()["c"]

    # =========================================================================
    # Episode CRUD
    # =========================================================================

    def add_episode(self, episode: Episode) -> str:
        """Add an episode to SQLite."""
        self.conn.execute(
            """INSERT INTO episodes (
                id, title, agent_id, session_id,
                started_at, ended_at,
                overall_valence, peak_arousal, tags_json,
                consolidated, schema_extracted, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                episode.id, episode.title, episode.agent_id, episode.session_id,
                episode.started_at.isoformat() if episode.started_at else None,
                episode.ended_at.isoformat() if episode.ended_at else None,
                episode.overall_valence.value, episode.peak_arousal,
                json.dumps(episode.tags),
                int(episode.consolidated), int(episode.schema_extracted),
                episode.created_at.isoformat(),
            ),
        )
        self.conn.commit()
        return episode.id

    def add_episode_step(self, episode_id: str, step: EpisodeStep) -> None:
        """Add a step to an episode."""
        self.conn.execute(
            "INSERT INTO episode_steps (episode_id, memory_id, position, role, timestamp) "
            "VALUES (?, ?, ?, ?, ?)",
            (episode_id, step.memory_id, step.position, step.role, step.timestamp.isoformat()),
        )
        self.conn.commit()

    def get_episode(self, episode_id: str) -> Optional[Episode]:
        """Get an episode with all its steps."""
        row = self.conn.execute("SELECT * FROM episodes WHERE id = ?", (episode_id,)).fetchone()
        if not row:
            return None
        step_rows = self.conn.execute(
            "SELECT * FROM episode_steps WHERE episode_id = ? ORDER BY position",
            (episode_id,),
        ).fetchall()
        steps = [
            EpisodeStep(
                memory_id=s["memory_id"],
                position=s["position"],
                role=s["role"],
                timestamp=datetime.fromisoformat(s["timestamp"]),
            )
            for s in step_rows
        ]
        return Episode(
            id=row["id"],
            title=row["title"],
            agent_id=row["agent_id"],
            session_id=row["session_id"],
            started_at=datetime.fromisoformat(row["started_at"]) if row["started_at"] else None,
            ended_at=datetime.fromisoformat(row["ended_at"]) if row["ended_at"] else None,
            overall_valence=row["overall_valence"],
            peak_arousal=row["peak_arousal"],
            tags=json.loads(row["tags_json"]),
            consolidated=bool(row["consolidated"]),
            schema_extracted=bool(row["schema_extracted"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            steps=steps,
        )

    def get_unconsolidated_episodes(self) -> list[Episode]:
        """Get episodes not yet processed by the Dream Engine."""
        rows = self.conn.execute(
            "SELECT id FROM episodes WHERE consolidated = 0 ORDER BY created_at"
        ).fetchall()
        return [self.get_episode(r["id"]) for r in rows if self.get_episode(r["id"])]

    # =========================================================================
    # Agent CRUD
    # =========================================================================

    def register_agent(self, profile: AgentProfile) -> str:
        """Register or update an agent profile."""
        self.conn.execute(
            """INSERT OR REPLACE INTO agents (
                id, display_name, generation, lineage, specialization,
                origin_story, color, symbol, registered_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                profile.id, profile.display_name, profile.generation,
                profile.lineage, profile.specialization,
                profile.origin_story, profile.color, profile.symbol,
                profile.registered_at.isoformat(),
            ),
        )
        self.conn.commit()
        return profile.id

    def list_agents(self) -> list[AgentProfile]:
        """List all registered agents."""
        rows = self.conn.execute("SELECT * FROM agents ORDER BY registered_at").fetchall()
        return [
            AgentProfile(
                id=r["id"],
                display_name=r["display_name"],
                generation=r["generation"],
                lineage=r["lineage"] or "Unknown",
                specialization=r["specialization"] or "General",
                origin_story=r["origin_story"],
                color=r["color"] or "#888888",
                symbol=r["symbol"] or "A",
                registered_at=datetime.fromisoformat(r["registered_at"]),
            )
            for r in rows
        ]

    # =========================================================================
    # Dream log
    # =========================================================================

    def log_dream_phase(
        self, phase: str, memories_processed: int = 0,
        links_created: int = 0, links_strengthened: int = 0,
        memories_pruned: int = 0, schemas_extracted: int = 0,
        notes: Optional[str] = None, success: bool = True,
    ) -> None:
        """Log a dream engine phase execution."""
        self.conn.execute(
            """INSERT INTO dream_log (
                phase, started_at, completed_at, memories_processed,
                links_created, links_strengthened, memories_pruned,
                schemas_extracted, notes, success
            ) VALUES (?, datetime('now'), datetime('now'), ?, ?, ?, ?, ?, ?, ?)""",
            (phase, memories_processed, links_created, links_strengthened,
             memories_pruned, schemas_extracted, notes, int(success)),
        )
        self.conn.commit()

    # =========================================================================
    # Stats
    # =========================================================================

    def stats(self) -> dict:
        """Get comprehensive graph statistics."""
        node_count = self.count_nodes()
        link_count = self.count_links()

        type_counts = {}
        for row in self.conn.execute(
            "SELECT memory_type, COUNT(*) as c FROM memory_nodes GROUP BY memory_type"
        ).fetchall():
            type_counts[row["memory_type"]] = row["c"]

        layer_counts = {}
        for row in self.conn.execute(
            "SELECT layer, COUNT(*) as c FROM memory_nodes GROUP BY layer"
        ).fetchall():
            layer_counts[row["layer"]] = row["c"]

        link_type_counts = {}
        for row in self.conn.execute(
            "SELECT link_type, COUNT(*) as c FROM associative_links GROUP BY link_type"
        ).fetchall():
            link_type_counts[row["link_type"]] = row["c"]

        episode_count = self.conn.execute("SELECT COUNT(*) as c FROM episodes").fetchone()["c"]

        return {
            "nodes": node_count,
            "links": link_count,
            "episodes": episode_count,
            "memory_types": type_counts,
            "layers": layer_counts,
            "link_types": link_type_counts,
            "igraph_vertices": self.graph.vcount() if self._graph else 0,
            "igraph_edges": self.graph.ecount() if self._graph else 0,
        }

    # =========================================================================
    # Row mapping helpers
    # =========================================================================

    @staticmethod
    def _row_to_memory_node(row: sqlite3.Row) -> MemoryNode:
        """Convert a SQLite row to a MemoryNode."""
        return MemoryNode(
            id=row["id"],
            content=row["content"],
            metadata=MemoryMetadata(
                agent_id=row["agent_id"],
                visibility=row["visibility"],
                layer=MemoryLayer(row["layer"]),
                memory_type=MemoryType(row["memory_type"]),
                tags=json.loads(row["tags_json"]),
                concepts=json.loads(row["concepts_json"]),
                session_id=row["session_id"],
                conversation_thread=row["conversation_thread"],
                episode_id=row["episode_id"],
                responding_to=json.loads(row["responding_to_json"]),
                related_agents=json.loads(row["related_agents_json"]),
                valence=row["valence"],
                arousal=row["arousal"],
                salience=row["salience"],
                source=row["source"],
                derived_from=json.loads(row["derived_from_json"]),
            ),
            strength=StrengthState(
                stability=row["stability"],
                difficulty=row["difficulty"],
                access_count=row["access_count"],
                access_timestamps=json.loads(row["access_timestamps_json"]),
                compressed_count=row["compressed_count"],
                compressed_avg_interval=row["compressed_avg_interval"],
                last_retrievability=row["last_retrievability"],
                last_activation=row["last_activation"],
                last_computed_at=row["last_computed_at"],
            ),
            created_at=datetime.fromisoformat(row["created_at"]),
            last_accessed_at=(
                datetime.fromisoformat(row["last_accessed_at"])
                if row["last_accessed_at"] else None
            ),
            promoted_at=(
                datetime.fromisoformat(row["promoted_at"])
                if row["promoted_at"] else None
            ),
        )

    @staticmethod
    def _row_to_link(row: sqlite3.Row) -> AssociativeLink:
        """Convert a SQLite row to an AssociativeLink."""
        return AssociativeLink(
            id=row["id"],
            source_id=row["source_id"],
            target_id=row["target_id"],
            link_type=LinkType(row["link_type"]),
            weight=row["weight"],
            activation_count=row["activation_count"],
            created_at=datetime.fromisoformat(row["created_at"]),
            last_activated=(
                datetime.fromisoformat(row["last_activated"])
                if row["last_activated"] else None
            ),
            source=row["source_reason"],
            evidence=row["evidence"],
        )
