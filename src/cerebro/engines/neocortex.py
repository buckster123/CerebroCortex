"""SchemaEngine - Schema formation and abstraction.

The neocortex: extracts abstract schemas and generalizations from
episodic memories. Schemas represent learned patterns like
"user projects follow: prototype -> iterate -> polish".

Responsibilities:
- Extract schemas from consolidated episodes
- Create derived_from links between schemas and source episodes
- Validate new experiences against existing schemas
- Schema evolution (update when new evidence arrives)
"""

import json
import logging
from datetime import datetime
from typing import Optional

from cerebro.config import (
    SCHEMA_DEMOTE_MAX_IDLE_CYCLES,
    SCHEMA_PROMOTE_MIN_ACCESSES,
    SCHEMA_PROMOTE_MIN_SUPPORTS,
)
from cerebro.models.memory import MemoryMetadata, MemoryNode, StrengthState
from cerebro.storage.graph_store import GraphStore
from cerebro.types import LinkType, MemoryLayer, MemoryType

logger = logging.getLogger("cerebro-schema")


class SchemaEngine:
    """Extracts and manages schematic memories (abstract patterns)."""

    def __init__(self, graph: GraphStore, vector_store=None):
        self._graph = graph
        self._vector = vector_store

    def create_schema(
        self,
        content: str,
        source_ids: list[str],
        tags: Optional[list[str]] = None,
        agent_id: str = "CLAUDE",
    ) -> MemoryNode:
        """Create a new schematic memory derived from source memories.

        Schemas are high-level abstractions extracted from multiple episodes
        or semantic memories. They start in long_term with high stability.

        Args:
            content: The schema/pattern description
            source_ids: Memory IDs this schema was derived from
            tags: Tags for categorization
            agent_id: Agent that created this schema

        Returns:
            The created schema MemoryNode
        """
        schema_tags = list(tags or [])
        # Track support count for promotion gating
        if not any(t.startswith("support_count:") for t in schema_tags):
            schema_tags.append("support_count:0")
        if not any(t.startswith("dream_cycles_idle:") for t in schema_tags):
            schema_tags.append("dream_cycles_idle:0")

        node = MemoryNode(
            content=content,
            metadata=MemoryMetadata(
                memory_type=MemoryType.SCHEMATIC,
                layer=MemoryLayer.WORKING,  # start in WORKING; earn promotion
                tags=schema_tags,
                agent_id=agent_id,
                salience=0.6,  # moderate until validated
                source="consolidation",
                derived_from=source_ids,
            ),
            strength=StrengthState(stability=7.0),  # 1 week; promoted schemas get 30d
        )

        self._graph.add_node(node)

        # Dual-write to vector store
        if self._vector:
            from cerebro.cortex import CerebroCortex
            coll = CerebroCortex._collection_for_type(node.metadata.memory_type)
            self._vector.add_node(coll, node)

        # Link to source memories
        for source_id in source_ids:
            if self._graph.get_node(source_id):
                self._graph.ensure_link(
                    node.id, source_id, LinkType.DERIVED_FROM,
                    weight=0.8, source="dream_pattern",
                    evidence="Schema derived from this memory",
                )

        return node

    def find_matching_schemas(
        self,
        tags: Optional[list[str]] = None,
        concepts: Optional[list[str]] = None,
        max_results: int = 5,
        agent_id: Optional[str] = None,
        conversation_thread: Optional[str] = None,
    ) -> list[MemoryNode]:
        """Find schemas matching given tags or concepts."""
        from cerebro.cortex import _scope_sql
        scope_clause, scope_params = _scope_sql(agent_id, conversation_thread)

        results = []
        seen_ids: set[str] = set()

        if tags:
            for tag in tags:
                rows = self._graph.conn.execute(
                    f"""SELECT id FROM memory_nodes
                    WHERE memory_type = 'schematic'
                    AND tags_json LIKE ?{scope_clause}
                    ORDER BY salience DESC
                    LIMIT ?""",
                    (f'%"{tag}"%', *scope_params, max_results),
                ).fetchall()
                for row in rows:
                    if row["id"] not in seen_ids:
                        node = self._graph.get_node(row["id"])
                        if node:
                            results.append(node)
                            seen_ids.add(row["id"])

        if concepts:
            for concept in concepts:
                rows = self._graph.conn.execute(
                    f"""SELECT id FROM memory_nodes
                    WHERE memory_type = 'schematic'
                    AND concepts_json LIKE ?{scope_clause}
                    ORDER BY salience DESC
                    LIMIT ?""",
                    (f'%"{concept}"%', *scope_params, max_results),
                ).fetchall()
                for row in rows:
                    if row["id"] not in seen_ids:
                        node = self._graph.get_node(row["id"])
                        if node:
                            results.append(node)
                            seen_ids.add(row["id"])

        return results[:max_results]

    def get_schema_sources(self, schema_id: str) -> list[str]:
        """Get the source memory IDs that a schema was derived from."""
        node = self._graph.get_node(schema_id)
        if not node:
            return []
        return node.metadata.derived_from

    def reinforce_schema(
        self,
        schema_id: str,
        supporting_id: str,
    ) -> bool:
        """Reinforce a schema with new supporting evidence.

        When a new experience matches an existing schema, strengthen the
        schema's stability and create a supports link. Also increments
        the support_count tracking tag.
        """
        node = self._graph.get_node(schema_id)
        if not node:
            return False

        # Create supports link
        self._graph.ensure_link(
            supporting_id, schema_id, LinkType.SUPPORTS,
            weight=0.7, source="encoding",
            evidence="New evidence supporting this schema",
        )

        # Increment support_count tag
        tags = list(node.metadata.tags)
        count = self._get_tag_int(tags, "support_count", 0) + 1
        tags = self._set_tag_int(tags, "support_count", count)
        self._graph.update_node_metadata(schema_id, tags_json=json.dumps(tags))

        # Boost schema salience slightly
        new_salience = min(1.0, node.metadata.salience + 0.05)
        self._graph.update_node_metadata(schema_id, salience=new_salience)

        return True

    def get_all_schemas(
        self,
        agent_id: Optional[str] = None,
        conversation_thread: Optional[str] = None,
    ) -> list[MemoryNode]:
        """Get all schematic memories."""
        from cerebro.cortex import _scope_sql
        scope_clause, scope_params = _scope_sql(agent_id, conversation_thread)

        query = f"SELECT id FROM memory_nodes WHERE memory_type = 'schematic'{scope_clause}"
        query += " ORDER BY salience DESC"

        rows = self._graph.conn.execute(query, scope_params).fetchall()
        return [n for r in rows if (n := self._graph.get_node(r["id"]))]

    def count_schemas(self) -> int:
        """Count total schematic memories."""
        row = self._graph.conn.execute(
            "SELECT COUNT(*) as c FROM memory_nodes WHERE memory_type = 'schematic'"
        ).fetchone()
        return row["c"]

    # =========================================================================
    # Schema validation & promotion
    # =========================================================================

    def evaluate_schema_candidates(self) -> dict[str, int]:
        """Evaluate WORKING-layer schemas for promotion or demotion.

        Promotion: support_count >= SCHEMA_PROMOTE_MIN_SUPPORTS AND
                   access_count >= SCHEMA_PROMOTE_MIN_ACCESSES
                   → promote to LONG_TERM, boost salience + stability.

        Demotion:  access_count == 0 after SCHEMA_DEMOTE_MAX_IDLE_CYCLES
                   dream cycles → delete the schema.

        Called from Dream Engine Phase 3.

        Returns:
            Dict with counts: {"promoted": N, "demoted": N}
        """
        promoted = 0
        demoted = 0

        rows = self._graph.conn.execute(
            "SELECT id FROM memory_nodes WHERE memory_type = 'schematic'"
        ).fetchall()

        for row in rows:
            node = self._graph.get_node(row["id"])
            if not node:
                continue

            tags = list(node.metadata.tags)
            support_count = self._get_tag_int(tags, "support_count", 0)
            idle_cycles = self._get_tag_int(tags, "dream_cycles_idle", 0)

            if node.metadata.layer == MemoryLayer.WORKING:
                # Check for promotion
                if (support_count >= SCHEMA_PROMOTE_MIN_SUPPORTS
                        and node.strength.access_count >= SCHEMA_PROMOTE_MIN_ACCESSES):
                    # Promote to LONG_TERM
                    self._graph.update_node_metadata(
                        row["id"],
                        layer=MemoryLayer.LONG_TERM.value,
                        salience=min(1.0, node.metadata.salience + 0.2),
                    )
                    # Boost stability to 30 days
                    self._graph.conn.execute(
                        "UPDATE memory_nodes SET stability = 30.0, promoted_at = ? WHERE id = ?",
                        (datetime.now().isoformat(), row["id"]),
                    )
                    self._graph.conn.commit()
                    promoted += 1
                    logger.info(f"Schema promoted to LONG_TERM: {row['id'][:12]}... "
                                f"(supports={support_count}, accesses={node.strength.access_count})")

                # Check for demotion (idle too long)
                elif node.strength.access_count == 0:
                    new_idle = idle_cycles + 1
                    if new_idle >= SCHEMA_DEMOTE_MAX_IDLE_CYCLES:
                        # Prune: no one ever accessed this schema
                        self._graph.delete_node(row["id"])
                        demoted += 1
                        logger.info(f"Schema pruned (idle {new_idle} cycles): {row['id'][:12]}...")
                    else:
                        # Increment idle counter
                        tags = self._set_tag_int(tags, "dream_cycles_idle", new_idle)
                        self._graph.update_node_metadata(row["id"], tags_json=json.dumps(tags))
                else:
                    # Has accesses but not enough supports yet — reset idle counter
                    if idle_cycles > 0:
                        tags = self._set_tag_int(tags, "dream_cycles_idle", 0)
                        self._graph.update_node_metadata(row["id"], tags_json=json.dumps(tags))

        return {"promoted": promoted, "demoted": demoted}

    # =========================================================================
    # Tag helpers for integer tracking tags (e.g. "support_count:3")
    # =========================================================================

    @staticmethod
    def _get_tag_int(tags: list[str], prefix: str, default: int = 0) -> int:
        """Extract integer from a 'prefix:N' tag."""
        for tag in tags:
            if tag.startswith(f"{prefix}:"):
                try:
                    return int(tag.split(":", 1)[1])
                except (ValueError, IndexError):
                    pass
        return default

    @staticmethod
    def _set_tag_int(tags: list[str], prefix: str, value: int) -> list[str]:
        """Set a 'prefix:N' tag, replacing existing if present."""
        new_tags = [t for t in tags if not t.startswith(f"{prefix}:")]
        new_tags.append(f"{prefix}:{value}")
        return new_tags
