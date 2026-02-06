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
from datetime import datetime
from typing import Optional

from cerebro.models.memory import MemoryMetadata, MemoryNode, StrengthState
from cerebro.storage.graph_store import GraphStore
from cerebro.types import LinkType, MemoryLayer, MemoryType


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
        node = MemoryNode(
            content=content,
            metadata=MemoryMetadata(
                memory_type=MemoryType.SCHEMATIC,
                layer=MemoryLayer.LONG_TERM,
                tags=tags or [],
                agent_id=agent_id,
                salience=0.9,  # schemas are high value
                source="consolidation",
                derived_from=source_ids,
            ),
            strength=StrengthState(stability=30.0),  # schemas are stable
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
    ) -> list[MemoryNode]:
        """Find schemas matching given tags or concepts."""
        results = []
        seen_ids: set[str] = set()

        if tags:
            for tag in tags:
                rows = self._graph.conn.execute(
                    """SELECT id FROM memory_nodes
                    WHERE memory_type = 'schematic'
                    AND tags_json LIKE ?
                    ORDER BY salience DESC
                    LIMIT ?""",
                    (f'%"{tag}"%', max_results),
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
                    """SELECT id FROM memory_nodes
                    WHERE memory_type = 'schematic'
                    AND concepts_json LIKE ?
                    ORDER BY salience DESC
                    LIMIT ?""",
                    (f'%"{concept}"%', max_results),
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
        schema's stability and create a supports link.
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

        # Boost schema salience slightly
        new_salience = min(1.0, node.metadata.salience + 0.05)
        self._graph.update_node_metadata(schema_id, salience=new_salience)

        return True

    def get_all_schemas(
        self,
        agent_id: Optional[str] = None,
    ) -> list[MemoryNode]:
        """Get all schematic memories."""
        query = "SELECT id FROM memory_nodes WHERE memory_type = 'schematic'"
        params: list = []

        if agent_id:
            query += " AND agent_id = ?"
            params.append(agent_id)

        query += " ORDER BY salience DESC"

        rows = self._graph.conn.execute(query, params).fetchall()
        return [n for r in rows if (n := self._graph.get_node(r["id"]))]

    def count_schemas(self) -> int:
        """Count total schematic memories."""
        row = self._graph.conn.execute(
            "SELECT COUNT(*) as c FROM memory_nodes WHERE memory_type = 'schematic'"
        ).fetchone()
        return row["c"]
