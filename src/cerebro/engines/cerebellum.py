"""ProceduralEngine - Procedural memory management.

The cerebellum: manages strategies, workflows, and learned patterns.
Procedural memories are "how to do X" knowledge extracted from episodes.

Responsibilities:
- Store and retrieve procedures/strategies
- Track success/failure of procedures
- Identify procedural patterns from episodes
- Suggest relevant procedures for current context
"""

import json
from typing import Optional

from cerebro.models.memory import MemoryMetadata, MemoryNode, StrengthState
from cerebro.storage.graph_store import GraphStore
from cerebro.types import LinkType, MemoryType


class ProceduralEngine:
    """Manages procedural memory: strategies, workflows, and patterns."""

    def __init__(self, graph: GraphStore, vector_store=None):
        self._graph = graph
        self._vector = vector_store

    def store_procedure(
        self,
        content: str,
        tags: Optional[list[str]] = None,
        derived_from: Optional[list[str]] = None,
        agent_id: str = "CLAUDE",
    ) -> MemoryNode:
        """Store a new procedural memory.

        Args:
            content: The procedure/strategy/workflow text
            tags: Tags for categorization
            derived_from: Memory IDs this procedure was extracted from
            agent_id: Agent that created this procedure

        Returns:
            The created MemoryNode
        """
        node = MemoryNode(
            content=content,
            metadata=MemoryMetadata(
                memory_type=MemoryType.PROCEDURAL,
                tags=tags or [],
                agent_id=agent_id,
                salience=0.8,  # procedures are valuable by default
                source="consolidation" if derived_from else "user_input",
                derived_from=derived_from or [],
            ),
            strength=StrengthState(stability=3.0),  # procedures start more stable
        )

        self._graph.add_node(node)

        # Dual-write to vector store
        if self._vector:
            from cerebro.cortex import CerebroCortex
            coll = CerebroCortex._collection_for_type(node.metadata.memory_type)
            self._vector.add_node(coll, node)

        # Link to source memories
        if derived_from:
            for parent_id in derived_from:
                if self._graph.get_node(parent_id):
                    self._graph.ensure_link(
                        node.id, parent_id, LinkType.DERIVED_FROM,
                        weight=0.8, source="consolidation",
                        evidence="Procedure extracted from this memory",
                    )

        return node

    def find_relevant_procedures(
        self,
        tags: Optional[list[str]] = None,
        concepts: Optional[list[str]] = None,
        max_results: int = 5,
        agent_id: Optional[str] = None,
        conversation_thread: Optional[str] = None,
    ) -> list[MemoryNode]:
        """Find procedures relevant to the given context.

        Searches by tags and concepts in the graph store.
        """
        from cerebro.cortex import _scope_sql
        scope_clause, scope_params = _scope_sql(agent_id, conversation_thread)

        results = []

        if tags:
            for tag in tags:
                rows = self._graph.conn.execute(
                    f"""SELECT id FROM memory_nodes
                    WHERE memory_type = 'procedural'
                    AND tags_json LIKE ?{scope_clause}
                    ORDER BY salience DESC
                    LIMIT ?""",
                    (f'%"{tag}"%', *scope_params, max_results),
                ).fetchall()
                for row in rows:
                    node = self._graph.get_node(row["id"])
                    if node and node.id not in {r.id for r in results}:
                        results.append(node)

        if concepts:
            for concept in concepts:
                rows = self._graph.conn.execute(
                    f"""SELECT id FROM memory_nodes
                    WHERE memory_type = 'procedural'
                    AND concepts_json LIKE ?{scope_clause}
                    ORDER BY salience DESC
                    LIMIT ?""",
                    (f'%"{concept}"%', *scope_params, max_results),
                ).fetchall()
                for row in rows:
                    node = self._graph.get_node(row["id"])
                    if node and node.id not in {r.id for r in results}:
                        results.append(node)

        return results[:max_results]

    def record_outcome(
        self,
        procedure_id: str,
        success: bool,
        salience_boost: float = 0.1,
    ) -> bool:
        """Record whether a procedure succeeded or failed.

        Successful procedures get salience boost.
        Failed procedures also get salience boost (learn from mistakes)
        but also get higher difficulty (harder to apply correctly).

        Args:
            procedure_id: The procedural memory ID
            success: Whether the procedure worked
            salience_boost: Additional salience to add

        Returns:
            True if the procedure was found and updated
        """
        node = self._graph.get_node(procedure_id)
        if not node:
            return False

        # Both success and failure increase salience (important either way)
        new_salience = min(1.0, node.metadata.salience + salience_boost)
        self._graph.update_node_metadata(procedure_id, salience=new_salience)

        if not success:
            # Failed procedures get higher difficulty (harder to apply)
            new_difficulty = min(10.0, node.strength.difficulty + 0.5)
            updated_strength = StrengthState(
                **{**node.strength.model_dump(), "difficulty": new_difficulty}
            )
            self._graph.update_node_strength(procedure_id, updated_strength)

        return True

    def get_all_procedures(
        self,
        agent_id: Optional[str] = None,
        min_salience: float = 0.0,
        conversation_thread: Optional[str] = None,
    ) -> list[MemoryNode]:
        """Get all procedural memories, optionally filtered."""
        from cerebro.cortex import _scope_sql
        scope_clause, scope_params = _scope_sql(agent_id, conversation_thread)

        query = f"SELECT id FROM memory_nodes WHERE memory_type = 'procedural'{scope_clause}"
        params: list = list(scope_params)

        if min_salience > 0:
            query += " AND salience >= ?"
            params.append(min_salience)

        query += " ORDER BY salience DESC"

        rows = self._graph.conn.execute(query, params).fetchall()
        return [n for r in rows if (n := self._graph.get_node(r["id"]))]

    def link_to_related_procedures(
        self,
        procedure_id: str,
        max_links: int = 3,
    ) -> list[str]:
        """Link a procedure to related procedures via shared tags."""
        node = self._graph.get_node(procedure_id)
        if not node or not node.metadata.tags:
            return []

        created = []
        for tag in node.metadata.tags:
            rows = self._graph.conn.execute(
                """SELECT id FROM memory_nodes
                WHERE memory_type = 'procedural'
                AND id != ?
                AND tags_json LIKE ?
                LIMIT ?""",
                (procedure_id, f'%"{tag}"%', max_links),
            ).fetchall()

            for row in rows:
                if not self._graph.has_link(procedure_id, row["id"]):
                    link_id = self._graph.ensure_link(
                        procedure_id, row["id"], LinkType.SUPPORTS,
                        weight=0.5, source="encoding",
                        evidence=f"Related procedures sharing tag: {tag}",
                    )
                    created.append(link_id)

        return created
