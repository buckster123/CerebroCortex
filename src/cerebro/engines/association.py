"""LinkEngine - Associative network management.

The association cortex: creates, strengthens, and traverses the
typed/weighted links that form the associative memory network.

Responsibilities:
- Auto-link new memories based on shared tags, concepts, context
- Hebbian strengthening of co-activated links during recall
- Contextual link creation (same session/episode/thread)
- Link inference (if A->B and B->C, consider A->C)
"""

from datetime import datetime
from typing import Optional

from cerebro.activation.spreading import spreading_activation
from cerebro.types import Visibility
from cerebro.config import LINK_TYPE_WEIGHTS
from cerebro.models.link import AssociativeLink
from cerebro.models.memory import MemoryNode
from cerebro.storage.graph_store import GraphStore
from cerebro.types import LinkType


class LinkEngine:
    """Manages the associative network between memories."""

    def __init__(self, graph: GraphStore):
        self._graph = graph

    # =========================================================================
    # Link creation
    # =========================================================================

    def create_link(
        self,
        source_id: str,
        target_id: str,
        link_type: LinkType,
        weight: float = 0.5,
        source: str = "system",
        evidence: Optional[str] = None,
    ) -> str:
        """Create or strengthen a link between two memories."""
        return self._graph.ensure_link(
            source_id, target_id, link_type,
            weight=weight, source=source, evidence=evidence,
        )

    def auto_link_on_store(self, node: MemoryNode, context_ids: Optional[list[str]] = None) -> list[str]:
        """Auto-create links when a new memory is stored.

        Creates links based on:
        - Shared tags/concepts with existing memories
        - Session/episode context
        - responding_to relationships
        - derived_from relationships

        Args:
            node: The newly stored memory node
            context_ids: IDs of memories that were active in the current context

        Returns:
            List of created link IDs
        """
        created = []

        # 1. Link to memories referenced in responding_to
        for ref_id in node.metadata.responding_to:
            if self._graph.get_node(ref_id):
                link_id = self._graph.ensure_link(
                    node.id, ref_id, LinkType.CONTEXTUAL,
                    weight=0.7, source="encoding",
                    evidence="Memory responds to this",
                )
                created.append(link_id)

        # 2. Link to derived_from parents
        for parent_id in node.metadata.derived_from:
            if self._graph.get_node(parent_id):
                link_id = self._graph.ensure_link(
                    node.id, parent_id, LinkType.DERIVED_FROM,
                    weight=0.8, source="encoding",
                    evidence="Derived from parent memory",
                )
                created.append(link_id)

        # 3. Link to context memories (co-activated during this store)
        if context_ids:
            for ctx_id in context_ids:
                if ctx_id != node.id and self._graph.get_node(ctx_id):
                    link_id = self._graph.ensure_link(
                        node.id, ctx_id, LinkType.CONTEXTUAL,
                        weight=0.5, source="encoding",
                        evidence="Co-active during encoding",
                    )
                    created.append(link_id)

        # 4. Link to memories with shared tags (lightweight, no vector search)
        if node.metadata.tags:
            created.extend(self._link_by_shared_tags(node))

        return created

    def _link_by_shared_tags(self, node: MemoryNode, max_links: int = 5) -> list[str]:
        """Find and link memories sharing tags with the given node.

        Only links to memories the owning agent can access (scope-aware).
        """
        created = []
        tags = set(node.metadata.tags)
        if not tags:
            return created

        # Build scope filter to prevent cross-agent PRIVATE links
        from cerebro.cortex import _scope_sql
        scope_clause, scope_params = _scope_sql(node.metadata.agent_id)

        # Query SQLite for memories sharing any tag
        rows = self._graph.conn.execute(
            f"""SELECT DISTINCT id, tags_json FROM memory_nodes
            WHERE id != ? AND id IN (
                SELECT id FROM memory_nodes
                WHERE tags_json LIKE '%' || ? || '%'
                {('OR tags_json LIKE ' + "'%' || ? || '%' ") * (len(tags) - 1) if len(tags) > 1 else ''}
            ){scope_clause}
            LIMIT ?""",
            (node.id, *tags, *scope_params, max_links),
        ).fetchall()

        import json
        for row in rows:
            other_tags = set(json.loads(row["tags_json"]))
            overlap = tags & other_tags
            if overlap:
                weight = min(0.3 + 0.1 * len(overlap), 0.8)
                link_id = self._graph.ensure_link(
                    node.id, row["id"], LinkType.SEMANTIC,
                    weight=weight, source="encoding",
                    evidence=f"Shared tags: {', '.join(overlap)}",
                )
                created.append(link_id)

        return created

    # =========================================================================
    # Hebbian learning
    # =========================================================================

    def strengthen_co_activated(
        self,
        activated_ids: list[str],
        boost: float = 0.05,
    ) -> int:
        """Hebbian learning: strengthen links between co-activated memories.

        Called after a recall operation to strengthen the paths that
        were traversed during spreading activation.

        Args:
            activated_ids: Memory IDs that were co-activated in this recall
            boost: Weight increase per co-activation

        Returns:
            Number of links strengthened
        """
        strengthened = 0
        for i, src_id in enumerate(activated_ids):
            for tgt_id in activated_ids[i + 1:]:
                if self._graph.has_link(src_id, tgt_id):
                    self._graph.strengthen_link(src_id, tgt_id, boost)
                    strengthened += 1
                elif self._graph.has_link(tgt_id, src_id):
                    self._graph.strengthen_link(tgt_id, src_id, boost)
                    strengthened += 1
        return strengthened

    # =========================================================================
    # Spreading activation (delegated to activation module)
    # =========================================================================

    def spread_activation(
        self,
        seed_ids: list[str],
        seed_weights: list[float],
        max_hops: int = 2,
        max_activated: int = 50,
        agent_id: Optional[str] = None,
        conversation_thread: Optional[str] = None,
    ) -> dict[str, float]:
        """Spread activation from seeds through the associative network.

        Thin wrapper over the spreading activation algorithm, using the
        graph store's igraph instance.
        """
        return spreading_activation(
            self._graph,
            seed_ids=seed_ids,
            seed_weights=seed_weights,
            max_hops=max_hops,
            max_activated=max_activated,
            agent_id=agent_id,
            conversation_thread=conversation_thread,
        )

    # =========================================================================
    # Queries
    # =========================================================================

    def get_neighbors(
        self,
        node_id: str,
        link_types: Optional[list[LinkType]] = None,
        min_weight: float = 0.0,
    ) -> list[tuple[str, float, str]]:
        """Get neighbors of a memory with optional filtering."""
        return self._graph.get_neighbors(
            node_id, link_types=link_types, min_weight=min_weight,
        )

    def get_link_count(self, node_id: str) -> int:
        """Get the total number of links for a memory."""
        return self._graph.get_degree(node_id)

    def get_strongest_connections(
        self,
        node_id: str,
        top_n: int = 10,
    ) -> list[tuple[str, float, str]]:
        """Get the N strongest connections for a memory."""
        neighbors = self._graph.get_neighbors(node_id)
        neighbors.sort(key=lambda x: x[1], reverse=True)
        return neighbors[:top_n]

    def find_path(self, source_id: str, target_id: str) -> Optional[list[str]]:
        """Find shortest path between two memories in the graph.

        Returns:
            List of memory IDs forming the path, or None if no path exists.
        """
        src_idx = self._graph._id_to_vertex.get(source_id)
        tgt_idx = self._graph._id_to_vertex.get(target_id)
        if src_idx is None or tgt_idx is None:
            return None

        try:
            path = self._graph.graph.get_shortest_path(src_idx, tgt_idx, mode="all")
            if not path:
                return None
            return [self._graph._vertex_to_id[v] for v in path]
        except Exception:
            return None

    def get_common_neighbors(self, id_a: str, id_b: str) -> list[str]:
        """Find memories connected to both A and B."""
        neighbors_a = {n[0] for n in self._graph.get_neighbors(id_a)}
        neighbors_b = {n[0] for n in self._graph.get_neighbors(id_b)}
        return list(neighbors_a & neighbors_b)
