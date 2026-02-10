"""ExecutiveEngine - Executive control and planning.

The prefrontal cortex: manages prospective memories, working memory,
prioritization, and layer promotion decisions.

Responsibilities:
- Prospective memory management (intentions, TODOs, deferred plans)
- Working memory management (what's currently relevant)
- Layer promotion/demotion decisions
- Priority ranking for recall results
"""

import time
from datetime import datetime
from typing import Optional

from cerebro.activation.decay import (
    apply_decay_tick,
    check_promotion_eligibility,
    compute_current_activation,
    compute_current_retrievability,
)
from cerebro.activation.strength import combined_recall_score
from cerebro.models.memory import MemoryMetadata, MemoryNode, StrengthState
from cerebro.storage.graph_store import GraphStore
from cerebro.types import MemoryLayer, MemoryType


class ExecutiveEngine:
    """Executive control: priorities, promotions, and prospective memory."""

    def __init__(self, graph: GraphStore, vector_store=None):
        self._graph = graph
        self._vector = vector_store

    # =========================================================================
    # Prospective memory (intentions, TODOs)
    # =========================================================================

    def store_intention(
        self,
        content: str,
        tags: Optional[list[str]] = None,
        agent_id: str = "CLAUDE",
        salience: float = 0.7,
    ) -> MemoryNode:
        """Store a prospective memory (future intention / TODO).

        Prospective memories start in working memory with high salience
        to ensure they surface when relevant.
        """
        node = MemoryNode(
            content=content,
            metadata=MemoryMetadata(
                memory_type=MemoryType.PROSPECTIVE,
                layer=MemoryLayer.WORKING,
                tags=tags or [],
                agent_id=agent_id,
                salience=salience,
                source="user_input",
            ),
            strength=StrengthState(stability=5.0),  # TODOs need to stick around
        )
        self._graph.add_node(node)

        # Dual-write to vector store
        if self._vector:
            from cerebro.cortex import CerebroCortex
            coll = CerebroCortex._collection_for_type(node.metadata.memory_type)
            self._vector.add_node(coll, node)

        return node

    def get_pending_intentions(
        self,
        agent_id: Optional[str] = None,
        min_salience: float = 0.3,
        conversation_thread: Optional[str] = None,
    ) -> list[MemoryNode]:
        """Get all pending prospective memories, sorted by salience."""
        from cerebro.cortex import _scope_sql
        scope_clause, scope_params = _scope_sql(agent_id, conversation_thread)

        query = f"""SELECT id FROM memory_nodes
            WHERE memory_type = 'prospective' AND salience >= ?{scope_clause}"""
        params: list = [min_salience, *scope_params]

        query += " ORDER BY salience DESC"

        rows = self._graph.conn.execute(query, params).fetchall()
        return [n for r in rows if (n := self._graph.get_node(r["id"]))]

    def resolve_intention(self, node_id: str) -> bool:
        """Mark a prospective memory as resolved by lowering its salience."""
        return self._graph.update_node_metadata(node_id, salience=0.1)

    # =========================================================================
    # Layer promotion
    # =========================================================================

    def check_and_promote(self, node_id: str) -> Optional[str]:
        """Check if a memory should be promoted and do it.

        Returns:
            New layer name if promoted, None otherwise.
        """
        node = self._graph.get_node(node_id)
        if not node:
            return None

        eligible, target = check_promotion_eligibility(node)
        if not eligible or target is None:
            return None

        self._graph.update_node_metadata(
            node_id,
            layer=target,
            promoted_at=datetime.now().isoformat(),
        )
        return target

    def run_promotion_sweep(self) -> dict[str, int]:
        """Check all memories for promotion eligibility.

        Returns:
            Dict of {layer: count} of promotions made.
        """
        promotions: dict[str, int] = {}
        all_ids = self._graph.get_all_node_ids()

        for node_id in all_ids:
            new_layer = self.check_and_promote(node_id)
            if new_layer:
                promotions[new_layer] = promotions.get(new_layer, 0) + 1

        return promotions

    # =========================================================================
    # Ranking and scoring
    # =========================================================================

    def rank_results(
        self,
        memory_ids: list[str],
        vector_similarities: Optional[dict[str, float]] = None,
        associative_scores: Optional[dict[str, float]] = None,
        explain: bool = False,
    ) -> list[tuple]:
        """Rank a set of memory IDs by combined recall score.

        Args:
            memory_ids: Candidate memory IDs to rank
            vector_similarities: Optional vector similarity scores per ID
            associative_scores: Optional spreading activation scores per ID
            explain: If True, include score breakdown dicts in results

        Returns:
            Sorted list of (memory_id, score) or (memory_id, score, explanation)
            from highest to lowest.
        """
        now = time.time()
        scored = []

        for mid in memory_ids:
            node = self._graph.get_node(mid)
            if not node:
                continue

            vector_sim = (vector_similarities or {}).get(mid, 0.0)
            assoc = (associative_scores or {}).get(mid, 0.0)

            # Compute current activation and retrievability
            base_level = compute_current_activation(node.strength, now)
            r = compute_current_retrievability(node.strength, now)

            from cerebro.activation.strength import recall_probability
            activation_score = recall_probability(base_level + assoc)

            score = combined_recall_score(
                vector_similarity=vector_sim,
                base_level=base_level,
                associative=assoc,
                fsrs_retrievability=r,
                salience=node.metadata.salience,
            )

            if explain:
                explanation = {
                    "memory_id": mid,
                    "composite_score": round(score, 4),
                    "vector_similarity": round(vector_sim, 4),
                    "actr_base_level": round(base_level, 4) if base_level != float("-inf") else None,
                    "actr_activation_score": round(activation_score, 4),
                    "spreading_activation": round(assoc, 4),
                    "fsrs_retrievability": round(r, 4),
                    "fsrs_stability_days": round(node.strength.stability, 2),
                    "salience": round(node.metadata.salience, 4),
                    "layer": node.metadata.layer.value,
                    "access_count": node.strength.access_count,
                    "age_hours": round((now - node.created_at.timestamp()) / 3600, 1),
                }
                scored.append((mid, score, explanation))
            else:
                scored.append((mid, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    # =========================================================================
    # Decay sweep
    # =========================================================================

    def run_decay_sweep(self) -> int:
        """Recompute cached activation/retrievability for all memories.

        Typically called during dream consolidation or periodically.

        Returns:
            Number of memories updated.
        """
        now = time.time()
        all_ids = self._graph.get_all_node_ids()
        updated = 0

        for node_id in all_ids:
            node = self._graph.get_node(node_id)
            if not node:
                continue

            new_strength = apply_decay_tick(node.strength, now)
            if (new_strength.last_retrievability != node.strength.last_retrievability or
                    new_strength.last_activation != node.strength.last_activation):
                self._graph.update_node_strength(node_id, new_strength)
                updated += 1

        return updated

    # =========================================================================
    # Working memory
    # =========================================================================

    def get_working_memory(
        self,
        agent_id: Optional[str] = None,
        limit: int = 20,
        conversation_thread: Optional[str] = None,
    ) -> list[MemoryNode]:
        """Get current working memory contents (high-activation recent memories)."""
        from cerebro.cortex import _scope_sql
        scope_clause, scope_params = _scope_sql(agent_id, conversation_thread)

        query = f"""SELECT id FROM memory_nodes
            WHERE layer = 'working'{scope_clause}"""
        params: list = list(scope_params)

        query += " ORDER BY last_retrievability DESC, salience DESC LIMIT ?"
        params.append(limit)

        rows = self._graph.conn.execute(query, params).fetchall()
        return [n for r in rows if (n := self._graph.get_node(r["id"]))]
