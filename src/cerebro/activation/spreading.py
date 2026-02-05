"""Spreading activation through the associative memory graph.

Implements Collins & Loftus style spreading activation:
1. Seed nodes are activated from vector search results
2. Activation spreads through typed, weighted links
3. Activation decays with each hop
4. Budget limits prevent unbounded traversal

This runs on igraph for C-speed graph traversal.
"""

from cerebro.config import (
    LINK_TYPE_WEIGHTS,
    SPREADING_ACTIVATION_THRESHOLD,
    SPREADING_DECAY_PER_HOP,
    SPREADING_MAX_ACTIVATED,
    SPREADING_MAX_HOPS,
)
from cerebro.storage.graph_store import GraphStore
from cerebro.types import LinkType


def spreading_activation(
    graph: GraphStore,
    seed_ids: list[str],
    seed_weights: list[float],
    max_hops: int = SPREADING_MAX_HOPS,
    decay_per_hop: float = SPREADING_DECAY_PER_HOP,
    activation_threshold: float = SPREADING_ACTIVATION_THRESHOLD,
    max_activated: int = SPREADING_MAX_ACTIVATED,
    link_type_weights: dict[LinkType, float] | None = None,
) -> dict[str, float]:
    """Spread activation from seed memories through the associative network.

    Args:
        graph: GraphStore instance (uses igraph for fast traversal)
        seed_ids: Memory IDs to start from (typically vector search results)
        seed_weights: Initial activation for each seed (typically similarity scores)
        max_hops: Maximum traversal depth
        decay_per_hop: Activation multiplier per hop (0.6 = 40% decay each hop)
        activation_threshold: Minimum activation to continue spreading
        max_activated: Maximum total activated nodes (budget)
        link_type_weights: Override default link type relevance weights

    Returns:
        Dict of {memory_id: activation_score} for all activated memories,
        normalized to [0, 1] range.
    """
    if not seed_ids:
        return {}

    lt_weights = link_type_weights or LINK_TYPE_WEIGHTS

    # Initialize activation map with seeds
    activated: dict[str, float] = {}
    for node_id, weight in zip(seed_ids, seed_weights):
        if node_id in graph._id_to_vertex:
            activated[node_id] = weight

    if not activated:
        return {}

    frontier = set(activated.keys())

    for hop in range(max_hops):
        if not frontier or len(activated) >= max_activated:
            break

        hop_decay = decay_per_hop ** (hop + 1)
        next_frontier: set[str] = set()

        for node_id in frontier:
            source_activation = activated.get(node_id, 0.0)
            if source_activation < activation_threshold:
                continue

            # Get neighbors via igraph (C-speed)
            neighbors = graph.get_neighbors(node_id)

            for neighbor_id, link_weight, link_type_str in neighbors:
                # Look up link type weight
                try:
                    lt = LinkType(link_type_str)
                    type_weight = lt_weights.get(lt, 0.5)
                except ValueError:
                    type_weight = 0.5

                # Compute spread amount
                spread = source_activation * link_weight * type_weight * hop_decay

                if spread < activation_threshold:
                    continue

                # Accumulate activation (diminishing returns for already-activated)
                if neighbor_id in activated:
                    existing = activated[neighbor_id]
                    # Sublinear accumulation: adding to existing gives less boost
                    activated[neighbor_id] = max(existing, existing + spread * 0.5)
                else:
                    activated[neighbor_id] = spread
                    next_frontier.add(neighbor_id)

                if len(activated) >= max_activated:
                    break

            if len(activated) >= max_activated:
                break

        frontier = next_frontier

    # Normalize to [0, 1]
    if activated:
        max_val = max(activated.values())
        if max_val > 0:
            activated = {k: v / max_val for k, v in activated.items()}

    return activated
