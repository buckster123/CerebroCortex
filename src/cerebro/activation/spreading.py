"""Spreading activation through the associative memory graph.

Implements Collins & Loftus style spreading activation:
1. Seed nodes are activated from vector search results
2. Activation spreads through typed, weighted links
3. Activation decays with each hop
4. Budget limits prevent unbounded traversal

This runs on igraph for C-speed graph traversal.
"""

from datetime import datetime
from typing import Optional

from cerebro.config import (
    LINK_DECAY_HALFLIFE_DAYS,
    LINK_TYPE_WEIGHTS,
    SPREADING_ACTIVATION_THRESHOLD,
    SPREADING_DECAY_PER_HOP,
    SPREADING_MAX_ACTIVATED,
    SPREADING_MAX_HOPS,
)
from cerebro.storage.graph_store import GraphStore
from cerebro.types import LinkType, Visibility


def effective_link_weight(
    stored_weight: float,
    last_activated: Optional[datetime],
    halflife_days: float = LINK_DECAY_HALFLIFE_DAYS,
) -> float:
    """Apply time decay to a link weight based on when it was last activated.

    Uses an FSRS-style power-law curve: w_eff = w * (1 + age/9H)^{-1}
    where H is the halflife. This decays slowly at first, then faster.

    Computed on-the-fly — stored weights are not mutated (Hebbian
    strengthening handles that separately).

    Args:
        stored_weight: The link's stored weight [0, 1]
        last_activated: When the link was last traversed/activated
        halflife_days: Days until ~50% decay

    Returns:
        Effective weight after time decay [0, stored_weight]
    """
    if last_activated is None or halflife_days <= 0:
        return stored_weight
    age_days = (datetime.now() - last_activated).total_seconds() / 86400.0
    if age_days <= 0:
        return stored_weight
    decay = (1.0 + age_days / (9.0 * halflife_days)) ** (-1.0)
    return stored_weight * decay


def _build_visibility_cache(
    graph: GraphStore,
    node_ids: set[str],
) -> dict[str, tuple[str, str, Optional[str]]]:
    """Build a cache of (agent_id, visibility, conversation_thread) for scope checks.

    Bulk-fetches from SQLite to avoid per-neighbor lookups.
    """
    if not node_ids:
        return {}
    cache = {}
    placeholders = ",".join("?" * len(node_ids))
    rows = graph.conn.execute(
        f"SELECT id, agent_id, visibility, conversation_thread FROM memory_nodes WHERE id IN ({placeholders})",
        list(node_ids),
    ).fetchall()
    for r in rows:
        cache[r["id"]] = (r["agent_id"], r["visibility"], r["conversation_thread"])
    return cache


def _check_access(
    vis_cache: dict[str, tuple[str, str, Optional[str]]],
    node_id: str,
    agent_id: Optional[str],
    conversation_thread: Optional[str],
) -> bool:
    """Check if agent can access a node using the visibility cache."""
    if agent_id is None:
        return True
    entry = vis_cache.get(node_id)
    if entry is None:
        return True  # not in cache = not in DB, will be filtered later
    owner, vis, thread = entry
    if vis == "shared":
        return True
    if vis == "private":
        return owner == agent_id
    if vis == "thread":
        if conversation_thread and thread:
            return thread == conversation_thread
        return owner == agent_id
    return False


def spreading_activation(
    graph: GraphStore,
    seed_ids: list[str],
    seed_weights: list[float],
    max_hops: int = SPREADING_MAX_HOPS,
    decay_per_hop: float = SPREADING_DECAY_PER_HOP,
    activation_threshold: float = SPREADING_ACTIVATION_THRESHOLD,
    max_activated: int = SPREADING_MAX_ACTIVATED,
    link_type_weights: dict[LinkType, float] | None = None,
    agent_id: Optional[str] = None,
    conversation_thread: Optional[str] = None,
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
        agent_id: Agent for scope filtering (None = no filter)
        conversation_thread: Thread ID for THREAD visibility matching

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

    # Build visibility cache for scope filtering
    vis_cache: dict[str, tuple[str, str, Optional[str]]] = {}
    if agent_id is not None:
        # Pre-fetch seeds; neighbors will be added per hop
        vis_cache = _build_visibility_cache(graph, set(activated.keys()))

    frontier = set(activated.keys())

    for hop in range(max_hops):
        if not frontier or len(activated) >= max_activated:
            break

        hop_decay = decay_per_hop ** (hop + 1)
        next_frontier: set[str] = set()

        # Collect all neighbor IDs for this hop for batch visibility lookup
        all_neighbor_ids: set[str] = set()
        frontier_neighbors: dict[str, list[tuple]] = {}
        for node_id in frontier:
            source_activation = activated.get(node_id, 0.0)
            if source_activation < activation_threshold:
                continue
            neighbors = graph.get_neighbors(node_id)
            frontier_neighbors[node_id] = neighbors
            for n in neighbors:
                if n[0] not in vis_cache:
                    all_neighbor_ids.add(n[0])

        # Batch-fetch visibility for new neighbor IDs
        if agent_id is not None and all_neighbor_ids:
            vis_cache.update(_build_visibility_cache(graph, all_neighbor_ids))

        for node_id, neighbors in frontier_neighbors.items():
            source_activation = activated.get(node_id, 0.0)

            for neighbor_entry in neighbors:
                neighbor_id = neighbor_entry[0]
                link_weight = neighbor_entry[1]
                link_type_str = neighbor_entry[2]
                last_activated_iso = neighbor_entry[3] if len(neighbor_entry) > 3 else None

                # Scope check: skip inaccessible neighbors
                if not _check_access(vis_cache, neighbor_id, agent_id, conversation_thread):
                    continue

                # Look up link type weight
                try:
                    lt = LinkType(link_type_str)
                    type_weight = lt_weights.get(lt, 0.5)
                except ValueError:
                    type_weight = 0.5

                # Apply link decay based on last activation time
                last_act_dt = None
                if last_activated_iso:
                    try:
                        last_act_dt = datetime.fromisoformat(last_activated_iso)
                    except (ValueError, TypeError):
                        pass
                decayed_weight = effective_link_weight(link_weight, last_act_dt)

                # Compute spread amount
                spread = source_activation * decayed_weight * type_weight * hop_decay

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
