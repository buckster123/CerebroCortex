"""Memory decay and layer promotion logic.

Handles the time-based degradation of memories and their promotion
between layers based on access patterns and strength.
"""

import time
from datetime import datetime

from cerebro.config import LAYER_CONFIG
from cerebro.models.memory import MemoryNode, StrengthState
from cerebro.activation.strength import base_level_activation, retrievability
from cerebro.types import MemoryLayer


def compute_current_retrievability(strength: StrengthState, current_time: float | None = None) -> float:
    """Compute current FSRS retrievability for a memory."""
    now = current_time or time.time()
    if not strength.access_timestamps:
        # Never accessed: use created time approximation
        return 0.0
    last_access = max(strength.access_timestamps)
    elapsed_days = max(now - last_access, 0) / 86400.0
    return retrievability(elapsed_days, strength.stability)


def compute_current_activation(strength: StrengthState, current_time: float | None = None) -> float:
    """Compute current ACT-R base-level activation for a memory."""
    now = current_time or time.time()
    return base_level_activation(
        strength.access_timestamps,
        now,
        strength.compressed_count,
        strength.compressed_avg_interval,
    )


def check_promotion_eligibility(node: MemoryNode) -> tuple[bool, str | None]:
    """Check if a memory is eligible for promotion to the next layer.

    Returns:
        (eligible, target_layer_or_reason)
    """
    current = node.metadata.layer.value
    config = LAYER_CONFIG.get(current)
    if not config:
        return False, "Unknown layer"

    threshold = config.get("promotion_access_count")
    if threshold is None:
        return False, "No promotion from this layer"

    if node.strength.access_count < threshold:
        return False, f"Need {threshold} accesses, have {node.strength.access_count}"

    min_age = config.get("promotion_min_age_hours")
    if min_age is not None:
        age_hours = (datetime.now() - node.created_at).total_seconds() / 3600
        if age_hours < min_age:
            return False, f"Need {min_age}h age, have {age_hours:.1f}h"

    # Determine target layer
    promotion_map = {
        "sensory": MemoryLayer.WORKING,
        "working": MemoryLayer.LONG_TERM,
        "long_term": MemoryLayer.CORTEX,
    }
    target = promotion_map.get(current)
    if target is None:
        return False, "No promotion target"

    return True, target.value


def apply_decay_tick(strength: StrengthState, current_time: float | None = None) -> StrengthState:
    """Recompute cached retrievability and activation values.

    Called periodically (e.g., during dream engine) to update cached values
    without recording a new access.
    """
    now = current_time or time.time()
    r = compute_current_retrievability(strength, now)
    a = compute_current_activation(strength, now)

    return StrengthState(
        stability=strength.stability,
        difficulty=strength.difficulty,
        access_count=strength.access_count,
        access_timestamps=strength.access_timestamps,
        compressed_count=strength.compressed_count,
        compressed_avg_interval=strength.compressed_avg_interval,
        last_retrievability=r,
        last_activation=a,
        last_computed_at=now,
    )
