"""ACT-R + FSRS hybrid memory strength model.

Combines:
- ACT-R base-level activation: B(t) = ln(Σ t_k^{-d})
  Power-law decay from access history (Anderson, 1993)
- FSRS retrievability: R(t,S) = (1 + t/9S)^{-1}
  Spaced-repetition forgetting curve (open-spaced-repetition)
- Combined recall score blending vector similarity, activation,
  retrievability, and salience
"""

import math
import time
from typing import Optional

from cerebro.config import (
    ACTR_B_CONSTANT,
    ACTR_DECAY_RATE,
    ACTR_MIN_TIME_SECONDS,
    ACTR_NOISE,
    ACTR_RETRIEVAL_THRESHOLD,
    FSRS_INITIAL_DIFFICULTY,
    FSRS_INITIAL_STABILITY,
    FSRS_MAX_STABILITY,
    FSRS_MIN_STABILITY,
    MAX_STORED_TIMESTAMPS,
    SCORE_WEIGHT_ACTIVATION,
    SCORE_WEIGHT_RETRIEVABILITY,
    SCORE_WEIGHT_SALIENCE,
    SCORE_WEIGHT_VECTOR,
)
from cerebro.models.memory import StrengthState


# =============================================================================
# ACT-R Base-Level Activation
# =============================================================================


def base_level_activation(
    access_timestamps: list[float],
    current_time: Optional[float] = None,
    compressed_count: int = 0,
    compressed_avg_interval: float = 0.0,
    decay: float = ACTR_DECAY_RATE,
) -> float:
    """Compute ACT-R base-level activation B(t) = ln(Σ t_k^{-d}).

    Args:
        access_timestamps: Unix timestamps of individual accesses
        current_time: Current unix timestamp (defaults to now)
        compressed_count: Number of old compressed accesses
        compressed_avg_interval: Average interval between compressed accesses (seconds)
        decay: Decay rate parameter d (default 0.5)

    Returns:
        Base-level activation value. Higher = more accessible.
        Returns -inf if never accessed.
    """
    if not access_timestamps and compressed_count == 0:
        return float("-inf")

    now = current_time or time.time()
    total = 0.0

    # Sum over individual timestamps (most recent, precise)
    for ts in access_timestamps:
        t_k = max(now - ts, ACTR_MIN_TIME_SECONDS)
        total += t_k ** (-decay)

    # Approximate contribution from compressed old accesses
    if compressed_count > 0 and compressed_avg_interval > 0:
        # Estimate where old accesses would have been
        oldest_individual = min(access_timestamps) if access_timestamps else now
        for k in range(compressed_count):
            t_k = max(now - oldest_individual + (k + 1) * compressed_avg_interval,
                      ACTR_MIN_TIME_SECONDS)
            total += t_k ** (-decay)

    if total <= 0:
        return float("-inf")

    return math.log(total) + ACTR_B_CONSTANT


# =============================================================================
# FSRS Retrievability (Forgetting Curve)
# =============================================================================


def retrievability(
    elapsed_days: float,
    stability: float,
) -> float:
    """FSRS forgetting curve: R(t,S) = (1 + t/9S)^{-1}.

    Args:
        elapsed_days: Time since last access in days
        stability: Memory stability in days (interval at which R=90%)

    Returns:
        Probability of recall [0, 1]
    """
    if stability <= 0:
        return 0.0
    if elapsed_days <= 0:
        return 1.0
    return (1.0 + elapsed_days / (9.0 * stability)) ** (-1.0)


def update_stability_on_recall(
    stability: float,
    difficulty: float,
    current_retrievability: float,
) -> float:
    """Update FSRS stability after successful recall.

    The "desirable difficulty" effect: lower retrievability at time of recall
    leads to larger stability gain.

    Args:
        stability: Current stability
        difficulty: Current difficulty (1-10)
        current_retrievability: R at time of recall

    Returns:
        New stability value
    """
    # FSRS SInc formula (simplified)
    s_inc = (
        math.exp(11.0 - difficulty)
        * (stability ** -0.2)
        * (math.exp((1.0 - current_retrievability) * 9.0) - 1.0)
    )
    new_s = stability * (1.0 + s_inc)
    return max(FSRS_MIN_STABILITY, min(FSRS_MAX_STABILITY, new_s))


def update_stability_on_lapse(
    stability: float,
    difficulty: float,
) -> float:
    """Update FSRS stability after a lapse (memory not recalled when needed).

    Args:
        stability: Current stability
        difficulty: Current difficulty

    Returns:
        New (reduced) stability value
    """
    new_s = stability * 0.3 * ((11.0 - difficulty) ** 0.2)
    return max(FSRS_MIN_STABILITY, new_s)


def update_difficulty_on_recall(
    difficulty: float,
    current_retrievability: float,
) -> float:
    """Update FSRS difficulty after recall. Mean-reverts toward baseline.

    Easy recalls reduce difficulty; hard recalls increase it.

    Args:
        difficulty: Current difficulty (1-10)
        current_retrievability: R at time of recall

    Returns:
        New difficulty value [1, 10]
    """
    delta = -0.8 * (current_retrievability - 0.5)
    new_d = difficulty + delta
    # Mean reversion toward initial difficulty
    new_d = 0.9 * new_d + 0.1 * FSRS_INITIAL_DIFFICULTY
    return max(1.0, min(10.0, new_d))


# =============================================================================
# Combined Recall Probability
# =============================================================================


def recall_probability(
    activation: float,
    threshold: float = ACTR_RETRIEVAL_THRESHOLD,
    noise: float = ACTR_NOISE,
) -> float:
    """ACT-R recall probability: P(t) = sigmoid((A(t) - tau) / s).

    Args:
        activation: Total activation A(t) = B(t) + associative
        threshold: Retrieval threshold tau
        noise: Noise parameter s

    Returns:
        Probability of recall [0, 1]
    """
    if activation == float("-inf"):
        return 0.0
    if noise <= 0:
        return 1.0 if activation >= threshold else 0.0
    x = (activation - threshold) / noise
    # Clamp to avoid overflow
    x = max(-20.0, min(20.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def combined_recall_score(
    vector_similarity: float,
    base_level: float,
    associative: float,
    fsrs_retrievability: float,
    salience: float,
    w_vector: float = SCORE_WEIGHT_VECTOR,
    w_activation: float = SCORE_WEIGHT_ACTIVATION,
    w_retrievability: float = SCORE_WEIGHT_RETRIEVABILITY,
    w_salience: float = SCORE_WEIGHT_SALIENCE,
) -> float:
    """Compute the final recall score for ranking search results.

    Blends four signals:
    - Vector similarity (semantic relevance from ChromaDB)
    - Activation score (ACT-R base-level + associative, sigmoidified)
    - Retrievability (FSRS forgetting curve)
    - Salience (emotional importance from metadata)

    Args:
        vector_similarity: Cosine similarity [0, 1]
        base_level: ACT-R base-level activation
        associative: Spreading activation from context
        fsrs_retrievability: FSRS R(t,S) [0, 1]
        salience: Emotional salience [0, 1]

    Returns:
        Combined score [0, 1]
    """
    # Convert activation to 0-1 range via sigmoid
    activation_score = recall_probability(base_level + associative)

    score = (
        w_vector * max(0.0, min(1.0, vector_similarity))
        + w_activation * activation_score
        + w_retrievability * max(0.0, min(1.0, fsrs_retrievability))
        + w_salience * max(0.0, min(1.0, salience))
    )
    return max(0.0, min(1.0, score))


# =============================================================================
# Access Recording
# =============================================================================


def record_access(strength: StrengthState, current_time: Optional[float] = None) -> StrengthState:
    """Record a memory access and update all strength parameters.

    This is the main entry point for updating a memory's strength when it
    is retrieved/accessed.

    Args:
        strength: Current strength state (will be copied, not mutated)
        current_time: Unix timestamp (defaults to now)

    Returns:
        New StrengthState with updated values
    """
    now = current_time or time.time()

    # Copy timestamps and add new one
    timestamps = list(strength.access_timestamps)
    timestamps.append(now)

    # Compress if too many
    compressed_count = strength.compressed_count
    compressed_avg_interval = strength.compressed_avg_interval
    if len(timestamps) > MAX_STORED_TIMESTAMPS:
        # Move oldest timestamps to compressed summary
        overflow = len(timestamps) - MAX_STORED_TIMESTAMPS
        old_timestamps = sorted(timestamps)[:overflow]

        if old_timestamps and len(old_timestamps) > 1:
            intervals = [old_timestamps[i+1] - old_timestamps[i]
                         for i in range(len(old_timestamps) - 1)]
            new_avg = sum(intervals) / len(intervals) if intervals else compressed_avg_interval
            total_count = compressed_count + overflow
            if total_count > 0:
                compressed_avg_interval = (
                    (compressed_count * compressed_avg_interval + overflow * new_avg)
                    / total_count
                )
            compressed_count = total_count
        else:
            compressed_count += overflow

        timestamps = sorted(timestamps)[overflow:]

    # Compute elapsed time for FSRS
    prev_access = strength.access_timestamps[-1] if strength.access_timestamps else now
    elapsed_seconds = max(now - prev_access, 0)
    elapsed_days = elapsed_seconds / 86400.0

    # Current retrievability before this access
    current_r = retrievability(elapsed_days, strength.stability)

    # Update FSRS stability (successful recall)
    new_stability = update_stability_on_recall(
        strength.stability, strength.difficulty, current_r
    )

    # Update difficulty
    new_difficulty = update_difficulty_on_recall(strength.difficulty, current_r)

    # Compute new base-level activation
    new_base = base_level_activation(
        timestamps, now, compressed_count, compressed_avg_interval
    )

    return StrengthState(
        stability=new_stability,
        difficulty=new_difficulty,
        access_count=strength.access_count + 1,
        access_timestamps=timestamps,
        compressed_count=compressed_count,
        compressed_avg_interval=compressed_avg_interval,
        last_retrievability=1.0,  # Just accessed, so R=1
        last_activation=new_base,
        last_computed_at=now,
    )
