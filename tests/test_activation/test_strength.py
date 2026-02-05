"""Tests for ACT-R + FSRS hybrid strength model."""

import math
import time

from cerebro.activation.strength import (
    base_level_activation,
    combined_recall_score,
    recall_probability,
    record_access,
    retrievability,
    update_difficulty_on_recall,
    update_stability_on_lapse,
    update_stability_on_recall,
)
from cerebro.models.memory import StrengthState


class TestBaseLevel:
    def test_never_accessed(self):
        """Memory with no accesses should have -inf activation."""
        result = base_level_activation([], current_time=1000.0)
        assert result == float("-inf")

    def test_single_recent_access(self):
        """Very recent access should give high activation."""
        now = 1000.0
        result = base_level_activation([now - 1.0], current_time=now)
        # B(t) = ln(1^{-0.5}) = ln(1) = 0
        assert result == 0.0

    def test_decay_with_time(self):
        """Activation should decrease as time since access increases."""
        now = 1000.0
        recent = base_level_activation([now - 10.0], current_time=now)
        old = base_level_activation([now - 10000.0], current_time=now)
        assert recent > old

    def test_more_accesses_higher_activation(self):
        """More accesses should give higher activation."""
        now = 1000.0
        one = base_level_activation([now - 100.0], current_time=now)
        three = base_level_activation([now - 100.0, now - 50.0, now - 10.0], current_time=now)
        assert three > one

    def test_power_law_decay(self):
        """Verify power law: B(t) = ln(t^{-d}) = -d*ln(t) for single access."""
        now = 10000.0
        t = 100.0  # 100 seconds ago
        result = base_level_activation([now - t], current_time=now)
        expected = math.log(t ** (-0.5))
        assert abs(result - expected) < 0.001

    def test_compressed_accesses(self):
        """Compressed old accesses should still contribute."""
        now = 1000.0
        with_compressed = base_level_activation(
            [now - 10.0], current_time=now,
            compressed_count=10, compressed_avg_interval=100.0,
        )
        without = base_level_activation([now - 10.0], current_time=now)
        assert with_compressed > without

    def test_min_time_clamped(self):
        """Access at exact current time should not blow up."""
        now = 1000.0
        result = base_level_activation([now], current_time=now)
        # Clamped to 1 second: ln(1^{-0.5}) = 0
        assert math.isfinite(result)


class TestRetrievability:
    def test_just_accessed(self):
        """R should be ~1.0 right after access."""
        assert retrievability(0.0, stability=5.0) == 1.0

    def test_at_stability_interval(self):
        """R should be ~0.9 at t=S (by FSRS definition)."""
        r = retrievability(5.0, stability=5.0)
        expected = (1.0 + 5.0 / 45.0) ** (-1.0)
        assert abs(r - expected) < 0.001
        # Should be close to 0.9
        assert abs(r - 0.9) < 0.02

    def test_decay_over_time(self):
        r_1day = retrievability(1.0, stability=5.0)
        r_10day = retrievability(10.0, stability=5.0)
        r_100day = retrievability(100.0, stability=5.0)
        assert r_1day > r_10day > r_100day

    def test_higher_stability_slower_decay(self):
        """Higher stability should mean slower forgetting."""
        r_low = retrievability(10.0, stability=2.0)
        r_high = retrievability(10.0, stability=50.0)
        assert r_high > r_low

    def test_zero_stability(self):
        assert retrievability(1.0, stability=0.0) == 0.0


class TestStabilityUpdates:
    def test_stability_increases_on_recall(self):
        new_s = update_stability_on_recall(stability=5.0, difficulty=5.0, current_retrievability=0.5)
        assert new_s > 5.0

    def test_desirable_difficulty(self):
        """Lower R at recall time should give bigger stability boost."""
        # Use D=8 (hard memory) so the boost doesn't hit the stability ceiling
        easy = update_stability_on_recall(1.0, 8.0, current_retrievability=0.9)
        hard = update_stability_on_recall(1.0, 8.0, current_retrievability=0.3)
        assert hard > easy  # harder recall = more learning

    def test_stability_decreases_on_lapse(self):
        new_s = update_stability_on_lapse(stability=10.0, difficulty=5.0)
        assert new_s < 10.0

    def test_stability_never_below_min(self):
        new_s = update_stability_on_lapse(stability=0.1, difficulty=10.0)
        assert new_s >= 0.1

    def test_difficulty_mean_reverts(self):
        """Difficulty should move toward baseline over time."""
        # Easy recall
        d1 = update_difficulty_on_recall(8.0, current_retrievability=0.9)
        assert d1 < 8.0  # should decrease (was easy)

        # Hard recall (low R)
        d2 = update_difficulty_on_recall(3.0, current_retrievability=0.1)
        assert d2 > 3.0  # should increase (was hard)

    def test_difficulty_clamped(self):
        d = update_difficulty_on_recall(1.0, current_retrievability=0.99)
        assert d >= 1.0
        d = update_difficulty_on_recall(10.0, current_retrievability=0.01)
        assert d <= 10.0


class TestRecallProbability:
    def test_neg_inf_activation(self):
        assert recall_probability(float("-inf")) == 0.0

    def test_high_activation(self):
        p = recall_probability(5.0)
        assert p > 0.95

    def test_low_activation(self):
        p = recall_probability(-5.0)
        assert p < 0.05

    def test_at_threshold(self):
        """At threshold, probability should be ~0.5."""
        p = recall_probability(0.0, threshold=0.0)
        assert abs(p - 0.5) < 0.01

    def test_monotonic(self):
        """Higher activation = higher recall probability."""
        p1 = recall_probability(-2.0)
        p2 = recall_probability(0.0)
        p3 = recall_probability(2.0)
        assert p1 < p2 < p3


class TestCombinedScore:
    def test_all_high(self):
        score = combined_recall_score(
            vector_similarity=0.9,
            base_level=3.0,
            associative=1.0,
            fsrs_retrievability=0.95,
            salience=0.9,
        )
        assert score > 0.8

    def test_all_low(self):
        score = combined_recall_score(
            vector_similarity=0.1,
            base_level=-5.0,
            associative=0.0,
            fsrs_retrievability=0.1,
            salience=0.1,
        )
        assert score < 0.2

    def test_weights_sum_to_1(self):
        from cerebro.config import (
            SCORE_WEIGHT_ACTIVATION,
            SCORE_WEIGHT_RETRIEVABILITY,
            SCORE_WEIGHT_SALIENCE,
            SCORE_WEIGHT_VECTOR,
        )
        total = (SCORE_WEIGHT_VECTOR + SCORE_WEIGHT_ACTIVATION
                 + SCORE_WEIGHT_RETRIEVABILITY + SCORE_WEIGHT_SALIENCE)
        assert abs(total - 1.0) < 0.001

    def test_vector_similarity_most_important(self):
        """Verify vector similarity has the highest weight."""
        from cerebro.config import (
            SCORE_WEIGHT_ACTIVATION,
            SCORE_WEIGHT_RETRIEVABILITY,
            SCORE_WEIGHT_SALIENCE,
            SCORE_WEIGHT_VECTOR,
        )
        assert SCORE_WEIGHT_VECTOR > SCORE_WEIGHT_ACTIVATION
        assert SCORE_WEIGHT_VECTOR > SCORE_WEIGHT_RETRIEVABILITY
        assert SCORE_WEIGHT_VECTOR > SCORE_WEIGHT_SALIENCE

    def test_bounded_output(self):
        """Score should always be in [0, 1]."""
        for vs in [0.0, 0.5, 1.0]:
            for bl in [-10.0, 0.0, 10.0]:
                for r in [0.0, 0.5, 1.0]:
                    score = combined_recall_score(vs, bl, 0.0, r, 0.5)
                    assert 0.0 <= score <= 1.0


class TestRecordAccess:
    def test_basic_access(self):
        s = StrengthState()
        now = time.time()
        s2 = record_access(s, current_time=now)

        assert s2.access_count == 1
        assert len(s2.access_timestamps) == 1
        assert s2.access_timestamps[0] == now
        assert s2.last_retrievability == 1.0

    def test_repeated_access(self):
        s = StrengthState()
        now = time.time()
        s = record_access(s, current_time=now)
        s = record_access(s, current_time=now + 3600)
        s = record_access(s, current_time=now + 7200)

        assert s.access_count == 3
        assert len(s.access_timestamps) == 3

    def test_stability_grows(self):
        """Stability should increase with repeated accesses."""
        s = StrengthState(stability=1.0)
        now = time.time()
        for i in range(5):
            s = record_access(s, current_time=now + i * 86400)
        assert s.stability > 1.0

    def test_timestamp_compression(self):
        """Timestamps beyond MAX_STORED should be compressed."""
        s = StrengthState()
        now = time.time()
        for i in range(60):
            s = record_access(s, current_time=now + i * 100)

        from cerebro.config import MAX_STORED_TIMESTAMPS
        assert len(s.access_timestamps) <= MAX_STORED_TIMESTAMPS
        assert s.compressed_count > 0
