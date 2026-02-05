"""Tests for the CerebroCortex main coordinator."""

import tempfile
from pathlib import Path

import pytest

from cerebro.cortex import CerebroCortex
from cerebro.types import EmotionalValence, LinkType, MemoryType


@pytest.fixture
def cortex():
    """CerebroCortex with temporary database."""
    with tempfile.TemporaryDirectory() as d:
        ctx = CerebroCortex(db_path=Path(d) / "test.db")
        ctx.initialize()
        yield ctx
        ctx.close()


class TestRemember:
    def test_store_basic_memory(self, cortex):
        node = cortex.remember("Python lists are mutable sequences that support indexing")
        assert node is not None
        assert node.content == "Python lists are mutable sequences that support indexing"

    def test_gates_short_content(self, cortex):
        result = cortex.remember("hi")
        assert result is None

    def test_deduplication(self, cortex):
        content = "FastAPI uses Starlette under the hood for web handling"
        first = cortex.remember(content)
        assert first is not None

        second = cortex.remember(content)
        assert second is None  # duplicate gated out

    def test_stores_with_type(self, cortex):
        node = cortex.remember(
            "When debugging: check logs first, then reproduce",
            memory_type=MemoryType.PROCEDURAL,
            tags=["debugging"],
        )
        assert node.metadata.memory_type == MemoryType.PROCEDURAL

    def test_enriches_with_concepts(self, cortex):
        node = cortex.remember(
            "FastAPI is a modern Python web framework for building APIs quickly"
        )
        assert len(node.metadata.concepts) > 0

    def test_applies_emotional_analysis(self, cortex):
        node = cortex.remember(
            "The breakthrough was amazing! Finally solved the bug after hours of frustration."
        )
        assert node.metadata.valence != EmotionalValence.NEUTRAL

    def test_creates_links_for_context(self, cortex):
        first = cortex.remember("First memory about Python programming and development")
        second = cortex.remember(
            "Second memory about Python modules and packages",
            context_ids=[first.id],
        )
        # Should have contextual link
        assert cortex.graph.has_link(second.id, first.id)


class TestRecall:
    def test_recall_with_context(self, cortex):
        m1 = cortex.remember("Python lists support append, extend, and insert")
        m2 = cortex.remember("Python dicts provide key-value mapping operations")
        m3 = cortex.remember("JavaScript arrays are similar to Python lists in many ways")

        # Link m1 and m3
        cortex.associate(m1.id, m3.id, LinkType.SEMANTIC, evidence="Both about lists")

        # Recall with m1 as context
        results = cortex.recall(
            "list operations", context_ids=[m1.id], top_k=5,
        )
        assert len(results) > 0

    def test_recall_returns_scored_results(self, cortex):
        cortex.remember("Memory about authentication and JWT tokens for web apps")
        cortex.remember("Memory about database optimization and query performance")

        results = cortex.recall("authentication")
        for node, score in results:
            assert 0.0 <= score <= 1.0

    def test_recall_filters_by_type(self, cortex):
        cortex.remember("A semantic fact about Python", memory_type=MemoryType.SEMANTIC)
        cortex.remember("A procedural workflow for testing", memory_type=MemoryType.PROCEDURAL)

        results = cortex.recall(
            "Python",
            memory_types=[MemoryType.SEMANTIC],
        )
        for node, _ in results:
            assert node.metadata.memory_type == MemoryType.SEMANTIC


class TestAssociate:
    def test_create_explicit_link(self, cortex):
        m1 = cortex.remember("Authentication module handles user login and sessions")
        m2 = cortex.remember("Security module manages JWT tokens and encryption")

        link_id = cortex.associate(
            m1.id, m2.id, LinkType.SUPPORTS,
            evidence="Auth depends on security",
        )
        assert link_id is not None

    def test_associate_nonexistent(self, cortex):
        result = cortex.associate(
            "nonexistent1", "nonexistent2", LinkType.CAUSAL,
        )
        assert result is None


class TestEpisodes:
    def test_episode_lifecycle(self, cortex):
        episode = cortex.episode_start(title="Test session")
        assert episode.id is not None

        ended = cortex.episode_end(episode.id, summary="Done")
        assert ended is not None
        assert ended.ended_at is not None

    def test_session_auto_episode(self, cortex):
        """Memories stored with session_id should auto-track in episodes."""
        node = cortex.remember(
            "First event in session about debugging and development work",
            session_id="test_session_001",
        )
        assert node is not None


class TestStats:
    def test_stats_structure(self, cortex):
        cortex.remember("A test memory for statistics about Python programming")

        stats = cortex.stats()
        assert "nodes" in stats
        assert "links" in stats
        assert "schemas" in stats
        assert stats["nodes"] >= 1


class TestInitialization:
    def test_double_initialize(self):
        with tempfile.TemporaryDirectory() as d:
            ctx = CerebroCortex(db_path=Path(d) / "test.db")
            ctx.initialize()
            ctx.initialize()  # should not error
            ctx.close()

    def test_uninitialized_raises(self):
        ctx = CerebroCortex()
        with pytest.raises(RuntimeError):
            _ = ctx.graph
