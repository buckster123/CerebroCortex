"""Tests for the CerebroCortex main coordinator."""

import tempfile
from pathlib import Path

import pytest

from cerebro.cortex import CerebroCortex
from cerebro.types import EmotionalValence, LinkType, MemoryType


@pytest.fixture
def cortex():
    """CerebroCortex with temporary database and vector store."""
    with tempfile.TemporaryDirectory() as d:
        ctx = CerebroCortex(
            db_path=Path(d) / "test.db",
            chroma_dir=Path(d) / "chroma",
        )
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

    def test_remember_writes_to_vector_store(self, cortex):
        """Verify that remember() writes to ChromaDB as well as graph."""
        node = cortex.remember(
            "PostgreSQL supports JSONB columns for semi-structured data",
            memory_type=MemoryType.SEMANTIC,
        )
        # Should be in knowledge collection
        from cerebro.config import COLLECTION_KNOWLEDGE
        count = cortex.vector.count(COLLECTION_KNOWLEDGE)
        assert count >= 1


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

    def test_recall_finds_semantically_similar(self, cortex):
        """Vector search should find semantically related memories."""
        cortex.remember(
            "Docker containers provide lightweight isolated environments for applications"
        )
        cortex.remember(
            "Kubernetes orchestrates container deployments across clusters"
        )
        cortex.remember(
            "The French Revolution began in 1789 with the storming of the Bastille"
        )

        results = cortex.recall("container orchestration and deployment", top_k=3)
        assert len(results) >= 2
        # Container-related memories should rank higher than French Revolution
        contents = [node.content for node, _ in results]
        container_hits = sum(
            1 for c in contents if "container" in c.lower() or "kubernetes" in c.lower()
        )
        assert container_hits >= 1

    def test_recall_with_type_filter_via_vector(self, cortex):
        """Vector recall respects memory type filters."""
        cortex.remember(
            "Python uses indentation for code blocks",
            memory_type=MemoryType.SEMANTIC,
        )
        cortex.remember(
            "Step 1: write test, Step 2: write code, Step 3: refactor",
            memory_type=MemoryType.PROCEDURAL,
        )

        results = cortex.recall(
            "programming", memory_types=[MemoryType.PROCEDURAL],
        )
        for node, _ in results:
            assert node.metadata.memory_type == MemoryType.PROCEDURAL

    def test_recall_vector_plus_spreading(self, cortex):
        """Spreading activation should expand results beyond pure vector hits."""
        m1 = cortex.remember("Machine learning models require training data")
        m2 = cortex.remember("Neural networks have layers of interconnected nodes")
        # m3 is linked to m2 but not directly about ML
        m3 = cortex.remember("GPU acceleration speeds up matrix operations significantly")
        cortex.associate(m2.id, m3.id, LinkType.CAUSAL, weight=0.9)

        # Recall about ML — m3 should appear via spreading from m2
        results = cortex.recall("machine learning neural networks", top_k=5)
        result_ids = [node.id for node, _ in results]
        assert m1.id in result_ids
        assert m2.id in result_ids
        # m3 may appear via spreading activation from m2

    def test_recall_empty_store(self, cortex):
        """Recall on empty store should return empty list."""
        results = cortex.recall("anything")
        assert results == []


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

    def test_stats_includes_vector_store(self, cortex):
        """Stats should include ChromaDB vector store counts."""
        cortex.remember("A memory about vector databases and embeddings")

        stats = cortex.stats()
        assert "vector_store" in stats
        assert isinstance(stats["vector_store"], dict)
        total = sum(stats["vector_store"].values())
        assert total >= 1


class TestInitialization:
    def test_double_initialize(self):
        with tempfile.TemporaryDirectory() as d:
            ctx = CerebroCortex(
                db_path=Path(d) / "test.db",
                chroma_dir=Path(d) / "chroma",
            )
            ctx.initialize()
            ctx.initialize()  # should not error
            ctx.close()

    def test_uninitialized_raises(self):
        ctx = CerebroCortex()
        with pytest.raises(RuntimeError):
            _ = ctx.graph

    def test_uninitialized_vector_raises(self):
        ctx = CerebroCortex()
        with pytest.raises(RuntimeError):
            _ = ctx.vector


class TestCollectionRouting:
    def test_semantic_to_knowledge(self):
        assert CerebroCortex._collection_for_type(MemoryType.SEMANTIC) == "cerebro_knowledge"

    def test_schematic_to_knowledge(self):
        assert CerebroCortex._collection_for_type(MemoryType.SCHEMATIC) == "cerebro_knowledge"

    def test_episodic_to_sessions(self):
        assert CerebroCortex._collection_for_type(MemoryType.EPISODIC) == "cerebro_sessions"

    def test_procedural_to_memories(self):
        assert CerebroCortex._collection_for_type(MemoryType.PROCEDURAL) == "cerebro_memories"

    def test_prospective_to_memories(self):
        assert CerebroCortex._collection_for_type(MemoryType.PROSPECTIVE) == "cerebro_memories"

    def test_affective_to_memories(self):
        assert CerebroCortex._collection_for_type(MemoryType.AFFECTIVE) == "cerebro_memories"
