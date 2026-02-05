"""Tests for the SemanticEngine (temporal lobe)."""

import json

from cerebro.engines.temporal import SemanticEngine
from cerebro.models.memory import MemoryMetadata, MemoryNode
from cerebro.types import MemoryType


class TestConceptExtraction:
    def test_extracts_concepts(self, graph_store):
        engine = SemanticEngine(graph_store)
        concepts = engine.extract_concepts(
            "Python lists are mutable sequences that support indexing and slicing"
        )
        assert len(concepts) > 0
        assert "python" in concepts

    def test_filters_stop_words(self, graph_store):
        engine = SemanticEngine(graph_store)
        concepts = engine.extract_concepts(
            "The cat is on the mat and the dog is under the table"
        )
        assert "the" not in concepts
        assert "is" not in concepts

    def test_max_concepts_limit(self, graph_store):
        engine = SemanticEngine(graph_store)
        concepts = engine.extract_concepts(
            "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron",
            max_concepts=5,
        )
        assert len(concepts) <= 5

    def test_empty_content(self, graph_store):
        engine = SemanticEngine(graph_store)
        concepts = engine.extract_concepts("")
        assert concepts == []


class TestSemanticLinks:
    def test_creates_semantic_links(self, graph_store):
        engine = SemanticEngine(graph_store)

        # Add an existing memory with concepts
        existing = MemoryNode(
            id="mem_sem1",
            content="Python list comprehensions are powerful",
            metadata=MemoryMetadata(concepts=["python", "list", "comprehensions"]),
        )
        graph_store.add_node(existing)

        # New memory sharing concepts
        new = MemoryNode(
            id="mem_sem2",
            content="Python list methods include append and extend",
            metadata=MemoryMetadata(concepts=["python", "list", "methods"]),
        )
        graph_store.add_node(new)

        created = engine.create_semantic_links(new)
        assert len(created) >= 1

    def test_no_links_without_concepts(self, graph_store):
        engine = SemanticEngine(graph_store)
        node = MemoryNode(
            id="mem_noconcept",
            content="a b c",  # too short for concept extraction
            metadata=MemoryMetadata(concepts=[]),
        )
        graph_store.add_node(node)

        created = engine.create_semantic_links(node)
        assert len(created) == 0


class TestEnrichNode:
    def test_enriches_with_concepts(self, graph_store):
        engine = SemanticEngine(graph_store)
        node = MemoryNode(
            id="mem_enrich1",
            content="FastAPI is a modern Python web framework for building APIs",
        )
        enriched = engine.enrich_node(node)
        assert len(enriched.metadata.concepts) > 0
        assert "fastapi" in enriched.metadata.concepts or "python" in enriched.metadata.concepts

    def test_preserves_existing_concepts(self, graph_store):
        engine = SemanticEngine(graph_store)
        node = MemoryNode(
            id="mem_enrich2",
            content="Some content",
            metadata=MemoryMetadata(concepts=["existing_concept"]),
        )
        enriched = engine.enrich_node(node)
        assert enriched.metadata.concepts == ["existing_concept"]


class TestContradictions:
    def test_detects_negation_conflict(self, graph_store):
        engine = SemanticEngine(graph_store)

        # Store a positive assertion
        assertion = MemoryNode(
            id="mem_assert1",
            content="Python is a compiled language that runs very fast natively",
            metadata=MemoryMetadata(
                memory_type=MemoryType.SEMANTIC,
                concepts=["python", "compiled", "language"],
            ),
        )
        graph_store.add_node(assertion)

        # Check contradiction with negation
        negation = MemoryNode(
            id="mem_neg1",
            content="Python is not a compiled language, it runs interpreted code natively",
            metadata=MemoryMetadata(
                memory_type=MemoryType.SEMANTIC,
                concepts=["python", "compiled", "language"],
            ),
        )

        contradictions = engine.find_contradictions(negation, candidates=["mem_assert1"])
        # Should detect the negation pattern
        assert len(contradictions) >= 1


class TestConceptCluster:
    def test_finds_cluster_members(self, graph_store):
        engine = SemanticEngine(graph_store)

        for i in range(5):
            graph_store.add_node(MemoryNode(
                id=f"mem_clust_{i}",
                content=f"Python fact {i}",
                metadata=MemoryMetadata(concepts=["python", f"fact{i}"]),
            ))

        cluster = engine.get_concept_cluster("python")
        assert len(cluster) == 5
