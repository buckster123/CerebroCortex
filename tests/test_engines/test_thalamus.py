"""Tests for the GatingEngine (thalamus)."""

from cerebro.engines.thalamus import GatingEngine
from cerebro.models.memory import MemoryNode
from cerebro.types import MemoryLayer, MemoryType


class TestGating:
    def test_gates_short_content(self, graph_store):
        engine = GatingEngine(graph_store)
        result = engine.evaluate_input("hi")
        assert result is None

    def test_accepts_valid_content(self, graph_store):
        engine = GatingEngine(graph_store)
        result = engine.evaluate_input("Python lists support indexing and slicing operations")
        assert result is not None
        assert result.content == "Python lists support indexing and slicing operations"

    def test_deduplication(self, graph_store):
        engine = GatingEngine(graph_store)
        content = "The quick brown fox jumps over the lazy dog"
        # First store: should succeed
        node = engine.evaluate_input(content)
        assert node is not None
        graph_store.add_node(node)

        # Second store: should be gated (duplicate)
        result = engine.evaluate_input(content)
        assert result is None

    def test_returns_memory_node(self, graph_store):
        engine = GatingEngine(graph_store)
        result = engine.evaluate_input(
            "FastAPI uses Starlette under the hood",
            tags=["python", "fastapi"],
        )
        assert isinstance(result, MemoryNode)
        assert result.metadata.tags == ["python", "fastapi"]


class TestClassification:
    def test_classifies_procedural(self, graph_store):
        engine = GatingEngine(graph_store)
        result = engine.evaluate_input(
            "Step 1: check the config. Step 2: restart the service."
        )
        assert result.metadata.memory_type == MemoryType.PROCEDURAL

    def test_classifies_affective(self, graph_store):
        engine = GatingEngine(graph_store)
        result = engine.evaluate_input(
            "I felt amazing when the deployment finally worked"
        )
        assert result.metadata.memory_type == MemoryType.AFFECTIVE

    def test_classifies_prospective(self, graph_store):
        engine = GatingEngine(graph_store)
        result = engine.evaluate_input(
            "Need to revisit the auth configuration later"
        )
        assert result.metadata.memory_type == MemoryType.PROSPECTIVE

    def test_classifies_episodic(self, graph_store):
        engine = GatingEngine(graph_store)
        result = engine.evaluate_input(
            "Deployed the app yesterday, then encountered a port conflict"
        )
        assert result.metadata.memory_type == MemoryType.EPISODIC

    def test_classifies_semantic_default(self, graph_store):
        engine = GatingEngine(graph_store)
        result = engine.evaluate_input(
            "Python is a dynamically typed programming language"
        )
        assert result.metadata.memory_type == MemoryType.SEMANTIC

    def test_explicit_type_overrides(self, graph_store):
        engine = GatingEngine(graph_store)
        result = engine.evaluate_input(
            "Some generic content here",
            memory_type=MemoryType.PROCEDURAL,
        )
        assert result.metadata.memory_type == MemoryType.PROCEDURAL


class TestSalience:
    def test_high_salience_keywords(self, graph_store):
        engine = GatingEngine(graph_store)
        result = engine.evaluate_input(
            "Critical bug found in the authentication system! This is important."
        )
        assert result.metadata.salience > 0.6

    def test_low_salience_generic(self, graph_store):
        engine = GatingEngine(graph_store)
        result = engine.evaluate_input(
            "Just some regular content about regular things"
        )
        assert result.metadata.salience <= 0.6

    def test_explicit_salience_overrides(self, graph_store):
        engine = GatingEngine(graph_store)
        result = engine.evaluate_input(
            "Some content", salience=0.95,
        )
        assert result.metadata.salience == 0.95

    def test_tags_boost_salience(self, graph_store):
        engine = GatingEngine(graph_store)
        with_tags = engine.evaluate_input(
            "Some content about things",
            tags=["important", "critical", "security"],
        )
        without_tags = engine.evaluate_input(
            "Some content about stuff",
        )
        assert with_tags.metadata.salience > without_tags.metadata.salience


class TestLayerAssignment:
    def test_procedural_gets_working(self, graph_store):
        engine = GatingEngine(graph_store)
        result = engine.evaluate_input(
            "How to debug async code: check event loop first",
            memory_type=MemoryType.PROCEDURAL,
        )
        assert result.metadata.layer == MemoryLayer.WORKING

    def test_high_salience_gets_working(self, graph_store):
        engine = GatingEngine(graph_store)
        result = engine.evaluate_input(
            "Some regular content here", salience=0.8,
        )
        assert result.metadata.layer == MemoryLayer.WORKING

    def test_low_salience_gets_sensory(self, graph_store):
        engine = GatingEngine(graph_store)
        result = engine.evaluate_input(
            "Some regular content here", salience=0.2,
        )
        assert result.metadata.layer == MemoryLayer.SENSORY
