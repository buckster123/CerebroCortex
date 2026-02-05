"""Tests for the ProceduralEngine (cerebellum)."""

from cerebro.engines.cerebellum import ProceduralEngine
from cerebro.models.memory import MemoryMetadata, MemoryNode
from cerebro.types import MemoryType


class TestStoreProcedure:
    def test_store_basic(self, graph_store):
        engine = ProceduralEngine(graph_store)
        node = engine.store_procedure(
            "When debugging: 1) Check logs, 2) Reproduce issue, 3) Fix and test",
            tags=["debugging", "workflow"],
        )
        assert node.metadata.memory_type == MemoryType.PROCEDURAL
        assert node.metadata.salience == 0.8

    def test_store_with_derivation(self, graph_store):
        engine = ProceduralEngine(graph_store)
        # Add source memories
        graph_store.add_node(MemoryNode(id="mem_proc_src1", content="Debug session 1"))
        graph_store.add_node(MemoryNode(id="mem_proc_src2", content="Debug session 2"))

        node = engine.store_procedure(
            "Extracted debugging strategy",
            derived_from=["mem_proc_src1", "mem_proc_src2"],
        )
        assert node.metadata.source == "consolidation"
        assert len(node.metadata.derived_from) == 2

        # Should have derived_from links
        assert graph_store.has_link(node.id, "mem_proc_src1")
        assert graph_store.has_link(node.id, "mem_proc_src2")


class TestFindProcedures:
    def test_find_by_tags(self, graph_store):
        engine = ProceduralEngine(graph_store)
        engine.store_procedure("Debug async code", tags=["debugging", "async"])
        engine.store_procedure("Deploy with Docker", tags=["deployment", "docker"])

        results = engine.find_relevant_procedures(tags=["debugging"])
        assert len(results) >= 1
        assert any("async" in n.content.lower() or "debug" in n.content.lower() for n in results)

    def test_find_by_concepts(self, graph_store):
        engine = ProceduralEngine(graph_store)
        # Store a procedure with concepts (add directly since store_procedure doesn't set concepts)
        node = MemoryNode(
            id="mem_proc_concept",
            content="How to optimize database queries",
            metadata=MemoryMetadata(
                memory_type=MemoryType.PROCEDURAL,
                concepts=["database", "optimization"],
                salience=0.8,
            ),
        )
        graph_store.add_node(node)

        results = engine.find_relevant_procedures(concepts=["database"])
        assert len(results) >= 1


class TestOutcomeTracking:
    def test_success_boosts_salience(self, graph_store):
        engine = ProceduralEngine(graph_store)
        node = engine.store_procedure("Try this approach", tags=["test"])
        original_salience = node.metadata.salience

        result = engine.record_outcome(node.id, success=True, salience_boost=0.1)
        assert result is True

        updated = graph_store.get_node(node.id)
        assert updated.metadata.salience > original_salience

    def test_failure_increases_difficulty(self, graph_store):
        engine = ProceduralEngine(graph_store)
        node = engine.store_procedure("Risky approach", tags=["test"])
        original_difficulty = node.strength.difficulty

        engine.record_outcome(node.id, success=False)

        updated = graph_store.get_node(node.id)
        assert updated.strength.difficulty > original_difficulty

    def test_outcome_nonexistent(self, graph_store):
        engine = ProceduralEngine(graph_store)
        result = engine.record_outcome("mem_nonexistent", success=True)
        assert result is False


class TestGetAllProcedures:
    def test_get_all(self, graph_store):
        engine = ProceduralEngine(graph_store)
        engine.store_procedure("Procedure 1", tags=["a"])
        engine.store_procedure("Procedure 2", tags=["b"])
        engine.store_procedure("Procedure 3", tags=["c"])

        all_procs = engine.get_all_procedures()
        assert len(all_procs) == 3

    def test_filter_by_salience(self, graph_store):
        engine = ProceduralEngine(graph_store)
        engine.store_procedure("High value procedure", tags=["important"])
        # Lower salience manually
        low = engine.store_procedure("Low value procedure", tags=["meh"])
        graph_store.update_node_metadata(low.id, salience=0.1)

        results = engine.get_all_procedures(min_salience=0.5)
        assert all(
            graph_store.get_node(n.id).metadata.salience >= 0.5
            for n in results
        )
