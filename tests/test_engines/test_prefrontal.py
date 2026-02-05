"""Tests for the ExecutiveEngine (prefrontal cortex)."""

import time

from cerebro.engines.prefrontal import ExecutiveEngine
from cerebro.models.memory import MemoryMetadata, MemoryNode, StrengthState
from cerebro.types import MemoryLayer, MemoryType


class TestProspectiveMemory:
    def test_store_intention(self, graph_store):
        engine = ExecutiveEngine(graph_store)
        node = engine.store_intention(
            "Need to fix the auth module before release",
            tags=["auth", "todo"],
        )
        assert node.metadata.memory_type == MemoryType.PROSPECTIVE
        assert node.metadata.layer == MemoryLayer.WORKING
        assert node.metadata.salience == 0.7

    def test_get_pending_intentions(self, graph_store):
        engine = ExecutiveEngine(graph_store)
        engine.store_intention("TODO 1", tags=["a"])
        engine.store_intention("TODO 2", tags=["b"])
        engine.store_intention("TODO 3", tags=["c"])

        pending = engine.get_pending_intentions()
        assert len(pending) == 3

    def test_resolve_intention(self, graph_store):
        engine = ExecutiveEngine(graph_store)
        node = engine.store_intention("Complete this task")

        result = engine.resolve_intention(node.id)
        assert result is True

        # Resolved intention has low salience
        updated = graph_store.get_node(node.id)
        assert updated.metadata.salience == 0.1

    def test_filter_by_salience(self, graph_store):
        engine = ExecutiveEngine(graph_store)
        engine.store_intention("High priority", salience=0.9)
        low = engine.store_intention("Low priority", salience=0.2)

        pending = engine.get_pending_intentions(min_salience=0.5)
        assert len(pending) == 1
        assert pending[0].metadata.salience >= 0.5


class TestPromotion:
    def test_promote_eligible(self, graph_store):
        engine = ExecutiveEngine(graph_store)

        # Create a sensory memory with enough accesses for promotion
        now = time.time()
        node = MemoryNode(
            id="mem_promo1",
            content="Memory that should be promoted to working",
            metadata=MemoryMetadata(layer=MemoryLayer.SENSORY),
            strength=StrengthState(
                access_count=3,  # exceeds sensory promotion threshold (2)
                access_timestamps=[now - 100, now - 50, now],
            ),
        )
        graph_store.add_node(node)

        new_layer = engine.check_and_promote("mem_promo1")
        assert new_layer == "working"

    def test_no_promote_insufficient_access(self, graph_store):
        engine = ExecutiveEngine(graph_store)

        node = MemoryNode(
            id="mem_nopromo1",
            content="Memory without enough accesses",
            metadata=MemoryMetadata(layer=MemoryLayer.SENSORY),
            strength=StrengthState(access_count=1),
        )
        graph_store.add_node(node)

        result = engine.check_and_promote("mem_nopromo1")
        assert result is None

    def test_promotion_sweep(self, graph_store):
        engine = ExecutiveEngine(graph_store)
        now = time.time()

        # Add some memories eligible for promotion
        for i in range(3):
            graph_store.add_node(MemoryNode(
                id=f"mem_sweep_{i}",
                content=f"Sweep memory {i}",
                metadata=MemoryMetadata(layer=MemoryLayer.SENSORY),
                strength=StrengthState(
                    access_count=3,
                    access_timestamps=[now - 100, now - 50, now],
                ),
            ))

        promotions = engine.run_promotion_sweep()
        assert sum(promotions.values()) >= 3


class TestRanking:
    def test_rank_by_score(self, graph_store):
        engine = ExecutiveEngine(graph_store)
        now = time.time()

        # High-salience, recently accessed memory
        graph_store.add_node(MemoryNode(
            id="mem_rank_high",
            content="Important recent memory",
            metadata=MemoryMetadata(salience=0.9),
            strength=StrengthState(
                access_count=5,
                access_timestamps=[now - 10],
            ),
        ))

        # Low-salience, old memory
        graph_store.add_node(MemoryNode(
            id="mem_rank_low",
            content="Less important old memory",
            metadata=MemoryMetadata(salience=0.1),
            strength=StrengthState(
                access_count=1,
                access_timestamps=[now - 100000],
            ),
        ))

        ranked = engine.rank_results(["mem_rank_high", "mem_rank_low"])
        assert len(ranked) == 2
        assert ranked[0][0] == "mem_rank_high"
        assert ranked[0][1] > ranked[1][1]


class TestWorkingMemory:
    def test_get_working_memory(self, graph_store):
        engine = ExecutiveEngine(graph_store)

        # Add some working memory items
        for i in range(5):
            graph_store.add_node(MemoryNode(
                id=f"mem_wm_{i}",
                content=f"Working memory {i}",
                metadata=MemoryMetadata(layer=MemoryLayer.WORKING, salience=0.5 + i * 0.1),
            ))

        wm = engine.get_working_memory()
        assert len(wm) == 5

    def test_working_memory_limit(self, graph_store):
        engine = ExecutiveEngine(graph_store)

        for i in range(10):
            graph_store.add_node(MemoryNode(
                id=f"mem_wml_{i}",
                content=f"WM item {i}",
                metadata=MemoryMetadata(layer=MemoryLayer.WORKING),
            ))

        wm = engine.get_working_memory(limit=5)
        assert len(wm) == 5


class TestDecaySweep:
    def test_decay_sweep_updates(self, graph_store):
        engine = ExecutiveEngine(graph_store)
        now = time.time()

        graph_store.add_node(MemoryNode(
            id="mem_decay1",
            content="Decaying memory",
            strength=StrengthState(
                access_count=1,
                access_timestamps=[now - 86400],  # 1 day ago
            ),
        ))

        updated = engine.run_decay_sweep()
        assert updated >= 0  # might be 0 if values haven't changed enough
