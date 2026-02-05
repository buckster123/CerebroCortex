"""Tests for the LinkEngine (association cortex)."""

from cerebro.engines.association import LinkEngine
from cerebro.models.memory import MemoryMetadata, MemoryNode
from cerebro.types import LinkType, MemoryType


class TestLinkCreation:
    def test_create_link(self, graph_store):
        engine = LinkEngine(graph_store)
        graph_store.add_node(MemoryNode(id="mem_a1", content="Node A"))
        graph_store.add_node(MemoryNode(id="mem_a2", content="Node B"))

        link_id = engine.create_link("mem_a1", "mem_a2", LinkType.SEMANTIC, weight=0.7)
        assert link_id is not None
        assert graph_store.has_link("mem_a1", "mem_a2")

    def test_create_link_strengthens_existing(self, graph_store):
        engine = LinkEngine(graph_store)
        graph_store.add_node(MemoryNode(id="mem_s1", content="Source"))
        graph_store.add_node(MemoryNode(id="mem_s2", content="Target"))

        engine.create_link("mem_s1", "mem_s2", LinkType.CAUSAL, weight=0.5)
        engine.create_link("mem_s1", "mem_s2", LinkType.CAUSAL, weight=0.8)

        link = graph_store.get_link("mem_s1", "mem_s2", LinkType.CAUSAL)
        assert link.weight >= 0.5  # should be at least the max


class TestAutoLinkOnStore:
    def test_auto_link_responding_to(self, graph_store):
        engine = LinkEngine(graph_store)
        parent = MemoryNode(id="mem_parent", content="Parent memory")
        graph_store.add_node(parent)

        child = MemoryNode(
            id="mem_child", content="Response memory",
            metadata=MemoryMetadata(responding_to=["mem_parent"]),
        )
        graph_store.add_node(child)

        created = engine.auto_link_on_store(child)
        assert len(created) >= 1
        assert graph_store.has_link("mem_child", "mem_parent")

    def test_auto_link_derived_from(self, graph_store):
        engine = LinkEngine(graph_store)
        source = MemoryNode(id="mem_src", content="Source memory")
        graph_store.add_node(source)

        derived = MemoryNode(
            id="mem_derived", content="Derived memory",
            metadata=MemoryMetadata(derived_from=["mem_src"]),
        )
        graph_store.add_node(derived)

        created = engine.auto_link_on_store(derived)
        assert len(created) >= 1

    def test_auto_link_context(self, graph_store):
        engine = LinkEngine(graph_store)
        for i in range(3):
            graph_store.add_node(MemoryNode(id=f"mem_ctx_{i}", content=f"Context {i}"))

        node = MemoryNode(id="mem_ctx_new", content="New memory")
        graph_store.add_node(node)

        created = engine.auto_link_on_store(node, context_ids=["mem_ctx_0", "mem_ctx_1"])
        assert len(created) >= 2

    def test_auto_link_shared_tags(self, graph_store):
        engine = LinkEngine(graph_store)
        existing = MemoryNode(
            id="mem_tag1", content="Python list operations",
            metadata=MemoryMetadata(tags=["python", "lists"]),
        )
        graph_store.add_node(existing)

        new = MemoryNode(
            id="mem_tag2", content="Python dict operations",
            metadata=MemoryMetadata(tags=["python", "dicts"]),
        )
        graph_store.add_node(new)

        created = engine.auto_link_on_store(new)
        # Should find shared "python" tag
        assert len(created) >= 1


class TestHebbianLearning:
    def test_strengthen_co_activated(self, graph_store):
        engine = LinkEngine(graph_store)
        graph_store.add_node(MemoryNode(id="mem_h1", content="Node 1"))
        graph_store.add_node(MemoryNode(id="mem_h2", content="Node 2"))
        graph_store.ensure_link("mem_h1", "mem_h2", LinkType.SEMANTIC, weight=0.5)

        original = graph_store.get_link("mem_h1", "mem_h2", LinkType.SEMANTIC)
        original_weight = original.weight

        count = engine.strengthen_co_activated(["mem_h1", "mem_h2"], boost=0.1)
        assert count == 1

        updated = graph_store.get_link("mem_h1", "mem_h2", LinkType.SEMANTIC)
        assert updated.weight > original_weight

    def test_no_link_no_strengthen(self, graph_store):
        engine = LinkEngine(graph_store)
        graph_store.add_node(MemoryNode(id="mem_nl1", content="Isolated 1"))
        graph_store.add_node(MemoryNode(id="mem_nl2", content="Isolated 2"))

        count = engine.strengthen_co_activated(["mem_nl1", "mem_nl2"])
        assert count == 0


class TestSpreadingActivation:
    def test_spread_from_seed(self, graph_store):
        engine = LinkEngine(graph_store)
        graph_store.add_node(MemoryNode(id="mem_sp0", content="Seed"))
        graph_store.add_node(MemoryNode(id="mem_sp1", content="Neighbor"))
        graph_store.ensure_link("mem_sp0", "mem_sp1", LinkType.SEMANTIC, weight=0.8)

        result = engine.spread_activation(["mem_sp0"], [1.0])
        assert "mem_sp0" in result
        assert "mem_sp1" in result


class TestQueries:
    def test_get_neighbors(self, graph_store):
        engine = LinkEngine(graph_store)
        graph_store.add_node(MemoryNode(id="mem_q0", content="Center"))
        graph_store.add_node(MemoryNode(id="mem_q1", content="Neighbor"))
        graph_store.ensure_link("mem_q0", "mem_q1", LinkType.TEMPORAL, weight=0.6)

        neighbors = engine.get_neighbors("mem_q0")
        assert len(neighbors) == 1
        assert neighbors[0][0] == "mem_q1"

    def test_get_strongest_connections(self, graph_store):
        engine = LinkEngine(graph_store)
        graph_store.add_node(MemoryNode(id="mem_sc0", content="Center"))
        for i in range(5):
            graph_store.add_node(MemoryNode(id=f"mem_sc{i+1}", content=f"Node {i}"))
            graph_store.ensure_link("mem_sc0", f"mem_sc{i+1}", LinkType.SEMANTIC, weight=0.1 * (i + 1))

        strongest = engine.get_strongest_connections("mem_sc0", top_n=3)
        assert len(strongest) == 3
        # Should be sorted by weight descending
        assert strongest[0][1] >= strongest[1][1] >= strongest[2][1]

    def test_find_path(self, graph_store):
        engine = LinkEngine(graph_store)
        for i in range(4):
            graph_store.add_node(MemoryNode(id=f"mem_p{i}", content=f"Path {i}"))
        graph_store.ensure_link("mem_p0", "mem_p1", LinkType.TEMPORAL, weight=0.7)
        graph_store.ensure_link("mem_p1", "mem_p2", LinkType.TEMPORAL, weight=0.7)
        graph_store.ensure_link("mem_p2", "mem_p3", LinkType.TEMPORAL, weight=0.7)

        path = engine.find_path("mem_p0", "mem_p3")
        assert path is not None
        assert path[0] == "mem_p0"
        assert path[-1] == "mem_p3"

    def test_find_path_no_connection(self, graph_store):
        engine = LinkEngine(graph_store)
        graph_store.add_node(MemoryNode(id="mem_iso1", content="Isolated 1"))
        graph_store.add_node(MemoryNode(id="mem_iso2", content="Isolated 2"))

        path = engine.find_path("mem_iso1", "mem_iso2")
        assert path is None or path == []

    def test_common_neighbors(self, graph_store):
        engine = LinkEngine(graph_store)
        graph_store.add_node(MemoryNode(id="mem_cn_a", content="Node A"))
        graph_store.add_node(MemoryNode(id="mem_cn_b", content="Node B"))
        graph_store.add_node(MemoryNode(id="mem_cn_c", content="Common"))

        graph_store.ensure_link("mem_cn_a", "mem_cn_c", LinkType.SEMANTIC, weight=0.7)
        graph_store.ensure_link("mem_cn_b", "mem_cn_c", LinkType.SEMANTIC, weight=0.7)

        common = engine.get_common_neighbors("mem_cn_a", "mem_cn_b")
        assert "mem_cn_c" in common
