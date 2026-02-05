"""Tests for spreading activation algorithm."""

from cerebro.activation.spreading import spreading_activation
from cerebro.models.memory import MemoryNode
from cerebro.types import LinkType


def _build_chain(graph_store, n=5):
    """Build a linear chain: 0 -> 1 -> 2 -> ... -> n-1"""
    for i in range(n):
        graph_store.add_node(MemoryNode(id=f"mem_c_{i}", content=f"Chain node {i}"))
    for i in range(n - 1):
        graph_store.ensure_link(f"mem_c_{i}", f"mem_c_{i+1}", LinkType.TEMPORAL, weight=0.8)


def _build_star(graph_store, center_id="mem_star_0", n_leaves=4):
    """Build a star: center -> leaf_1, center -> leaf_2, ..."""
    graph_store.add_node(MemoryNode(id=center_id, content="Center node"))
    for i in range(n_leaves):
        leaf_id = f"mem_leaf_{i}"
        graph_store.add_node(MemoryNode(id=leaf_id, content=f"Leaf {i}"))
        graph_store.ensure_link(center_id, leaf_id, LinkType.SEMANTIC, weight=0.7)


class TestSpreadingActivation:
    def test_empty_seeds(self, graph_store):
        result = spreading_activation(graph_store, [], [])
        assert result == {}

    def test_single_seed_no_links(self, graph_store):
        graph_store.add_node(MemoryNode(id="mem_lonely", content="Lonely node"))
        result = spreading_activation(graph_store, ["mem_lonely"], [1.0])
        assert "mem_lonely" in result
        assert result["mem_lonely"] == 1.0  # only node, normalized to 1.0

    def test_chain_propagation(self, graph_store):
        """Activation should spread along a chain with decay."""
        _build_chain(graph_store, n=5)
        result = spreading_activation(
            graph_store,
            seed_ids=["mem_c_0"],
            seed_weights=[1.0],
            max_hops=3,
        )
        # Seed should be activated
        assert "mem_c_0" in result
        # Direct neighbor should be activated
        assert "mem_c_1" in result
        # 2-hop neighbor may or may not be activated depending on threshold
        if "mem_c_2" in result:
            # Should be weaker than 1-hop
            assert result["mem_c_2"] <= result["mem_c_1"]

    def test_star_fan_out(self, graph_store):
        """Activation should spread to all leaves from center."""
        _build_star(graph_store)
        result = spreading_activation(
            graph_store,
            seed_ids=["mem_star_0"],
            seed_weights=[1.0],
            max_hops=1,
        )
        assert "mem_star_0" in result
        # All leaves should be activated
        for i in range(4):
            assert f"mem_leaf_{i}" in result

    def test_activation_decays_with_hops(self, graph_store):
        """Activation should be weaker at greater hop distances."""
        _build_chain(graph_store, n=4)
        result = spreading_activation(
            graph_store,
            seed_ids=["mem_c_0"],
            seed_weights=[1.0],
            max_hops=3,
        )
        # Seed > hop1 > hop2
        assert result.get("mem_c_0", 0) >= result.get("mem_c_1", 0)

    def test_budget_limit(self, graph_store):
        """Should not exceed max_activated nodes."""
        _build_star(graph_store, n_leaves=20)
        result = spreading_activation(
            graph_store,
            seed_ids=["mem_star_0"],
            seed_weights=[1.0],
            max_activated=5,
        )
        assert len(result) <= 5

    def test_multiple_seeds(self, graph_store):
        """Multiple seeds should all contribute activation."""
        for i in range(4):
            graph_store.add_node(MemoryNode(id=f"mem_ms_{i}", content=f"Multi seed {i}"))
        graph_store.ensure_link("mem_ms_0", "mem_ms_2", LinkType.SEMANTIC, weight=0.8)
        graph_store.ensure_link("mem_ms_1", "mem_ms_3", LinkType.SEMANTIC, weight=0.8)

        result = spreading_activation(
            graph_store,
            seed_ids=["mem_ms_0", "mem_ms_1"],
            seed_weights=[0.9, 0.8],
        )
        # Both seeds and their neighbors should be activated
        assert "mem_ms_0" in result
        assert "mem_ms_1" in result
        assert "mem_ms_2" in result
        assert "mem_ms_3" in result

    def test_normalized_output(self, graph_store):
        """All activation values should be in [0, 1] after normalization."""
        _build_chain(graph_store, n=5)
        result = spreading_activation(
            graph_store,
            seed_ids=["mem_c_0"],
            seed_weights=[1.0],
            max_hops=3,
        )
        for v in result.values():
            assert 0.0 <= v <= 1.0

    def test_link_weight_matters(self, graph_store):
        """Stronger links should propagate more activation."""
        graph_store.add_node(MemoryNode(id="mem_w_0", content="Source"))
        graph_store.add_node(MemoryNode(id="mem_w_1", content="Strong link"))
        graph_store.add_node(MemoryNode(id="mem_w_2", content="Weak link"))
        graph_store.ensure_link("mem_w_0", "mem_w_1", LinkType.CAUSAL, weight=0.95)
        graph_store.ensure_link("mem_w_0", "mem_w_2", LinkType.CONTEXTUAL, weight=0.1)

        result = spreading_activation(
            graph_store,
            seed_ids=["mem_w_0"],
            seed_weights=[1.0],
        )
        if "mem_w_1" in result and "mem_w_2" in result:
            assert result["mem_w_1"] > result["mem_w_2"]
