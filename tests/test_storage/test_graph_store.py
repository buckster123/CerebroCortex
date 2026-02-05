"""Tests for GraphStore (SQLite + igraph hybrid)."""

from datetime import datetime, timedelta

from cerebro.models.agent import AgentProfile
from cerebro.models.episode import Episode, EpisodeStep
from cerebro.models.link import AssociativeLink
from cerebro.models.memory import MemoryMetadata, MemoryNode
from cerebro.types import EmotionalValence, LinkType, MemoryType


class TestGraphStoreNodes:
    def test_add_and_get_node(self, graph_store):
        node = MemoryNode(id="mem_test_1", content="Python is great")
        graph_store.add_node(node)

        retrieved = graph_store.get_node("mem_test_1")
        assert retrieved is not None
        assert retrieved.id == "mem_test_1"
        assert retrieved.metadata.memory_type == MemoryType.SEMANTIC

    def test_node_in_igraph(self, graph_store):
        node = MemoryNode(id="mem_ig_1", content="Test igraph")
        graph_store.add_node(node)

        assert graph_store.graph.vcount() == 1
        assert graph_store._id_to_vertex["mem_ig_1"] == 0

    def test_delete_node(self, graph_store):
        node = MemoryNode(id="mem_del_1", content="To be deleted")
        graph_store.add_node(node)
        assert graph_store.count_nodes() == 1

        result = graph_store.delete_node("mem_del_1")
        assert result is True
        assert graph_store.count_nodes() == 0
        assert graph_store.get_node("mem_del_1") is None

    def test_duplicate_detection(self, graph_store):
        node = MemoryNode(id="mem_dup_1", content="Unique content here")
        graph_store.add_node(node)

        assert graph_store.find_duplicate_content("Unique content here") == "mem_dup_1"
        assert graph_store.find_duplicate_content("Different content") is None

    def test_update_strength(self, graph_store):
        node = MemoryNode(id="mem_str_1", content="Test strength")
        graph_store.add_node(node)

        from cerebro.models.memory import StrengthState
        new_strength = StrengthState(
            stability=5.0,
            difficulty=3.0,
            access_count=10,
            access_timestamps=[1000.0, 2000.0, 3000.0],
        )
        graph_store.update_node_strength("mem_str_1", new_strength)

        updated = graph_store.get_node("mem_str_1")
        assert updated.strength.stability == 5.0
        assert updated.strength.difficulty == 3.0
        assert updated.strength.access_count == 10
        assert len(updated.strength.access_timestamps) == 3

    def test_get_nodes_since(self, graph_store):
        past = datetime.now() - timedelta(hours=1)
        n1 = MemoryNode(id="mem_old", content="Old memory", created_at=past)
        n2 = MemoryNode(id="mem_new", content="New memory")
        graph_store.add_node(n1)
        graph_store.add_node(n2)

        recent = graph_store.get_nodes_since(datetime.now() - timedelta(seconds=5))
        ids = [n.id for n in recent]
        assert "mem_new" in ids

    def test_count_and_list(self, graph_store):
        for i in range(5):
            graph_store.add_node(MemoryNode(id=f"mem_c_{i}", content=f"Content {i}"))

        assert graph_store.count_nodes() == 5
        all_ids = graph_store.get_all_node_ids()
        assert len(all_ids) == 5


class TestGraphStoreLinks:
    def _add_three_nodes(self, store):
        for i in range(3):
            store.add_node(MemoryNode(id=f"mem_l_{i}", content=f"Node {i}"))

    def test_add_link(self, graph_store):
        self._add_three_nodes(graph_store)
        graph_store.ensure_link("mem_l_0", "mem_l_1", LinkType.SEMANTIC, weight=0.8)

        link = graph_store.get_link("mem_l_0", "mem_l_1", LinkType.SEMANTIC)
        assert link is not None
        assert link.weight == 0.8
        assert link.link_type == LinkType.SEMANTIC

    def test_link_in_igraph(self, graph_store):
        self._add_three_nodes(graph_store)
        graph_store.ensure_link("mem_l_0", "mem_l_1", LinkType.CAUSAL, weight=0.9)

        assert graph_store.graph.ecount() == 1
        assert graph_store.graph.es[0]["weight"] == 0.9
        assert graph_store.graph.es[0]["link_type"] == "causal"

    def test_neighbors(self, graph_store):
        self._add_three_nodes(graph_store)
        graph_store.ensure_link("mem_l_0", "mem_l_1", LinkType.SEMANTIC, weight=0.8)
        graph_store.ensure_link("mem_l_0", "mem_l_2", LinkType.CAUSAL, weight=0.6)

        neighbors = graph_store.get_neighbors("mem_l_0")
        assert len(neighbors) == 2
        neighbor_ids = [n[0] for n in neighbors]
        assert "mem_l_1" in neighbor_ids
        assert "mem_l_2" in neighbor_ids

    def test_neighbors_filtered_by_type(self, graph_store):
        self._add_three_nodes(graph_store)
        graph_store.ensure_link("mem_l_0", "mem_l_1", LinkType.SEMANTIC, weight=0.8)
        graph_store.ensure_link("mem_l_0", "mem_l_2", LinkType.CAUSAL, weight=0.6)

        neighbors = graph_store.get_neighbors("mem_l_0", link_types=[LinkType.CAUSAL])
        assert len(neighbors) == 1
        assert neighbors[0][0] == "mem_l_2"

    def test_neighbors_filtered_by_weight(self, graph_store):
        self._add_three_nodes(graph_store)
        graph_store.ensure_link("mem_l_0", "mem_l_1", LinkType.SEMANTIC, weight=0.9)
        graph_store.ensure_link("mem_l_0", "mem_l_2", LinkType.SEMANTIC, weight=0.2)

        neighbors = graph_store.get_neighbors("mem_l_0", min_weight=0.5)
        assert len(neighbors) == 1
        assert neighbors[0][0] == "mem_l_1"

    def test_hebbian_strengthening(self, graph_store):
        self._add_three_nodes(graph_store)
        graph_store.ensure_link("mem_l_0", "mem_l_1", LinkType.SEMANTIC, weight=0.5)

        graph_store.strengthen_link("mem_l_0", "mem_l_1", boost=0.2)
        link = graph_store.get_link("mem_l_0", "mem_l_1", LinkType.SEMANTIC)
        assert link.weight == 0.7

    def test_hebbian_capped_at_1(self, graph_store):
        self._add_three_nodes(graph_store)
        graph_store.ensure_link("mem_l_0", "mem_l_1", LinkType.SEMANTIC, weight=0.9)

        graph_store.strengthen_link("mem_l_0", "mem_l_1", boost=0.5)
        link = graph_store.get_link("mem_l_0", "mem_l_1", LinkType.SEMANTIC)
        assert link.weight == 1.0

    def test_upsert_link(self, graph_store):
        """Adding same link type twice should update, not duplicate."""
        self._add_three_nodes(graph_store)
        graph_store.ensure_link("mem_l_0", "mem_l_1", LinkType.SEMANTIC, weight=0.5)
        graph_store.ensure_link("mem_l_0", "mem_l_1", LinkType.SEMANTIC, weight=0.8)

        # Should have only 1 link, with max weight
        assert graph_store.count_links() == 1
        link = graph_store.get_link("mem_l_0", "mem_l_1", LinkType.SEMANTIC)
        assert link.weight == 0.8

    def test_degree(self, graph_store):
        self._add_three_nodes(graph_store)
        graph_store.ensure_link("mem_l_0", "mem_l_1", LinkType.SEMANTIC)
        graph_store.ensure_link("mem_l_0", "mem_l_2", LinkType.CAUSAL)

        assert graph_store.get_degree("mem_l_0") == 2
        assert graph_store.get_degree("mem_l_1") == 1
        assert graph_store.get_degree("mem_l_2") == 1

    def test_has_link(self, graph_store):
        self._add_three_nodes(graph_store)
        graph_store.ensure_link("mem_l_0", "mem_l_1", LinkType.SEMANTIC)

        assert graph_store.has_link("mem_l_0", "mem_l_1") is True
        assert graph_store.has_link("mem_l_0", "mem_l_2") is False

    def test_cascade_delete(self, graph_store):
        """Deleting a node should cascade-delete its links."""
        self._add_three_nodes(graph_store)
        graph_store.ensure_link("mem_l_0", "mem_l_1", LinkType.SEMANTIC)
        graph_store.ensure_link("mem_l_0", "mem_l_2", LinkType.CAUSAL)
        assert graph_store.count_links() == 2

        graph_store.delete_node("mem_l_0")
        assert graph_store.count_links() == 0


class TestGraphStoreEpisodes:
    def test_create_episode(self, graph_store):
        ep = Episode(id="ep_test_1", title="Debugging Session")
        graph_store.add_episode(ep)

        retrieved = graph_store.get_episode("ep_test_1")
        assert retrieved is not None
        assert retrieved.title == "Debugging Session"
        assert retrieved.consolidated is False

    def test_episode_with_steps(self, graph_store):
        # Add memory nodes first
        for i in range(3):
            graph_store.add_node(MemoryNode(id=f"mem_ep_{i}", content=f"Step {i}"))

        ep = Episode(id="ep_steps_1", title="Build Episode")
        graph_store.add_episode(ep)

        for i in range(3):
            step = EpisodeStep(memory_id=f"mem_ep_{i}", position=i, role="event")
            graph_store.add_episode_step("ep_steps_1", step)

        retrieved = graph_store.get_episode("ep_steps_1")
        assert len(retrieved.steps) == 3
        assert retrieved.steps[0].memory_id == "mem_ep_0"
        assert retrieved.steps[2].position == 2

    def test_unconsolidated_episodes(self, graph_store):
        ep1 = Episode(id="ep_uc_1", title="Not consolidated")
        ep2 = Episode(id="ep_uc_2", title="Also not consolidated")
        graph_store.add_episode(ep1)
        graph_store.add_episode(ep2)

        uncons = graph_store.get_unconsolidated_episodes()
        assert len(uncons) == 2


class TestGraphStoreAgents:
    def test_register_agent(self, graph_store):
        agent = AgentProfile(id="TEST", display_name="Test Agent", specialization="Testing")
        graph_store.register_agent(agent)

        agents = graph_store.list_agents()
        assert len(agents) == 1
        assert agents[0].id == "TEST"
        assert agents[0].display_name == "Test Agent"

    def test_update_agent(self, graph_store):
        agent = AgentProfile(id="TEST", display_name="Old Name")
        graph_store.register_agent(agent)

        agent.display_name = "New Name"
        graph_store.register_agent(agent)

        agents = graph_store.list_agents()
        assert len(agents) == 1
        assert agents[0].display_name == "New Name"


class TestGraphStoreResync:
    def test_resync_preserves_graph(self, graph_store):
        for i in range(3):
            graph_store.add_node(MemoryNode(id=f"mem_r_{i}", content=f"Node {i}"))
        graph_store.ensure_link("mem_r_0", "mem_r_1", LinkType.SEMANTIC, weight=0.8)
        graph_store.ensure_link("mem_r_1", "mem_r_2", LinkType.TEMPORAL, weight=0.6)

        # Resync should rebuild igraph from SQLite
        graph_store.resync_igraph()

        assert graph_store.graph.vcount() == 3
        assert graph_store.graph.ecount() == 2

        neighbors = graph_store.get_neighbors("mem_r_0")
        assert len(neighbors) == 1
        assert neighbors[0][0] == "mem_r_1"


class TestGraphStoreStats:
    def test_stats(self, graph_store):
        graph_store.add_node(MemoryNode(
            id="mem_s_1", content="Semantic fact",
            metadata=MemoryMetadata(memory_type=MemoryType.SEMANTIC),
        ))
        graph_store.add_node(MemoryNode(
            id="mem_s_2", content="Procedure step",
            metadata=MemoryMetadata(memory_type=MemoryType.PROCEDURAL),
        ))
        graph_store.ensure_link("mem_s_1", "mem_s_2", LinkType.SEMANTIC)

        stats = graph_store.stats()
        assert stats["nodes"] == 2
        assert stats["links"] == 1
        assert stats["memory_types"]["semantic"] == 1
        assert stats["memory_types"]["procedural"] == 1
        assert stats["igraph_vertices"] == 2
        assert stats["igraph_edges"] == 1
