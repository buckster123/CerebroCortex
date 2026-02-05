"""Tests for the EpisodicEngine (hippocampus)."""

from cerebro.engines.hippocampus import EpisodicEngine
from cerebro.models.memory import MemoryNode
from cerebro.types import EmotionalValence, LinkType


class TestEpisodeLifecycle:
    def test_start_episode(self, graph_store):
        engine = EpisodicEngine(graph_store)
        episode = engine.start_episode(title="Test Episode")
        assert episode.id is not None
        assert episode.title == "Test Episode"
        assert episode.started_at is not None

    def test_add_step(self, graph_store):
        engine = EpisodicEngine(graph_store)
        episode = engine.start_episode(title="Step test")

        # Add some memories
        graph_store.add_node(MemoryNode(id="mem_ep_s1", content="First event"))
        graph_store.add_node(MemoryNode(id="mem_ep_s2", content="Second event"))

        step1 = engine.add_step(episode.id, "mem_ep_s1")
        step2 = engine.add_step(episode.id, "mem_ep_s2")

        assert step1.position == 0
        assert step2.position == 1

    def test_temporal_links_created(self, graph_store):
        engine = EpisodicEngine(graph_store)
        episode = engine.start_episode(title="Temporal links")

        graph_store.add_node(MemoryNode(id="mem_tl1", content="First"))
        graph_store.add_node(MemoryNode(id="mem_tl2", content="Second"))

        engine.add_step(episode.id, "mem_tl1")
        engine.add_step(episode.id, "mem_tl2")

        # Should have created temporal link from first to second
        assert graph_store.has_link("mem_tl1", "mem_tl2")

    def test_end_episode(self, graph_store):
        engine = EpisodicEngine(graph_store)
        episode = engine.start_episode(title="Ending test")
        ended = engine.end_episode(
            episode.id,
            summary="It was a good episode",
            valence=EmotionalValence.POSITIVE,
        )
        assert ended is not None
        assert ended.ended_at is not None
        assert ended.overall_valence == EmotionalValence.POSITIVE.value

    def test_get_episode_memories(self, graph_store):
        engine = EpisodicEngine(graph_store)
        episode = engine.start_episode(title="Memory list test")

        for i in range(3):
            graph_store.add_node(MemoryNode(id=f"mem_epm_{i}", content=f"Event {i}"))
            engine.add_step(episode.id, f"mem_epm_{i}")

        memories = engine.get_episode_memories(episode.id)
        assert len(memories) == 3
        assert memories[0] == "mem_epm_0"
        assert memories[2] == "mem_epm_2"


class TestSessionEpisodes:
    def test_auto_create_episode_for_session(self, graph_store):
        engine = EpisodicEngine(graph_store)
        graph_store.add_node(MemoryNode(id="mem_ses1", content="Session event"))

        step = engine.add_to_current_episode("session_123", "mem_ses1")
        assert step is not None

        episode = engine.get_active_episode("session_123")
        assert episode is not None

    def test_reuse_existing_episode(self, graph_store):
        engine = EpisodicEngine(graph_store)
        graph_store.add_node(MemoryNode(id="mem_ses2a", content="Event A"))
        graph_store.add_node(MemoryNode(id="mem_ses2b", content="Event B"))

        engine.add_to_current_episode("session_456", "mem_ses2a")
        engine.add_to_current_episode("session_456", "mem_ses2b")

        episode = engine.get_active_episode("session_456")
        assert episode is not None
        assert len(episode.steps) == 2


class TestConsolidation:
    def test_get_unconsolidated(self, graph_store):
        engine = EpisodicEngine(graph_store)
        engine.start_episode(title="Unconsolidated 1")
        engine.start_episode(title="Unconsolidated 2")

        episodes = engine.get_unconsolidated()
        assert len(episodes) >= 2

    def test_mark_consolidated(self, graph_store):
        engine = EpisodicEngine(graph_store)
        episode = engine.start_episode(title="To consolidate")

        result = engine.mark_consolidated(episode.id)
        assert result is True

        # Should no longer appear in unconsolidated
        episodes = engine.get_unconsolidated()
        consolidated_ids = [e.id for e in episodes]
        assert episode.id not in consolidated_ids


class TestRecentEpisodes:
    def test_get_recent(self, graph_store):
        engine = EpisodicEngine(graph_store)
        for i in range(5):
            engine.start_episode(title=f"Recent {i}")

        recent = engine.get_recent_episodes(limit=3)
        assert len(recent) == 3
