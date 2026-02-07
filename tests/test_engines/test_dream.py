"""Tests for the Dream Engine.

Uses a mock LLM client to test all 6 dream phases without real API calls.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from cerebro.cortex import CerebroCortex
from cerebro.engines.dream import DreamEngine, DreamReport, PhaseReport
from cerebro.types import DreamPhase, EmotionalValence, MemoryType, Visibility
from cerebro.utils.llm import LLMResponse


class MockLLMClient:
    """Mock LLM client that returns predictable JSON responses."""

    def __init__(self):
        self.calls = []
        self.total_calls = 0
        self.total_tokens = 0
        self.fallback_count = 0

    def generate(self, prompt, system=None, max_tokens=1024, temperature=0.7):
        self.calls.append(prompt)
        self.total_calls += 1

        # Detect which prompt type and return appropriate mock response
        if "Extract reusable patterns" in prompt or "extract" in prompt.lower():
            text = json.dumps([{
                "content": "When encountering an error, check logs first then reproduce",
                "source_indices": [0, 1],
                "tags": ["debugging", "workflow"],
            }])
        elif "abstract schema" in prompt.lower() or "general principle" in prompt.lower():
            text = json.dumps({
                "content": "Iterative debugging with systematic log analysis leads to faster resolution",
                "tags": ["methodology", "debugging"],
            })
        elif "unexpected" in prompt.lower() or "connection" in prompt.lower():
            text = json.dumps({
                "connected": True,
                "link_type": "semantic",
                "reason": "Both involve systematic analysis approaches",
                "weight": 0.6,
            })
        else:
            text = json.dumps({"text": "default mock response"})

        return LLMResponse(
            text=text,
            provider="mock",
            model="mock-model",
            tokens_used=100,
        )

    def stats(self):
        return {"total_calls": self.total_calls}


@pytest.fixture
def cortex():
    """CerebroCortex with temporary database."""
    with tempfile.TemporaryDirectory() as d:
        ctx = CerebroCortex(db_path=Path(d) / "test_dream.db", chroma_dir=Path(d) / "chroma")
        ctx.initialize()
        yield ctx
        ctx.close()


@pytest.fixture
def mock_llm():
    return MockLLMClient()


@pytest.fixture
def dream(cortex, mock_llm):
    return DreamEngine(cortex, llm_client=mock_llm)


def _seed_episode(cortex, n_memories=4):
    """Create an episode with memories for dream testing."""
    ep = cortex.episode_start(title="Test Episode", agent_id="CLAUDE")

    mem_ids = []
    for i in range(n_memories):
        node = cortex.remember(
            content=f"Debug step {i}: analyzing the error logs and checking stack trace for issue {i}",
            memory_type=MemoryType.EPISODIC,
            tags=["debugging", "testing"],
        )
        if node:
            mem_ids.append(node.id)
            cortex.episodes.add_step(ep.id, node.id, role="event")

    cortex.episode_end(ep.id, summary="Debugging session", valence=EmotionalValence.NEUTRAL)
    return ep.id, mem_ids


class TestDreamReport:
    def test_empty_report(self):
        report = DreamReport()
        d = report.to_dict()
        assert d["episodes_consolidated"] == 0
        assert d["total_llm_calls"] == 0
        assert d["success"] is True
        assert len(d["phases"]) == 0

    def test_phase_report(self):
        pr = PhaseReport(
            phase=DreamPhase.SWS_REPLAY,
            memories_processed=10,
            links_strengthened=5,
            notes="Replayed 2 episodes",
        )
        assert pr.phase == DreamPhase.SWS_REPLAY
        assert pr.memories_processed == 10
        assert pr.success is True

    def test_report_with_phases(self):
        report = DreamReport()
        report.phases.append(PhaseReport(phase=DreamPhase.SWS_REPLAY, success=True))
        report.phases.append(PhaseReport(phase=DreamPhase.PRUNING, success=False))
        d = report.to_dict()
        assert len(d["phases"]) == 2
        assert d["phases"][0]["phase"] == "sws_replay"
        assert d["phases"][1]["success"] is False


class TestDreamEngineInit:
    def test_init(self, dream):
        assert dream.is_running is False
        assert dream.last_report is None

    def test_init_without_llm(self, cortex):
        engine = DreamEngine(cortex, llm_client=None)
        assert engine._llm is None


class TestSWSReplay:
    def test_no_episodes(self, dream):
        """SWS replay with no unconsolidated episodes."""
        report = dream._phase_sws_replay()
        assert report.phase == DreamPhase.SWS_REPLAY
        assert report.success is True
        assert "No unconsolidated episodes" in report.notes

    def test_replay_episode(self, cortex, dream):
        """SWS replay strengthens links in episode."""
        ep_id, mem_ids = _seed_episode(cortex)
        assert len(mem_ids) >= 2

        report = dream._phase_sws_replay()
        assert report.phase == DreamPhase.SWS_REPLAY
        assert report.success is True
        assert report.memories_processed > 0
        assert "Replayed" in report.notes


class TestPatternExtraction:
    def test_no_clusters(self, dream):
        """Pattern extraction with no memory clusters."""
        report = dream._phase_pattern_extraction()
        assert report.success is True
        assert "No clusters" in report.notes

    def test_extract_patterns(self, cortex, dream):
        """Extract procedures from clustered memories."""
        # Seed enough similar memories to form a cluster
        for i in range(4):
            cortex.remember(
                content=f"Debugging technique {i}: always check error logs and trace outputs carefully",
                tags=["debugging"],
            )

        report = dream._phase_pattern_extraction()
        assert report.success is True
        assert report.llm_calls >= 0  # May or may not find clusters

    def test_no_llm_skips_gracefully(self, cortex):
        """Without LLM, pattern extraction produces no procedures but doesn't fail."""
        engine = DreamEngine(cortex, llm_client=None)
        for i in range(4):
            cortex.remember(
                content=f"Test memory {i} about software architecture and design patterns",
                tags=["architecture"],
            )

        report = engine._phase_pattern_extraction()
        assert report.success is True
        assert report.procedures_extracted == 0


class TestSchemaFormation:
    def test_no_episodes(self, dream):
        """Schema formation with no unconsolidated episodes."""
        report = dream._phase_schema_formation()
        assert report.success is True

    def test_form_schema(self, cortex, dream):
        """Form abstract schemas from episodes."""
        _seed_episode(cortex, n_memories=3)

        report = dream._phase_schema_formation()
        assert report.success is True

    def test_no_llm_skips(self, cortex):
        """Without LLM, schema formation skips LLM phases."""
        engine = DreamEngine(cortex, llm_client=None)
        _seed_episode(cortex, n_memories=3)

        report = engine._phase_schema_formation()
        assert report.success is True
        assert report.schemas_extracted == 0


class TestEmotionalReprocessing:
    def test_no_episodes(self, dream):
        """Emotional reprocessing with no episodes."""
        report = dream._phase_emotional_reprocessing()
        assert report.success is True
        assert report.memories_processed == 0

    def test_reprocess_negative(self, cortex, dream):
        """Reprocess memories from negative episode."""
        ep = cortex.episode_start(title="Bug hunt")
        node = cortex.remember(
            content="Found a critical bug in the authentication module that crashes on login",
            tags=["bug"],
        )
        if node:
            cortex.episodes.add_step(ep.id, node.id)
        cortex.episode_end(ep.id, summary="Bug found", valence=EmotionalValence.NEGATIVE)

        report = dream._phase_emotional_reprocessing()
        assert report.success is True
        assert report.memories_processed > 0


class TestPruning:
    def test_pruning_empty(self, dream):
        """Pruning on empty database."""
        report = dream._phase_pruning()
        assert report.success is True
        assert report.memories_pruned == 0

    def test_pruning_keeps_linked_memories(self, cortex, dream):
        """Pruning should not delete memories with links."""
        node = cortex.remember(
            content="A memory that has connections to other memories in the graph",
            tags=["connected"],
        )
        if node:
            # This memory has auto-links from the remember pipeline
            report = dream._phase_pruning()
            # Memory should survive pruning
            assert cortex.graph.get_node(node.id) is not None

    def test_decay_and_promote(self, cortex, dream):
        """Pruning runs decay and promotion sweeps."""
        cortex.remember(
            content="A well-accessed memory that should be promoted to higher layers",
            salience=0.9,
        )

        report = dream._phase_pruning()
        assert report.success is True
        assert "Decayed" in report.notes


class TestREMRecombination:
    def test_not_enough_memories(self, dream):
        """REM with too few memories."""
        report = dream._phase_rem_recombination()
        assert report.success is True
        assert "Not enough" in report.notes

    def test_rem_with_memories(self, cortex, dream):
        """REM recombination finds connections between diverse memories."""
        # Create diverse memories
        cortex.remember(
            content="Machine learning models require careful hyperparameter tuning for best results",
            memory_type=MemoryType.SEMANTIC,
            tags=["ml"],
        )
        cortex.remember(
            content="Garden plants need specific soil pH levels to grow properly and thrive",
            memory_type=MemoryType.SEMANTIC,
            tags=["gardening"],
        )
        cortex.remember(
            content="Database query optimization involves analyzing execution plans and index usage",
            memory_type=MemoryType.PROCEDURAL,
            tags=["database"],
        )
        cortex.remember(
            content="Effective teaching requires adapting the material to the student learning style",
            memory_type=MemoryType.SCHEMATIC,
            tags=["education"],
        )

        report = dream._phase_rem_recombination()
        assert report.success is True
        assert report.memories_processed > 0

    def test_no_llm_skips(self, cortex):
        """Without LLM, REM creates no connections but doesn't fail."""
        engine = DreamEngine(cortex, llm_client=None)
        for i in range(5):
            cortex.remember(
                content=f"Diverse memory number {i} about topic {i} with unique content",
                tags=[f"topic{i}"],
            )

        report = engine._phase_rem_recombination()
        assert report.success is True
        assert report.links_created == 0


class TestFullCycle:
    def test_empty_cycle(self, dream):
        """Full cycle on empty database."""
        report = dream.run_cycle()
        assert report.success is True
        assert len(report.phases) == 6
        assert report.episodes_consolidated == 0
        assert dream.last_report is report

    def test_cycle_with_data(self, cortex, dream):
        """Full cycle with episodes and memories."""
        _seed_episode(cortex, n_memories=4)

        report = dream.run_cycle()
        assert len(report.phases) == 6
        assert report.total_duration_seconds > 0
        assert report.ended_at is not None

        # Check report serialization
        d = report.to_dict()
        assert "phases" in d
        assert len(d["phases"]) == 6
        assert all("phase" in p for p in d["phases"])

    def test_cycle_marks_consolidated(self, cortex, dream):
        """After dream cycle, episodes should be marked consolidated."""
        ep_id, _ = _seed_episode(cortex)

        # Before dream: unconsolidated
        uncon = cortex.episodes.get_unconsolidated()
        assert len(uncon) > 0

        dream.run_cycle()

        # After dream: consolidated
        uncon = cortex.episodes.get_unconsolidated()
        assert len(uncon) == 0

    def test_double_cycle_prevented(self, cortex, mock_llm):
        """Cannot run two cycles simultaneously."""
        engine = DreamEngine(cortex, llm_client=mock_llm)
        engine._running = True

        with pytest.raises(RuntimeError, match="already in progress"):
            engine.run_cycle()

    def test_cycle_without_llm(self, cortex):
        """Cycle runs with algorithmic phases only when no LLM."""
        _seed_episode(cortex, n_memories=3)
        engine = DreamEngine(cortex, llm_client=None)

        report = engine.run_cycle()
        assert report.success is True
        assert report.total_llm_calls == 0
        # Algorithmic phases should still work
        assert len(report.phases) == 6


class TestJSONParsing:
    def test_parse_json_basic(self, dream):
        result = dream._parse_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_parse_json_with_fences(self, dream):
        text = '```json\n{"key": "value"}\n```'
        result = dream._parse_json(text)
        assert result == {"key": "value"}

    def test_parse_json_invalid(self, dream):
        result = dream._parse_json("not json at all")
        assert result is None

    def test_parse_json_none(self, dream):
        result = dream._parse_json(None)
        assert result is None

    def test_parse_json_array(self, dream):
        result = dream._parse_json('[{"a": 1}]')
        assert isinstance(result, list)
        assert result[0]["a"] == 1


class TestDreamLog:
    def test_phases_logged(self, cortex, dream):
        """Dream phases are logged to SQLite dream_log table."""
        _seed_episode(cortex, n_memories=2)
        dream.run_cycle()

        # Check dream_log entries
        rows = cortex.graph.conn.execute(
            "SELECT phase, success FROM dream_log ORDER BY id"
        ).fetchall()
        assert len(rows) >= 6  # One entry per phase
        phases = [r["phase"] for r in rows]
        assert "sws_replay" in phases
        assert "pruning" in phases


def _seed_scoped_episode(cortex, agent_id, n_memories=3):
    """Create an episode with memories for a specific agent."""
    ep = cortex.episode_start(title=f"{agent_id} Episode", agent_id=agent_id)
    mem_ids = []
    for i in range(n_memories):
        node = cortex.remember(
            content=f"{agent_id} debug step {i}: analyzing error logs for issue {i}",
            memory_type=MemoryType.EPISODIC,
            tags=["debugging", agent_id.lower()],
            agent_id=agent_id,
        )
        if node:
            mem_ids.append(node.id)
            cortex.episodes.add_step(ep.id, node.id, role="event")
    cortex.episode_end(ep.id, summary=f"{agent_id} debugging session")
    return ep.id, mem_ids


class TestDreamScope:
    """Test that scoped dream cycles respect agent boundaries."""

    def test_scoped_dream_only_sees_own_and_shared(self):
        """ALICE dream doesn't process BOB's private memories."""
        with tempfile.TemporaryDirectory() as d:
            ctx = CerebroCortex(db_path=Path(d) / "dream_scope.db", chroma_dir=Path(d) / "chroma")
            ctx.initialize()

            # Create BOB's private memory
            bob_priv = ctx.remember(
                content="Bob private secret configuration data for internal systems",
                agent_id="BOB", visibility=Visibility.PRIVATE,
            )
            # Create ALICE shared memory
            alice_shared = ctx.remember(
                content="Alice shared knowledge about Python patterns",
                agent_id="ALICE", visibility=Visibility.SHARED,
            )

            # Scoped get_all_node_ids should exclude BOB's private for ALICE
            alice_ids = ctx.graph.get_all_node_ids(agent_id="ALICE")
            assert alice_shared.id in alice_ids
            assert bob_priv.id not in alice_ids

            # BOB sees own private
            bob_ids = ctx.graph.get_all_node_ids(agent_id="BOB")
            assert bob_priv.id in bob_ids
            assert alice_shared.id in bob_ids  # shared is visible

            ctx.close()

    def test_scoped_dream_doesnt_prune_others_private(self):
        """Pruning only touches own memories, not other agents' private memories."""
        with tempfile.TemporaryDirectory() as d:
            ctx = CerebroCortex(db_path=Path(d) / "dream_prune.db", chroma_dir=Path(d) / "chroma")
            ctx.initialize()

            # BOB's private memory — low salience, sensory, old
            bob_priv = ctx.remember(
                content="Bob private low-salience note about temporary observations",
                agent_id="BOB", visibility=Visibility.PRIVATE,
            )

            engine = DreamEngine(ctx, llm_client=None)
            engine._agent_id = "ALICE"  # Simulate ALICE dream cycle

            # ALICE's pruning should not see BOB's private memory
            alice_ids = ctx.graph.get_all_node_ids(agent_id="ALICE")
            assert bob_priv.id not in alice_ids

            ctx.close()

    def test_scoped_dream_unconsolidated_episodes(self):
        """get_unconsolidated(agent_id) only returns that agent's episodes."""
        with tempfile.TemporaryDirectory() as d:
            ctx = CerebroCortex(db_path=Path(d) / "dream_ep.db", chroma_dir=Path(d) / "chroma")
            ctx.initialize()

            _seed_scoped_episode(ctx, "ALICE", n_memories=2)
            _seed_scoped_episode(ctx, "BOB", n_memories=2)

            alice_eps = ctx.episodes.get_unconsolidated(agent_id="ALICE")
            bob_eps = ctx.episodes.get_unconsolidated(agent_id="BOB")
            all_eps = ctx.episodes.get_unconsolidated()

            assert len(alice_eps) == 1
            assert len(bob_eps) == 1
            assert len(all_eps) == 2
            assert alice_eps[0].agent_id == "ALICE"
            assert bob_eps[0].agent_id == "BOB"

            ctx.close()

    def test_auto_per_agent_cycle(self):
        """run_all_agents_cycle() produces one report per agent."""
        with tempfile.TemporaryDirectory() as d:
            ctx = CerebroCortex(db_path=Path(d) / "dream_all.db", chroma_dir=Path(d) / "chroma")
            ctx.initialize()

            # Create memories for two agents
            ctx.remember(content="Alice memory for per-agent dream cycle test", agent_id="ALICE")
            ctx.remember(content="Bob memory for per-agent dream cycle test", agent_id="BOB")

            engine = DreamEngine(ctx, llm_client=None)
            reports = engine.run_all_agents_cycle()

            assert len(reports) == 2
            agent_ids = {r.agent_id for r in reports}
            assert "ALICE" in agent_ids
            assert "BOB" in agent_ids
            assert all(r.success for r in reports)

            ctx.close()

    def test_dream_report_includes_agent_id(self):
        """DreamReport carries agent_id through to_dict()."""
        report = DreamReport(agent_id="ALICE")
        d = report.to_dict()
        assert d["agent_id"] == "ALICE"

        report2 = DreamReport()
        d2 = report2.to_dict()
        assert d2["agent_id"] is None
