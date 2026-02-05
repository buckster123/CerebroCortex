"""Tests for the CerebroCortex CLI."""

import tempfile
from pathlib import Path

import pytest
from click.testing import CliRunner

from cerebro.cortex import CerebroCortex
from cerebro.interfaces.cli import cli
import cerebro.interfaces.cli as cli_mod


@pytest.fixture
def cortex():
    """CerebroCortex with temporary database for CLI tests."""
    with tempfile.TemporaryDirectory() as d:
        ctx = CerebroCortex(db_path=Path(d) / "test_cli.db")
        ctx.initialize()
        yield ctx
        ctx.close()


@pytest.fixture
def runner(cortex):
    """Click test runner with injected cortex."""
    original = cli_mod._cortex
    cli_mod._cortex = cortex
    yield CliRunner()
    cli_mod._cortex = original


class TestStats:
    def test_stats_text(self, runner):
        result = runner.invoke(cli, ["stats"])
        assert result.exit_code == 0
        assert "CerebroCortex Statistics" in result.output
        assert "Memories:" in result.output
        assert "Links:" in result.output

    def test_stats_json(self, runner):
        result = runner.invoke(cli, ["stats", "--json"])
        assert result.exit_code == 0
        import json
        data = json.loads(result.output)
        assert "nodes" in data
        assert "links" in data


class TestRemember:
    def test_remember_basic(self, runner):
        result = runner.invoke(cli, [
            "remember", "Python is a dynamically typed programming language",
        ])
        assert result.exit_code == 0
        assert "Stored:" in result.output

    def test_remember_with_type(self, runner):
        result = runner.invoke(cli, [
            "remember", "Step 1: check logs carefully. Step 2: reproduce the issue.",
            "--type", "procedural",
        ])
        assert result.exit_code == 0
        assert "procedural" in result.output

    def test_remember_gated(self, runner):
        result = runner.invoke(cli, ["remember", "hi"])
        assert result.exit_code == 0
        assert "gated out" in result.output

    def test_remember_with_tags(self, runner):
        result = runner.invoke(cli, [
            "remember", "FastAPI uses Starlette under the hood for performance",
            "--tags", "python", "--tags", "web",
        ])
        assert result.exit_code == 0
        assert "Stored:" in result.output

    def test_remember_json(self, runner):
        result = runner.invoke(cli, [
            "remember", "A test memory for JSON output verification purposes",
            "--json",
        ])
        assert result.exit_code == 0
        import json
        data = json.loads(result.output)
        assert data["stored"] is True
        assert "id" in data


class TestRecall:
    def test_recall_empty(self, runner):
        result = runner.invoke(cli, ["recall", "nonexistent"])
        assert result.exit_code == 0
        assert "No memories" in result.output or "Found 0" in result.output

    def test_recall_with_stored(self, runner):
        runner.invoke(cli, ["remember", "Python lists support append extend and insert operations"])
        result = runner.invoke(cli, ["recall", "list operations"])
        assert result.exit_code == 0

    def test_recall_json(self, runner):
        runner.invoke(cli, ["remember", "Redis is an in-memory data structure store for caching"])
        result = runner.invoke(cli, ["recall", "caching", "--json"])
        assert result.exit_code == 0
        import json
        data = json.loads(result.output)
        assert "results" in data

    def test_recall_with_type_filter(self, runner):
        runner.invoke(cli, [
            "remember", "FastAPI is a modern Python web framework for APIs",
            "--type", "semantic",
        ])
        result = runner.invoke(cli, ["recall", "web framework", "--type", "semantic"])
        assert result.exit_code == 0


class TestAssociate:
    def test_create_link(self, runner):
        r1 = runner.invoke(cli, ["remember", "Auth module handles login and sessions flow", "--json"])
        r2 = runner.invoke(cli, ["remember", "Security module manages JWT token encryption", "--json"])

        import json
        id1 = json.loads(r1.output)["id"]
        id2 = json.loads(r2.output)["id"]

        result = runner.invoke(cli, ["associate", id1, id2, "supports"])
        assert result.exit_code == 0
        assert "Link created" in result.output

    def test_link_not_found(self, runner):
        result = runner.invoke(cli, ["associate", "fake1", "fake2", "causal"])
        assert result.exit_code == 1


class TestEpisode:
    def test_episode_start(self, runner):
        result = runner.invoke(cli, ["episode", "start", "--title", "Test Episode"])
        assert result.exit_code == 0
        assert "Episode started:" in result.output

    def test_episode_end_not_found(self, runner):
        result = runner.invoke(cli, ["episode", "end", "nonexistent"])
        assert result.exit_code == 1


class TestSession:
    def test_session_save(self, runner):
        result = runner.invoke(cli, [
            "session", "save", "Built the REST API and CLI",
            "--priority", "HIGH",
            "--type", "technical",
        ])
        assert result.exit_code == 0
        assert "Session saved:" in result.output

    def test_session_recall_empty(self, runner):
        result = runner.invoke(cli, ["session", "recall", "--hours", "1"])
        assert result.exit_code == 0
        assert "No session notes" in result.output

    def test_session_recall_with_data(self, runner):
        runner.invoke(cli, [
            "session", "save", "Previous session about testing the CLI",
        ])
        result = runner.invoke(cli, ["session", "recall", "--hours", "1"])
        assert result.exit_code == 0


class TestAgents:
    def test_list_empty(self, runner):
        result = runner.invoke(cli, ["agents", "list"])
        assert result.exit_code == 0
        assert "No agents" in result.output

    def test_register_and_list(self, runner):
        result = runner.invoke(cli, [
            "agents", "register", "TEST_BOT", "Test Bot",
            "--specialization", "Testing", "--symbol", "T",
        ])
        assert result.exit_code == 0
        assert "Agent registered" in result.output

        result = runner.invoke(cli, ["agents", "list"])
        assert "TEST_BOT" in result.output

    def test_list_json(self, runner):
        runner.invoke(cli, [
            "agents", "register", "JSON_BOT", "JSON Bot",
        ])
        result = runner.invoke(cli, ["agents", "list", "--json"])
        import json
        data = json.loads(result.output)
        assert data["count"] == 1


class TestHealth:
    def test_health_text(self, runner):
        result = runner.invoke(cli, ["health"])
        assert result.exit_code == 0
        assert "CerebroCortex Health Report" in result.output
        assert "Memories:" in result.output

    def test_health_json(self, runner):
        result = runner.invoke(cli, ["health", "--json"])
        assert result.exit_code == 0
        import json
        data = json.loads(result.output)
        assert "memories" in data
        assert "by_type" in data
