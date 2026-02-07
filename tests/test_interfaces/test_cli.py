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
        ctx = CerebroCortex(db_path=Path(d) / "test_cli.db", chroma_dir=Path(d) / "chroma")
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


class TestGetMemory:
    def test_get_basic(self, runner):
        import json as json_mod
        r = runner.invoke(cli, [
            "remember", "Python is a dynamically typed programming language", "--json",
        ])
        mem_id = json_mod.loads(r.output)["id"]

        result = runner.invoke(cli, ["get", mem_id])
        assert result.exit_code == 0
        assert "Python" in result.output
        assert mem_id in result.output

    def test_get_json(self, runner):
        import json as json_mod
        r = runner.invoke(cli, [
            "remember", "FastAPI uses Starlette under the hood for performance", "--json",
        ])
        mem_id = json_mod.loads(r.output)["id"]

        result = runner.invoke(cli, ["get", mem_id, "--json"])
        assert result.exit_code == 0
        data = json_mod.loads(result.output)
        assert data["id"] == mem_id

    def test_get_not_found(self, runner):
        result = runner.invoke(cli, ["get", "nonexistent"])
        assert result.exit_code == 1


class TestDeleteMemory:
    def test_delete_with_force(self, runner):
        import json as json_mod
        r = runner.invoke(cli, [
            "remember", "A temporary memory about testing deletion flows", "--json",
        ])
        mem_id = json_mod.loads(r.output)["id"]

        result = runner.invoke(cli, ["delete", mem_id, "--force"])
        assert result.exit_code == 0
        assert "Deleted" in result.output

        # Verify gone
        result = runner.invoke(cli, ["get", mem_id])
        assert result.exit_code == 1

    def test_delete_not_found(self, runner):
        result = runner.invoke(cli, ["delete", "nonexistent", "--force"])
        assert result.exit_code == 1


class TestUpdateMemory:
    def test_update_salience(self, runner):
        import json as json_mod
        r = runner.invoke(cli, [
            "remember", "An important memory about system architecture patterns", "--json",
        ])
        mem_id = json_mod.loads(r.output)["id"]

        result = runner.invoke(cli, ["update", mem_id, "--salience", "0.95"])
        assert result.exit_code == 0
        assert "Updated" in result.output

    def test_update_tags(self, runner):
        import json as json_mod
        r = runner.invoke(cli, [
            "remember", "Memory about Python web frameworks and their ecosystems", "--json",
        ])
        mem_id = json_mod.loads(r.output)["id"]

        result = runner.invoke(cli, ["update", mem_id, "--tags", "python", "--tags", "web"])
        assert result.exit_code == 0
        assert "python" in result.output

    def test_update_not_found(self, runner):
        result = runner.invoke(cli, ["update", "nonexistent", "--salience", "0.5"])
        assert result.exit_code == 1


class TestEpisodeQueries:
    def test_episode_list_empty(self, runner):
        result = runner.invoke(cli, ["episode", "list"])
        assert result.exit_code == 0
        assert "No episodes" in result.output

    def test_episode_list_with_data(self, runner):
        r = runner.invoke(cli, ["episode", "start", "--title", "Test Episode"])
        # Extract ep_id from first line: "Episode started: ep_xxx"
        ep_id = r.output.strip().split("\n")[0].split(": ")[1]
        runner.invoke(cli, ["episode", "end", ep_id])

        result = runner.invoke(cli, ["episode", "list"])
        assert result.exit_code == 0
        assert "Test Episode" in result.output or ep_id in result.output

    def test_episode_get(self, runner):
        r = runner.invoke(cli, ["episode", "start", "--title", "Detail Ep"])
        ep_id = r.output.strip().split("\n")[0].split(": ")[1]

        result = runner.invoke(cli, ["episode", "get", ep_id])
        assert result.exit_code == 0
        assert ep_id in result.output

    def test_episode_get_not_found(self, runner):
        result = runner.invoke(cli, ["episode", "get", "nonexistent"])
        assert result.exit_code == 1


class TestIntentions:
    def test_intention_add(self, runner):
        result = runner.invoke(cli, [
            "intention", "add", "Remember to update docs after refactoring",
        ])
        assert result.exit_code == 0
        assert "Intention stored" in result.output

    def test_intention_list_empty(self, runner):
        result = runner.invoke(cli, ["intention", "list"])
        assert result.exit_code == 0
        assert "No pending" in result.output

    def test_intention_list_with_data(self, runner):
        runner.invoke(cli, ["intention", "add", "TODO: write more tests for memory system"])
        result = runner.invoke(cli, ["intention", "list"])
        assert result.exit_code == 0
        assert "write more tests" in result.output

    def test_intention_resolve(self, runner):
        import json as json_mod
        r = runner.invoke(cli, [
            "intention", "add", "Intention to resolve after task completion", "--json",
        ])
        mem_id = json_mod.loads(r.output)["id"]

        result = runner.invoke(cli, ["intention", "resolve", mem_id])
        assert result.exit_code == 0
        assert "Resolved" in result.output

    def test_intention_resolve_not_found(self, runner):
        result = runner.invoke(cli, ["intention", "resolve", "nonexistent"])
        assert result.exit_code == 1


class TestGraphCommands:
    def test_path_no_path(self, runner):
        result = runner.invoke(cli, ["graph", "path", "fake1", "fake2"])
        assert result.exit_code == 0
        assert "No path" in result.output

    def test_common_empty(self, runner):
        result = runner.invoke(cli, ["graph", "common", "fake1", "fake2"])
        assert result.exit_code == 0
        assert "No common" in result.output or "0" in result.output


class TestSchemaCommands:
    def test_schema_list_empty(self, runner):
        result = runner.invoke(cli, ["schema", "list"])
        assert result.exit_code == 0
        assert "No schemas" in result.output

    def test_schema_create(self, runner):
        import json as json_mod
        r1 = runner.invoke(cli, ["remember", "Python uses indentation for code blocks", "--json"])
        id1 = json_mod.loads(r1.output)["id"]

        result = runner.invoke(cli, [
            "schema", "create", "Languages use different block delimiters",
            "--source", id1,
        ])
        assert result.exit_code == 0
        assert "Schema created" in result.output or "schema" in result.output.lower()


class TestProcedureCommands:
    def test_procedure_list_empty(self, runner):
        result = runner.invoke(cli, ["procedure", "list"])
        assert result.exit_code == 0
        assert "No procedures" in result.output

    def test_procedure_add(self, runner):
        result = runner.invoke(cli, [
            "procedure", "add", "Step 1: Read logs. Step 2: Reproduce. Step 3: Fix.",
            "--tags", "debugging",
        ])
        assert result.exit_code == 0
        assert "Procedure stored" in result.output or "procedure" in result.output.lower()


class TestEmotionsCommand:
    def test_emotions(self, runner):
        result = runner.invoke(cli, ["emotions"])
        assert result.exit_code == 0
