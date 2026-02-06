"""Tests for the CLI import commands."""

import json
import tempfile
from pathlib import Path

import pytest
from click.testing import CliRunner

from cerebro.cortex import CerebroCortex
from cerebro.interfaces import cli as cli_mod
from cerebro.interfaces.cli import cli


@pytest.fixture
def cortex():
    with tempfile.TemporaryDirectory() as d:
        ctx = CerebroCortex(db_path=Path(d) / "test_cli_import.db", chroma_dir=Path(d) / "chroma")
        ctx.initialize()
        cli_mod._cortex = ctx
        yield ctx
        ctx.close()
        cli_mod._cortex = None


@pytest.fixture
def runner():
    return CliRunner()


def _extract_json(output: str) -> dict:
    """Extract the JSON object from CLI output (skip non-JSON lines)."""
    # Find the first line starting with { and parse from there
    lines = output.strip().split("\n")
    json_start = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("{"):
            json_start = i
            break
    if json_start is not None:
        json_text = "\n".join(lines[json_start:])
        return json.loads(json_text)
    raise ValueError(f"No JSON found in output: {output}")


class TestImportNeoCortex:
    def test_import_neocortex(self, cortex, runner, tmp_path):
        data = {
            "format_version": "1.0",
            "collections": {
                "cortex_shared": [
                    {"id": "n1", "content": "Imported Neo-Cortex memory about system design", "message_type": "fact"},
                ],
            },
        }
        path = tmp_path / "export.json"
        path.write_text(json.dumps(data))

        result = runner.invoke(cli, ["import", "neocortex", str(path)])
        assert result.exit_code == 0
        assert "Imported:  1" in result.output

    def test_import_neocortex_json_output(self, cortex, runner, tmp_path):
        data = {
            "format_version": "1.0",
            "collections": {
                "cortex_shared": [
                    {"id": "n1", "content": "Memory for JSON output testing purposes", "message_type": "observation"},
                ],
            },
        }
        path = tmp_path / "export.json"
        path.write_text(json.dumps(data))

        result = runner.invoke(cli, ["import", "neocortex", str(path), "--json"])
        assert result.exit_code == 0
        out = _extract_json(result.output)
        assert out["memories_imported"] == 1


class TestImportJSON:
    def test_import_json(self, cortex, runner, tmp_path):
        data = [
            {"content": "Generic JSON memory about Python programming language"},
        ]
        path = tmp_path / "memories.json"
        path.write_text(json.dumps(data))

        result = runner.invoke(cli, ["import", "json", str(path)])
        assert result.exit_code == 0
        assert "Imported:  1" in result.output

    def test_import_json_output(self, cortex, runner, tmp_path):
        data = [{"content": "Another JSON memory for testing the import flow"}]
        path = tmp_path / "memories.json"
        path.write_text(json.dumps(data))

        result = runner.invoke(cli, ["import", "json", str(path), "--json"])
        assert result.exit_code == 0
        out = _extract_json(result.output)
        assert out["memories_imported"] == 1


class TestImportMarkdown:
    def test_import_markdown(self, cortex, runner, tmp_path):
        path = tmp_path / "notes.md"
        path.write_text("## Topic\nThis is a memory about software architecture.\n")

        result = runner.invoke(cli, ["import", "markdown", str(path)])
        assert result.exit_code == 0
        assert "Imported:  1" in result.output

    def test_import_markdown_json_output(self, cortex, runner, tmp_path):
        path = tmp_path / "notes.md"
        path.write_text("## Topic\nMarkdown memory for JSON output verification.\n")

        result = runner.invoke(cli, ["import", "markdown", str(path), "--json"])
        assert result.exit_code == 0
        out = _extract_json(result.output)
        assert out["memories_imported"] == 1
