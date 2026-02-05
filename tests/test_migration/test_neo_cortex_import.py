"""Tests for the Neo-Cortex importer."""

import json
import tempfile
from pathlib import Path

import pytest

from cerebro.cortex import CerebroCortex
from cerebro.migration.neo_cortex_import import (
    ImportReport,
    MESSAGE_TYPE_MAP,
    NeoCortexImporter,
)
from cerebro.types import LinkType, MemoryType


@pytest.fixture
def cortex():
    with tempfile.TemporaryDirectory() as d:
        ctx = CerebroCortex(db_path=Path(d) / "test_import.db")
        ctx.initialize()
        yield ctx
        ctx.close()


@pytest.fixture
def importer(cortex):
    return NeoCortexImporter(cortex)


def _make_export(records, collection="cortex_shared"):
    """Build a minimal Neo-Cortex export dict."""
    return {
        "format_version": "1.0",
        "agent_id": "CLAUDE",
        "exported_at": "2026-02-05T12:00:00",
        "collections": {collection: records},
        "metadata": {"total_memories": len(records)},
    }


class TestTypeMapping:
    def test_all_types_mapped(self):
        expected = {
            "fact", "observation", "discovery", "cultural",
            "dialogue", "question", "clarification",
            "task", "protocol", "reminder", "session_note",
        }
        assert set(MESSAGE_TYPE_MAP.keys()) == expected

    def test_fact_maps_to_semantic(self):
        assert MESSAGE_TYPE_MAP["fact"] == MemoryType.SEMANTIC

    def test_dialogue_maps_to_episodic(self):
        assert MESSAGE_TYPE_MAP["dialogue"] == MemoryType.EPISODIC

    def test_task_maps_to_prospective(self):
        assert MESSAGE_TYPE_MAP["task"] == MemoryType.PROSPECTIVE

    def test_protocol_maps_to_procedural(self):
        assert MESSAGE_TYPE_MAP["protocol"] == MemoryType.PROCEDURAL


class TestImportReport:
    def test_empty_report(self):
        report = ImportReport()
        d = report.to_dict()
        assert d["memories_imported"] == 0
        assert d["total_errors"] == 0

    def test_report_caps_errors(self):
        report = ImportReport(errors=[f"err{i}" for i in range(50)])
        d = report.to_dict()
        assert len(d["errors"]) == 20
        assert d["total_errors"] == 50


class TestBasicImport:
    def test_empty_export(self, importer):
        report = importer.import_data({"collections": {}})
        assert report.memories_imported == 0

    def test_single_memory(self, importer, cortex):
        data = _make_export([{
            "id": "neo_1",
            "content": "Python is a dynamically typed programming language",
            "message_type": "fact",
            "agent_id": "CLAUDE",
            "visibility": "shared",
            "layer": "working",
            "tags": ["python"],
            "access_count": 3,
            "attention_weight": 0.8,
            "created_at": "2026-01-15T10:00:00",
            "responding_to": [],
        }])

        report = importer.import_data(data)
        assert report.memories_imported == 1
        assert report.memories_skipped == 0

        # Verify node in graph
        new_id = report.id_mapping["neo_1"]
        node = cortex.graph.get_node(new_id)
        assert node is not None
        assert "Python" in node.content
        assert node.metadata.memory_type == MemoryType.SEMANTIC
        assert node.metadata.salience == 0.8
        assert "neo:fact" in node.metadata.tags
        assert "python" in node.metadata.tags
        assert node.metadata.source == "import"

    def test_multiple_memories(self, importer):
        records = [
            {"id": f"neo_{i}", "content": f"Memory number {i} with enough content to pass gating", "message_type": "observation"}
            for i in range(5)
        ]
        data = _make_export(records)
        report = importer.import_data(data)
        assert report.memories_imported == 5

    def test_skip_short_content(self, importer):
        data = _make_export([{"id": "neo_1", "content": "hi", "message_type": "fact"}])
        report = importer.import_data(data)
        assert report.memories_imported == 0
        assert report.memories_skipped == 1

    def test_skip_empty_content(self, importer):
        data = _make_export([{"id": "neo_1", "content": "", "message_type": "fact"}])
        report = importer.import_data(data)
        assert report.memories_imported == 0
        assert report.memories_skipped == 1

    def test_deduplication(self, importer):
        content = "Unique memory about quantum computing and particle physics"
        records = [
            {"id": "neo_1", "content": content, "message_type": "fact"},
            {"id": "neo_2", "content": content, "message_type": "fact"},
        ]
        data = _make_export(records)
        report = importer.import_data(data)
        assert report.memories_imported == 1
        assert report.memories_skipped == 1


class TestMultiCollection:
    def test_imports_all_collections(self, importer):
        data = {
            "format_version": "1.0",
            "collections": {
                "cortex_shared": [
                    {"id": "s1", "content": "Shared memory about distributed systems architecture", "message_type": "fact"},
                ],
                "cortex_private": [
                    {"id": "p1", "content": "Private memory about personal debugging workflow preferences", "message_type": "observation"},
                ],
            },
        }
        report = importer.import_data(data)
        assert report.memories_imported == 2


class TestRespondingToLinks:
    def test_creates_contextual_links(self, importer, cortex):
        records = [
            {"id": "neo_a", "content": "First observation about the system's performance bottleneck", "message_type": "observation", "responding_to": []},
            {"id": "neo_b", "content": "Follow-up analysis of the performance issues identified earlier", "message_type": "observation", "responding_to": ["neo_a"]},
        ]
        data = _make_export(records)
        report = importer.import_data(data)

        assert report.memories_imported == 2
        assert report.links_created == 1

        # Verify the link exists
        id_a = report.id_mapping["neo_a"]
        id_b = report.id_mapping["neo_b"]
        link = cortex.graph.get_link(id_b, id_a, LinkType.CONTEXTUAL)
        assert link is not None
        assert link.source == "migration"
        assert link.weight == 0.6

    def test_skip_missing_target(self, importer):
        """Links to non-imported IDs are silently skipped."""
        records = [
            {"id": "neo_a", "content": "Memory that responds to a non-existent memory ID", "message_type": "fact", "responding_to": ["nonexistent_id"]},
        ]
        data = _make_export(records)
        report = importer.import_data(data)
        assert report.memories_imported == 1
        assert report.links_created == 0

    def test_responding_to_as_json_string(self, importer):
        """responding_to stored as JSON string in ChromaDB metadata."""
        records = [
            {"id": "neo_a", "content": "First memory about graph algorithms and traversal", "message_type": "fact", "responding_to": "[]"},
            {"id": "neo_b", "content": "Second memory referencing the first graph memory above", "message_type": "fact", "responding_to": json.dumps(["neo_a"])},
        ]
        data = _make_export(records)
        report = importer.import_data(data)
        assert report.links_created == 1


class TestAgentImport:
    def test_agent_profile_registered(self, importer, cortex):
        records = [
            {
                "id": "agent_1",
                "content": "Agent Profile: Researcher\n\nAgent ID: RESEARCHER\nDisplay Name: Researcher\nSpecialization: Deep research\nGeneration: 1\nLineage: Science",
                "message_type": "agent_profile",
                "agent_id": "RESEARCHER",
            },
        ]
        data = _make_export(records)
        report = importer.import_data(data)

        assert report.agents_registered == 1
        assert report.memories_imported == 0  # Agents don't become memories

        agents = cortex.graph.list_agents()
        assert any(a.id == "RESEARCHER" for a in agents)


class TestStrengthSeeding:
    def test_seed_from_access_count(self, importer):
        records = [{
            "id": "neo_1",
            "content": "A frequently accessed memory about database optimization",
            "message_type": "fact",
            "access_count": 25,
            "created_at": "2026-01-01T00:00:00",
        }]
        data = _make_export(records)
        report = importer.import_data(data)

        new_id = report.id_mapping["neo_1"]
        node = importer.cortex.graph.get_node(new_id)
        assert node is not None
        assert node.strength.access_count == 25
        assert len(node.strength.access_timestamps) == 25
        assert node.strength.stability > 0

    def test_seed_high_access_count_compresses(self, importer):
        records = [{
            "id": "neo_1",
            "content": "A very frequently accessed memory about API design",
            "message_type": "fact",
            "access_count": 100,
            "created_at": "2025-12-01T00:00:00",
        }]
        data = _make_export(records)
        report = importer.import_data(data)

        new_id = report.id_mapping["neo_1"]
        node = importer.cortex.graph.get_node(new_id)
        assert node.strength.access_count == 100
        assert len(node.strength.access_timestamps) == 50  # compressed
        assert node.strength.compressed_count == 50


class TestFileImport:
    def test_import_from_file(self, importer, tmp_path):
        data = _make_export([{
            "id": "neo_1",
            "content": "Memory imported from a file on the filesystem",
            "message_type": "fact",
        }])
        file_path = tmp_path / "export.json"
        file_path.write_text(json.dumps(data))

        report = importer.import_file(file_path)
        assert report.memories_imported == 1
        assert report.duration_seconds > 0
