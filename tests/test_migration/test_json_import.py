"""Tests for the generic JSON importer."""

import json
import tempfile
from pathlib import Path

import pytest

from cerebro.cortex import CerebroCortex
from cerebro.migration.json_import import JSONImporter, JSONImportReport
from cerebro.types import MemoryType


@pytest.fixture
def cortex():
    with tempfile.TemporaryDirectory() as d:
        ctx = CerebroCortex(db_path=Path(d) / "test_json_import.db")
        ctx.initialize()
        yield ctx
        ctx.close()


@pytest.fixture
def importer(cortex):
    return JSONImporter(cortex)


class TestJSONImportReport:
    def test_empty_report(self):
        r = JSONImportReport()
        d = r.to_dict()
        assert d["memories_imported"] == 0
        assert d["total_errors"] == 0


class TestSimpleFormat:
    def test_basic_list(self, importer, cortex):
        data = [
            {"content": "Python uses indentation for code blocks instead of braces"},
            {"content": "JavaScript has prototype-based inheritance model"},
        ]
        report = importer.import_data(data)
        assert report.memories_imported == 2

    def test_with_type(self, importer, cortex):
        data = [{"content": "Always check error logs before debugging further", "type": "procedural"}]
        report = importer.import_data(data)
        assert report.memories_imported == 1

        ids = cortex.graph.get_all_node_ids()
        node = cortex.graph.get_node(ids[0])
        assert node.metadata.memory_type == MemoryType.PROCEDURAL

    def test_with_tags(self, importer, cortex):
        data = [{"content": "FastAPI supports async request handlers natively", "tags": ["python", "web"]}]
        report = importer.import_data(data)
        ids = cortex.graph.get_all_node_ids()
        node = cortex.graph.get_node(ids[0])
        assert "python" in node.metadata.tags
        assert "web" in node.metadata.tags

    def test_with_salience(self, importer, cortex):
        data = [{"content": "Critical: never expose API keys in client code", "salience": 0.95}]
        report = importer.import_data(data)
        ids = cortex.graph.get_all_node_ids()
        node = cortex.graph.get_node(ids[0])
        assert node.metadata.salience == 0.95

    def test_skip_short(self, importer):
        data = [{"content": "hi"}, {"content": "ok"}]
        report = importer.import_data(data)
        assert report.memories_imported == 0
        assert report.memories_skipped == 2

    def test_skip_duplicates(self, importer):
        content = "A unique observation about machine learning model training"
        data = [{"content": content}, {"content": content}]
        report = importer.import_data(data)
        assert report.memories_imported == 1
        assert report.memories_skipped == 1

    def test_comma_separated_tags(self, importer, cortex):
        data = [{"content": "Memory with comma-separated tags for categorization", "tags": "python, web, api"}]
        report = importer.import_data(data)
        ids = cortex.graph.get_all_node_ids()
        node = cortex.graph.get_node(ids[0])
        assert "python" in node.metadata.tags
        assert "web" in node.metadata.tags


class TestFullFormat:
    def test_nested_metadata(self, importer, cortex):
        data = [{
            "content": "FastAPI is built on top of Starlette and Pydantic",
            "metadata": {
                "memory_type": "semantic",
                "tags": ["fastapi"],
                "agent_id": "TESTER",
                "salience": 0.7,
            },
        }]
        report = importer.import_data(data)
        assert report.memories_imported == 1

        ids = cortex.graph.get_all_node_ids()
        node = cortex.graph.get_node(ids[0])
        assert node.metadata.memory_type == MemoryType.SEMANTIC
        assert node.metadata.agent_id == "TESTER"
        assert node.metadata.salience == 0.7

    def test_nested_strength(self, importer, cortex):
        data = [{
            "content": "igraph provides C-speed graph traversal algorithms",
            "strength": {"stability": 5.0, "difficulty": 3.0, "access_count": 10},
        }]
        report = importer.import_data(data)
        ids = cortex.graph.get_all_node_ids()
        node = cortex.graph.get_node(ids[0])
        assert node.strength.stability == 5.0
        assert node.strength.difficulty == 3.0


class TestDictWrapper:
    def test_memories_key(self, importer):
        data = {"memories": [
            {"content": "Dictionary wrapper with memories key for organization"},
        ]}
        report = importer.import_data(data)
        assert report.memories_imported == 1

    def test_records_key(self, importer):
        data = {"records": [
            {"content": "Dictionary wrapper with records key as alternative"},
        ]}
        report = importer.import_data(data)
        assert report.memories_imported == 1


class TestFileImport:
    def test_from_file(self, importer, tmp_path):
        data = [{"content": "Memory loaded from a JSON file on the filesystem"}]
        path = tmp_path / "memories.json"
        path.write_text(json.dumps(data))
        report = importer.import_file(path)
        assert report.memories_imported == 1
        assert report.duration_seconds > 0

    def test_source_marked_import(self, importer, cortex):
        data = [{"content": "Source field should be marked as import origin"}]
        importer.import_data(data)
        ids = cortex.graph.get_all_node_ids()
        node = cortex.graph.get_node(ids[0])
        assert node.metadata.source == "import"


class TestErrorHandling:
    def test_non_dict_record(self, importer):
        data = ["string_record", 42]
        report = importer.import_data(data)
        assert report.memories_skipped == 2
        assert len(report.errors) == 2

    def test_bad_root_type(self, importer):
        report = importer.import_data("not a list or dict")
        assert len(report.errors) == 1
