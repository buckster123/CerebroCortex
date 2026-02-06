"""Tests for the Markdown importer."""

import tempfile
from pathlib import Path

import pytest

from cerebro.cortex import CerebroCortex
from cerebro.migration.markdown_import import MarkdownImporter, MarkdownImportReport
from cerebro.types import MemoryType


@pytest.fixture
def cortex():
    with tempfile.TemporaryDirectory() as d:
        ctx = CerebroCortex(db_path=Path(d) / "test_md_import.db", chroma_dir=Path(d) / "chroma")
        ctx.initialize()
        yield ctx
        ctx.close()


@pytest.fixture
def importer(cortex):
    return MarkdownImporter(cortex)


class TestMarkdownImportReport:
    def test_empty_report(self):
        r = MarkdownImportReport()
        d = r.to_dict()
        assert d["memories_imported"] == 0
        assert d["total_errors"] == 0


class TestSectionParsing:
    def test_heading_sections(self, importer, cortex):
        text = """## Python Basics
Python is a high-level interpreted programming language.

## JavaScript Basics
JavaScript is the language of the web browser.
"""
        report = importer.import_text(text)
        assert report.memories_imported == 2

        ids = cortex.graph.get_all_node_ids()
        assert len(ids) == 2

        # Content includes the title
        nodes = [cortex.graph.get_node(mid) for mid in ids]
        contents = [n.content for n in nodes]
        assert any("Python Basics" in c for c in contents)
        assert any("JavaScript Basics" in c for c in contents)

    def test_paragraph_fallback(self, importer, cortex):
        text = """First paragraph about machine learning algorithms.

Second paragraph about deep neural network architectures.

Third paragraph about reinforcement learning strategies.
"""
        report = importer.import_text(text)
        assert report.memories_imported == 3

    def test_skip_short_sections(self, importer):
        text = """## Title
ok

## Good Section
This has enough content to be imported as a memory.
"""
        report = importer.import_text(text)
        assert report.memories_imported == 1
        assert report.memories_skipped == 1

    def test_deduplication(self, importer):
        text = """## Section A
This is the same content repeated twice.

## Section B
This is the same content repeated twice.
"""
        report = importer.import_text(text)
        # Titles differ, so content is "Section A: ..." vs "Section B: ..."
        assert report.memories_imported == 2

    def test_tags_from_title(self, importer, cortex):
        text = """## Error Handling
Always catch specific exceptions instead of bare except."""
        report = importer.import_text(text)
        ids = cortex.graph.get_all_node_ids()
        node = cortex.graph.get_node(ids[0])
        assert "error_handling" in node.metadata.tags


class TestFrontmatter:
    def test_basic_frontmatter(self, importer, cortex):
        text = """---
type: procedural
tags: [debugging, workflow]
agent_id: TESTER
---

## Debug Steps
First check the error logs then reproduce the issue.
"""
        report = importer.import_text(text)
        assert report.memories_imported == 1

        ids = cortex.graph.get_all_node_ids()
        node = cortex.graph.get_node(ids[0])
        assert node.metadata.memory_type == MemoryType.PROCEDURAL
        assert "debugging" in node.metadata.tags
        assert "workflow" in node.metadata.tags

    def test_no_frontmatter(self, importer):
        text = """## Simple Section
Just a section with no frontmatter at all."""
        report = importer.import_text(text)
        assert report.memories_imported == 1

    def test_frontmatter_defaults(self, importer, cortex):
        text = """---
type: semantic
---

First standalone paragraph about graph databases.

Second standalone paragraph about vector embeddings.
"""
        report = importer.import_text(text)
        assert report.memories_imported == 2

        ids = cortex.graph.get_all_node_ids()
        for mid in ids:
            node = cortex.graph.get_node(mid)
            assert node.metadata.memory_type == MemoryType.SEMANTIC


class TestFileImport:
    def test_from_file(self, importer, tmp_path):
        path = tmp_path / "notes.md"
        path.write_text("## Test Note\nImported from a markdown file on disk.")
        report = importer.import_file(path)
        assert report.memories_imported == 1
        assert report.duration_seconds > 0


class TestSourceTracking:
    def test_source_is_import(self, importer, cortex):
        text = """## Source Check
This memory should be marked with source=import."""
        importer.import_text(text)
        ids = cortex.graph.get_all_node_ids()
        node = cortex.graph.get_node(ids[0])
        assert node.metadata.source == "import"


class TestEdgeCases:
    def test_empty_text(self, importer):
        report = importer.import_text("")
        assert report.memories_imported == 0

    def test_only_headings(self, importer):
        text = """## A
## B
## C"""
        report = importer.import_text(text)
        assert report.memories_imported == 0  # No content under headings

    def test_mixed_heading_levels(self, importer):
        text = """## Main Topic
Some content about the main topic here.

### Sub Topic
Sub-content is part of the main section.
"""
        report = importer.import_text(text)
        # ### is NOT treated as section separator (only ##)
        assert report.memories_imported == 1
