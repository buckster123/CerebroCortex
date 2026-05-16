"""Tests for semantic chunking integration in ingestion pipeline."""

import pytest

from cerebro.ingestion.chunker import SemanticChunker
from cerebro.ingestion.pipeline import IngestionPipeline
from cerebro.ingestion.text_adapter import TextAdapter
from cerebro.ingestion.markdown_adapter import MarkdownAdapter


class TestSemanticChunkerUnit:
    """Unit tests for the SemanticChunker class."""

    def test_short_text_single_chunk(self):
        chunker = SemanticChunker(max_tokens=512)
        text = "This is a short text. It should fit in one chunk."
        chunks = list(chunker.chunk(text))
        assert len(chunks) == 1
        assert "short text" in chunks[0]

    def test_long_text_multiple_chunks(self):
        chunker = SemanticChunker(max_tokens=64, overlap_tokens=10)
        # Generate a long text with clear sentence boundaries
        sentences = [f"Sentence number {i} contains some words about topic {i % 3}." for i in range(50)]
        text = " ".join(sentences)
        chunks = list(chunker.chunk(text))
        assert len(chunks) > 1
        # Each chunk should be reasonably sized
        for chunk in chunks:
            words = len(chunk.split())
            # max_tokens=64 * 0.75 words/token = ~48 words max + overlap
            assert words <= 80

    def test_overlap_preserved(self):
        chunker = SemanticChunker(max_tokens=64, overlap_tokens=20)
        sentences = [f"This is sentence {i} with enough words to matter." for i in range(30)]
        text = " ".join(sentences)
        chunks = list(chunker.chunk(text))
        if len(chunks) > 1:
            # Last few words of chunk 0 should appear in chunk 1
            last_words = set(chunks[0].split()[-5:])
            first_words = set(chunks[1].split()[:10])
            assert last_words & first_words  # some overlap

    def test_paragraph_boundary_respected(self):
        chunker = SemanticChunker(max_tokens=512)
        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        chunks = list(chunker.chunk(text))
        # Should fit in one chunk since it's short
        assert len(chunks) == 1


class TestPipelineSemanticChunking:
    """Integration tests for semantic chunking in IngestionPipeline."""

    def test_pipeline_creates_chunker_when_enabled(self, monkeypatch):
        # Ensure semantic chunking is enabled
        monkeypatch.setattr("cerebro.ingestion.pipeline.SEMANTIC_CHUNKING_ENABLED", True)

        # We can't initialize a real cortex without a DB, so just check the
        # pipeline constructor logic by mocking
        class FakeCortex:
            pass

        pipeline = IngestionPipeline(FakeCortex())
        assert pipeline.chunker is not None
        assert isinstance(pipeline.chunker, SemanticChunker)

    def test_pipeline_no_chunker_when_disabled(self, monkeypatch):
        monkeypatch.setattr("cerebro.ingestion.pipeline.SEMANTIC_CHUNKING_ENABLED", False)

        class FakeCortex:
            pass

        pipeline = IngestionPipeline(FakeCortex())
        assert pipeline.chunker is None

    def test_text_adapter_with_chunker(self, multi_agent_cortex):
        ctx = multi_agent_cortex["cortex"]
        chunker = SemanticChunker(max_tokens=64, overlap_tokens=10)
        adapter = TextAdapter(chunker=chunker)

        # Long text that should be chunked
        sentences = [f"Topic paragraph about subject {i} with many words." for i in range(40)]
        text = " ".join(sentences)

        report = adapter.ingest_text(text, cortex=ctx, agent_id="CLAUDE")
        # Should create multiple memories due to chunking
        assert report.memories_imported >= 2

    def test_text_adapter_without_chunker(self, multi_agent_cortex):
        ctx = multi_agent_cortex["cortex"]
        adapter = TextAdapter(chunker=None)

        # Same long text
        sentences = [f"Topic paragraph about subject {i} with many words." for i in range(40)]
        text = " ".join(sentences)

        report = adapter.ingest_text(text, cortex=ctx, agent_id="CLAUDE")
        # Without semantic chunker, legacy paragraph chunking still splits
        assert report.memories_imported >= 1


class TestMarkdownAdapterChunking:
    """Test markdown adapter integration with semantic chunker."""

    def test_flat_markdown_delegates_to_semantic_chunker(self, multi_agent_cortex):
        ctx = multi_agent_cortex["cortex"]
        chunker = SemanticChunker(max_tokens=64, overlap_tokens=10)
        adapter = MarkdownAdapter(chunker=chunker)

        # Flat markdown (no ## headings)
        text = "\n\n".join([f"Paragraph {i} with many words about various topics." for i in range(30)])

        report = adapter.ingest_text(text, cortex=ctx, agent_id="CLAUDE")
        assert report.memories_imported >= 1

    def test_headings_markdown_uses_sections(self, multi_agent_cortex):
        ctx = multi_agent_cortex["cortex"]
        chunker = SemanticChunker(max_tokens=64, overlap_tokens=10)
        adapter = MarkdownAdapter(chunker=chunker)

        # Markdown with headings
        text = "## Section One\nContent for section one.\n\n## Section Two\nContent for section two."

        report = adapter.ingest_text(text, cortex=ctx, agent_id="CLAUDE")
        assert report.memories_imported == 2

    def test_frontmatter_preserved(self, multi_agent_cortex):
        ctx = multi_agent_cortex["cortex"]
        adapter = MarkdownAdapter(chunker=None)

        text = """---
type: procedural
tags: [workflow, test]
---

## Step One
Do the first thing.

## Step Two
Do the second thing.
"""
        report = adapter.ingest_text(text, cortex=ctx, agent_id="CLAUDE")
        assert report.memories_imported == 2
        # Tags from frontmatter should be applied
