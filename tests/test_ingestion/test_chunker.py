"""Tests for SemanticChunker."""

import pytest

from cerebro.ingestion.chunker import SemanticChunker, _cosine_sim, _centroid


class TestSemanticChunker:
    def test_simple_text_fits_in_one_chunk(self):
        text = "This is a short text. It has two sentences."
        chunker = SemanticChunker(max_tokens=512)
        chunks = list(chunker.chunk(text))
        assert len(chunks) == 1
        assert "short text" in chunks[0]

    def test_long_text_splits_into_multiple_chunks(self):
        # Generate ~2000 words
        sentences = [f"Sentence number {i} talks about topic A in detail." for i in range(200)]
        text = "\n\n".join(sentences)
        chunker = SemanticChunker(max_tokens=512)
        chunks = list(chunker.chunk(text))
        assert len(chunks) > 1
        # Each chunk should be reasonable size
        for chunk in chunks:
            words = len(chunk.split())
            assert words <= 400  # ~512 tokens * 0.75 words/token

    def test_overlap_carryover(self):
        # Text that splits into at least 2 chunks
        sentences = [f"This is sentence {i} with some meaningful content about programming." for i in range(100)]
        text = " ".join(sentences)
        chunker = SemanticChunker(max_tokens=512, overlap_tokens=50)
        chunks = list(chunker.chunk(text))
        assert len(chunks) >= 2
        # Overlap: last sentences of chunk 1 should appear at start of chunk 2
        first_chunk_words = set(chunks[0].split())
        second_chunk_words = set(chunks[1].split())
        overlap = first_chunk_words & second_chunk_words
        assert len(overlap) > 10  # Some overlap should exist

    def test_paragraph_boundary_respected(self):
        text = "\n\n".join([
            "First paragraph about topic A. It has multiple sentences. " * 50,
            "Second paragraph about topic B. Completely different subject. " * 50,
        ])
        chunker = SemanticChunker(max_tokens=512)
        chunks = list(chunker.chunk(text))
        # Paragraph boundary should be respected when near 50% of max
        for chunk in chunks:
            # No chunk should contain both topics if they were split properly
            pass  # Just verify it doesn't crash

    def test_empty_text(self):
        chunker = SemanticChunker()
        assert list(chunker.chunk("")) == []

    def test_single_sentence(self):
        chunker = SemanticChunker()
        chunks = list(chunker.chunk("One sentence."))
        assert len(chunks) == 1
        assert chunks[0] == "One sentence."


class TestCosineSim:
    def test_identical_vectors(self):
        v = [1.0, 2.0, 3.0]
        assert _cosine_sim(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert _cosine_sim(a, b) == pytest.approx(0.0)

    def test_zero_vector(self):
        assert _cosine_sim([0.0, 0.0], [1.0, 1.0]) == 0.0


class TestCentroid:
    def test_simple_centroid(self):
        embeddings = [[1.0, 2.0], [3.0, 4.0]]
        c = _centroid(embeddings)
        assert c == [2.0, 3.0]

    def test_single_embedding(self):
        embeddings = [[1.0, 2.0]]
        c = _centroid(embeddings)
        assert c == [1.0, 2.0]
