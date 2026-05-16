"""Semantic chunking for document ingestion.

Replaces naive paragraph splitting with sentence-aware chunking that respects
topic boundaries. Uses embedding similarity to detect when a sentence starts
a new topic, falling back to paragraph boundaries when embeddings are
unavailable.
"""

import logging
import re
from typing import Iterator, Optional

logger = logging.getLogger(__name__)

# Regex for sentence splitting: handles . ! ? followed by space or end
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")

# Approximate words per token ratio
_WORDS_PER_TOKEN = 0.75


class SemanticChunker:
    """Split text into semantically coherent chunks.

    Strategy:
    1. Split into sentences.
    2. Greedily add sentences to current chunk until near max_tokens.
    3. Look ahead: if next sentence starts a new topic (detected by
       embedding similarity drop or paragraph boundary), break early.
    4. Yield chunk; carry over last N sentences as overlap.
    """

    def __init__(
        self,
        max_tokens: int = 512,
        overlap_tokens: int = 50,
        topic_similarity_threshold: float = 0.6,
        embedding_func=None,
    ):
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.topic_threshold = topic_similarity_threshold
        self.embedding_func = embedding_func
        self.max_words = int(max_tokens * _WORDS_PER_TOKEN)
        self.overlap_words = int(overlap_tokens * _WORDS_PER_TOKEN)

    def chunk(self, text: str) -> Iterator[str]:
        """Yield semantically coherent chunks from *text*."""
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if not paragraphs:
            return

        sentences: list[str] = []
        sent_paragraph_idx: list[int] = []  # which paragraph each sentence came from

        for p_idx, para in enumerate(paragraphs):
            para_sents = _SENTENCE_RE.split(para)
            for s in para_sents:
                s = s.strip()
                if s:
                    sentences.append(s)
                    sent_paragraph_idx.append(p_idx)

        if not sentences:
            return

        # If the whole text fits in one chunk, yield it
        total_words = sum(len(s.split()) for s in sentences)
        if total_words <= self.max_words:
            yield " ".join(sentences)
            return

        # Embeddings for topic detection (lazy — only compute if needed)
        embeddings: Optional[list] = None

        current_sents: list[str] = []
        current_words = 0

        for i, sentence in enumerate(sentences):
            s_words = len(sentence.split())

            # Check if adding this sentence would exceed the word budget
            would_exceed = current_words + s_words > self.max_words

            # Check for topic boundary (if we have sentences and embeddings available)
            topic_break = False
            if current_sents and self.embedding_func is not None and not would_exceed:
                # Lazy compute embeddings on first need
                if embeddings is None:
                    try:
                        embeddings = self.embedding_func(sentences)
                    except Exception as exc:
                        logger.debug(f"Embedding-based topic detection failed: {exc}")
                        self.embedding_func = None  # disable for rest of document

                if embeddings is not None and i < len(sentences):
                    # Cosine similarity between current chunk centroid and next sentence
                    chunk_emb = _centroid(embeddings[max(0, i - len(current_sents)) : i])
                    next_emb = embeddings[i]
                    sim = _cosine_sim(chunk_emb, next_emb)
                    if sim < self.topic_threshold:
                        topic_break = True

            # Also break on paragraph boundaries if we're near the limit
            para_break = False
            if (
                current_sents
                and i > 0
                and sent_paragraph_idx[i] != sent_paragraph_idx[i - 1]
                and current_words > self.max_words * 0.5
            ):
                para_break = True

            if (would_exceed or topic_break or para_break) and current_sents:
                yield " ".join(current_sents)

                # Overlap: carry over last sentences up to overlap budget
                overlap_sents: list[str] = []
                overlap_words = 0
                for s in reversed(current_sents):
                    sw = len(s.split())
                    if overlap_words + sw > self.overlap_words:
                        break
                    overlap_sents.insert(0, s)
                    overlap_words += sw

                current_sents = overlap_sents
                current_words = overlap_words

            current_sents.append(sentence)
            current_words += s_words

        # Yield final chunk
        if current_sents:
            yield " ".join(current_sents)


def _centroid(embeddings: list) -> list:
    """Compute the mean of a list of embedding vectors."""
    if not embeddings:
        return embeddings[0] if embeddings else []
    dim = len(embeddings[0])
    return [sum(e[i] for e in embeddings) / len(embeddings) for i in range(dim)]


def _cosine_sim(a: list, b: list) -> float:
    """Cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
