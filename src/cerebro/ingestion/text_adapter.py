"""Text ingestion adapter.

Supports .txt, .py, .js, .ts, .rs, .go, .java, .rb, .sh, and other text files.
Chunking strategy:
1. Split by blank lines into paragraphs
2. If a paragraph exceeds ~500 words, split at sentence boundaries
3. Skip chunks shorter than 10 characters
"""

import re
import time
from pathlib import Path
from typing import Optional

from cerebro.config import DEFAULT_AGENT_ID
from cerebro.ingestion.base import IngestionAdapter, IngestionResult
from cerebro.ingestion.chunker import SemanticChunker

MAX_CHUNK_WORDS = 500
MIN_CHUNK_LENGTH = 10


class TextAdapter(IngestionAdapter):
    """Import memories from plain text and code files."""

    SUPPORTED = {
        ".txt", ".py", ".js", ".ts", ".rs", ".go", ".java", ".rb", ".sh",
        ".c", ".cpp", ".h", ".hpp", ".cs", ".swift", ".kt", ".scala",
        ".r", ".m", ".mm", ".sql", ".yaml", ".yml", ".json", ".xml",
        ".toml", ".ini", ".cfg", ".conf", ".properties", ".md", ".rst",
    }

    def __init__(self, chunker: Optional[SemanticChunker] = None):
        self.chunker = chunker

    def can_ingest(self, path: Path) -> bool:
        return path.suffix.lower() in self.SUPPORTED

    def ingest(
        self,
        path: Path,
        *,
        cortex,
        tags: Optional[list[str]] = None,
        agent_id: str = DEFAULT_AGENT_ID,
        session_id: Optional[str] = None,
    ) -> IngestionResult:
        """Import from a text file."""
        text = path.read_text(encoding="utf-8", errors="replace")
        return self.ingest_text(
            text,
            cortex=cortex,
            title=path.name,
            tags=tags,
            agent_id=agent_id,
            session_id=session_id,
        )

    def ingest_text(
        self,
        text: str,
        *,
        cortex,
        title: Optional[str] = None,
        tags: Optional[list[str]] = None,
        agent_id: str = DEFAULT_AGENT_ID,
        session_id: Optional[str] = None,
    ) -> IngestionResult:
        """Import from raw text content."""
        from cerebro.types import MemoryType

        report = IngestionResult()
        start = time.time()

        # Use semantic chunker if available, otherwise fall back to legacy
        if self.chunker is not None:
            chunks = list(self.chunker.chunk(text))
        else:
            chunks = self._chunk_text(text)

        base_tags = list(tags or [])
        if title:
            base_tags.append(f"source:{title}")

        contents = []
        for chunk in chunks:
            chunk = chunk.strip()
            if len(chunk) < MIN_CHUNK_LENGTH:
                report.memories_skipped += 1
                continue
            contents.append(chunk)

        # Use bulk_remember for efficiency (skips per-chunk vector search)
        nodes = cortex.bulk_remember(
            contents=contents,
            memory_type=MemoryType.SEMANTIC,
            tags=base_tags,
            agent_id=agent_id,
            session_id=session_id,
        )

        for node in nodes:
            if node is not None:
                report.memories_imported += 1
            else:
                report.memories_skipped += 1

        report.duration_seconds = time.time() - start
        return report

    @staticmethod
    def _chunk_text(text: str) -> list[str]:
        """Legacy chunking: split by paragraphs, with large-paragraph splitting."""
        paragraphs = re.split(r"\n\s*\n", text)
        chunks = []

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            words = para.split()
            if len(words) <= MAX_CHUNK_WORDS:
                chunks.append(para)
            else:
                sentences = re.split(r"(?<=[.!?])\s+", para)
                current: list[str] = []
                current_words = 0

                for sentence in sentences:
                    s_words = len(sentence.split())
                    if current_words + s_words > MAX_CHUNK_WORDS and current:
                        chunks.append(" ".join(current))
                        current = []
                        current_words = 0
                    current.append(sentence)
                    current_words += s_words

                if current:
                    chunks.append(" ".join(current))

        return chunks
