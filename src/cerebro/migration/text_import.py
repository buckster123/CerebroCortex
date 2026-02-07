"""Import memories from plain text and code files into CerebroCortex.

Chunking strategy:
1. Split by blank lines into paragraphs
2. If a paragraph exceeds ~500 words, split at sentence boundaries
3. Skip chunks shorter than 10 characters

Supports .txt, .py, .js, .ts, .rs, .go, .java, .rb, .sh, and other text files.
"""

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from cerebro.config import DEFAULT_AGENT_ID
from cerebro.cortex import CerebroCortex
from cerebro.models.memory import MemoryMetadata, MemoryNode, StrengthState
from cerebro.types import MemoryType


MAX_CHUNK_WORDS = 500
MIN_CHUNK_LENGTH = 10


@dataclass
class TextImportReport:
    """Report from a text import operation."""
    memories_imported: int = 0
    memories_skipped: int = 0
    errors: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "memories_imported": self.memories_imported,
            "memories_skipped": self.memories_skipped,
            "errors": self.errors[:20],
            "total_errors": len(self.errors),
            "duration_seconds": round(self.duration_seconds, 2),
        }


class TextImporter:
    """Import memories from plain text and code files."""

    def __init__(self, cortex: CerebroCortex):
        self.cortex = cortex

    def import_file(
        self,
        path: Path,
        tags: Optional[list[str]] = None,
        agent_id: str = DEFAULT_AGENT_ID,
    ) -> TextImportReport:
        """Import from a text file."""
        text = path.read_text(encoding="utf-8", errors="replace")
        return self.import_text(text, tags=tags, agent_id=agent_id)

    def import_text(
        self,
        text: str,
        tags: Optional[list[str]] = None,
        agent_id: str = DEFAULT_AGENT_ID,
    ) -> TextImportReport:
        """Import from raw text content."""
        report = TextImportReport()
        start = time.time()

        chunks = self._chunk_text(text)
        base_tags = list(tags or [])

        for i, chunk in enumerate(chunks):
            chunk = chunk.strip()
            if len(chunk) < MIN_CHUNK_LENGTH:
                report.memories_skipped += 1
                continue

            if self.cortex.graph.find_duplicate_content(chunk):
                report.memories_skipped += 1
                continue

            now = time.time()
            node = MemoryNode(
                content=chunk,
                metadata=MemoryMetadata(
                    memory_type=MemoryType.SEMANTIC,
                    tags=base_tags.copy(),
                    agent_id=agent_id,
                    source="import",
                ),
                strength=StrengthState(
                    access_timestamps=[now],
                    access_count=1,
                    last_computed_at=now,
                ),
            )

            try:
                self.cortex.graph.add_node(node)
                coll = self.cortex._collection_for_type(node.metadata.memory_type)
                self.cortex.vector.add_node(coll, node)
                report.memories_imported += 1
            except Exception as e:
                report.errors.append(f"Chunk {i}: {e}")
                report.memories_skipped += 1

        self.cortex.graph.resync_igraph()
        report.duration_seconds = time.time() - start
        return report

    @staticmethod
    def _chunk_text(text: str) -> list[str]:
        """Split text into chunks by paragraphs, with large-paragraph splitting."""
        # Split on blank lines
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
                # Split long paragraphs at sentence boundaries
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
