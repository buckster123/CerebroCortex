"""IngestionPipeline — orchestrates file ingestion through adapters."""

import logging
import time
from pathlib import Path
from typing import Optional

from cerebro.config import (
    DEFAULT_AGENT_ID,
    SEMANTIC_CHUNKING_ENABLED,
    SEMANTIC_CHUNK_OVERLAP,
    SEMANTIC_CHUNK_SIZE,
)
from cerebro.ingestion.base import IngestionAdapter, IngestionResult
from cerebro.ingestion.chunker import SemanticChunker
from cerebro.ingestion.csv_adapter import CSVAdapter
from cerebro.ingestion.html_adapter import HTMLAdapter
from cerebro.ingestion.image_adapter import ImageAdapter
from cerebro.ingestion.json_adapter import JSONAdapter
from cerebro.ingestion.markdown_adapter import MarkdownAdapter
from cerebro.ingestion.pdf_adapter import PDFAdapter
from cerebro.ingestion.text_adapter import TextAdapter

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """Routes files to the correct adapter and collects results."""

    def __init__(self, cortex, chunker: Optional[SemanticChunker] = None):
        self.cortex = cortex
        # Build chunker if not provided and semantic chunking is enabled
        if chunker is None and SEMANTIC_CHUNKING_ENABLED:
            from cerebro.storage.embeddings import get_embedding_function

            try:
                embed_fn = get_embedding_function("auto")
                chunker = SemanticChunker(
                    max_tokens=SEMANTIC_CHUNK_SIZE,
                    overlap_tokens=SEMANTIC_CHUNK_OVERLAP,
                    embedding_func=embed_fn,
                )
            except Exception as exc:
                logger.warning(f"Failed to load embedding function for semantic chunking: {exc}")
                chunker = None

        self.chunker = chunker
        self.adapters: list[IngestionAdapter] = [
            ImageAdapter(),
            PDFAdapter(),
            HTMLAdapter(),
            CSVAdapter(),
            MarkdownAdapter(chunker=self.chunker),
            JSONAdapter(),
            TextAdapter(chunker=self.chunker),
        ]

    def ingest_file(
        self,
        path: Path,
        *,
        tags: Optional[list[str]] = None,
        agent_id: str = "CLAUDE",
        session_id: Optional[str] = None,
    ) -> IngestionResult:
        """Ingest a single file.

        Args:
            path: File to ingest.
            tags: Additional tags for all memories created.
            agent_id: Agent storing the memories.
            session_id: Optional session ID for episode tracking.

        Returns:
            IngestionResult with counts and errors.
        """
        path = Path(path)
        if not path.exists():
            return IngestionResult(
                errors=[f"File not found: {path}"], duration_seconds=0.0
            )

        adapter = next((a for a in self.adapters if a.can_ingest(path)), None)
        if adapter is None:
            return IngestionResult(
                errors=[f"No adapter for file type: {path.suffix}"],
                duration_seconds=0.0,
            )

        return adapter.ingest(
            path,
            cortex=self.cortex,
            tags=tags,
            agent_id=agent_id,
            session_id=session_id,
        )

    def ingest_text(
        self,
        text: str,
        *,
        title: Optional[str] = None,
        tags: Optional[list[str]] = None,
        agent_id: str = "CLAUDE",
        session_id: Optional[str] = None,
    ) -> IngestionResult:
        """Ingest raw text directly via the text adapter.

        This is a convenience wrapper around :py:class:`TextAdapter`.
        """
        from cerebro.ingestion.text_adapter import TextAdapter

        adapter = TextAdapter()
        return adapter.ingest_text(
            text,
            cortex=self.cortex,
            title=title,
            tags=tags,
            agent_id=agent_id,
            session_id=session_id,
        )
