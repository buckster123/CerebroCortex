"""Unified ingestion pipeline for CerebroCortex.

All file imports route through :py:class:`IngestionPipeline` which guarantees
that memories go through the full encoding pipeline (Thalamus gating, Semantic
enrichment, Amygdala emotion tagging, auto-linking) instead of bypassing it.

Usage::

    from cerebro.ingestion import IngestionPipeline
    pipeline = IngestionPipeline(cortex)
    report = pipeline.ingest_file(Path("notes.md"), tags=["project"])
"""

from cerebro.ingestion.base import IngestionAdapter, IngestionResult
from cerebro.ingestion.csv_adapter import CSVAdapter
from cerebro.ingestion.html_adapter import HTMLAdapter
from cerebro.ingestion.image_adapter import ImageAdapter
from cerebro.ingestion.pdf_adapter import PDFAdapter
from cerebro.ingestion.pipeline import IngestionPipeline

__all__ = [
    "IngestionAdapter",
    "IngestionResult",
    "IngestionPipeline",
    "ImageAdapter",
    "PDFAdapter",
    "HTMLAdapter",
    "CSVAdapter",
]
