"""Tests for multimodal ingestion adapters (Phase B)."""

import tempfile
from pathlib import Path

import pytest

from cerebro.cortex import CerebroCortex
from cerebro.ingestion import IngestionPipeline
from cerebro.ingestion.csv_adapter import CSVAdapter
from cerebro.ingestion.html_adapter import HTMLAdapter
from cerebro.ingestion.image_adapter import ImageAdapter
from cerebro.ingestion.pdf_adapter import PDFAdapter
from cerebro.models.attachment import Attachment
from cerebro.types import MediaType


@pytest.fixture
def temp_cortex():
    """CerebroCortex with temp storage."""
    with tempfile.TemporaryDirectory() as d:
        ctx = CerebroCortex(
            db_path=Path(d) / "test.db",
            chroma_dir=Path(d) / "chroma",
        )
        ctx.initialize()
        yield ctx
        ctx.close()


class TestCSVAdapter:
    def test_can_ingest_csv(self):
        assert CSVAdapter().can_ingest(Path("data.csv"))
        assert not CSVAdapter().can_ingest(Path("data.txt"))

    def test_ingest_small_csv(self, temp_cortex, tmp_path):
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("name,age,city\nAlice,30,NYC\nBob,25,LA\n")

        adapter = CSVAdapter()
        result = adapter.ingest(csv_path, cortex=temp_cortex)

        assert result.memories_imported == 2
        assert len(result.memories_created) == 2
        assert not result.errors

    def test_ingest_large_csv_schema_mode(self, temp_cortex, tmp_path):
        csv_path = tmp_path / "big.csv"
        lines = ["id,name"] + [f"{i},item{i}" for i in range(250)]
        csv_path.write_text("\n".join(lines))

        adapter = CSVAdapter()
        result = adapter.ingest(csv_path, cortex=temp_cortex)

        # Should switch to schema-only mode
        assert result.memories_imported == 1
        assert not result.errors

    def test_ingest_empty_csv(self, temp_cortex, tmp_path):
        csv_path = tmp_path / "empty.csv"
        csv_path.write_text("")

        adapter = CSVAdapter()
        result = adapter.ingest(csv_path, cortex=temp_cortex)

        assert result.memories_imported == 0
        assert result.errors


class TestHTMLAdapter:
    def test_can_ingest_html(self):
        assert HTMLAdapter().can_ingest(Path("page.html"))
        assert HTMLAdapter().can_ingest(Path("page.htm"))
        assert not HTMLAdapter().can_ingest(Path("page.txt"))

    def test_ingest_html(self, temp_cortex, tmp_path):
        pytest.importorskip("bs4")
        html_path = tmp_path / "test.html"
        html_path.write_text(
            "<html><head><title>Test Page</title></head>"
            "<body><h1>Hello</h1><p>This is a test.</p></body></html>"
        )

        adapter = HTMLAdapter()
        result = adapter.ingest(html_path, cortex=temp_cortex)

        assert result.memories_imported == 1
        assert not result.errors

    def test_ingest_html_no_text(self, temp_cortex, tmp_path):
        pytest.importorskip("bs4")
        html_path = tmp_path / "empty.html"
        html_path.write_text("<html><script>alert('hi')</script></html>")

        adapter = HTMLAdapter()
        result = adapter.ingest(html_path, cortex=temp_cortex)

        assert result.memories_imported == 0
        assert result.errors


class TestImageAdapter:
    def test_can_ingest_image(self):
        assert ImageAdapter().can_ingest(Path("photo.png"))
        assert ImageAdapter().can_ingest(Path("photo.jpg"))
        assert not ImageAdapter().can_ingest(Path("doc.pdf"))

    def test_caption_fallback(self, tmp_path):
        adapter = ImageAdapter()
        img_path = tmp_path / "screenshot_login_bug.png"
        caption = adapter._caption_image(img_path)
        assert "screenshot_login_bug.png" in caption

    def test_hash_file(self, tmp_path):
        adapter = ImageAdapter()
        txt = tmp_path / "test.txt"
        txt.write_text("hello")
        h = adapter._hash_file(txt)
        assert len(h) == 64  # sha256 hex

    def test_mime_type(self):
        adapter = ImageAdapter()
        assert adapter._mime_type(Path("test.png")) == "image/png"
        assert adapter._mime_type(Path("test.jpg")) == "image/jpeg"


class TestPDFAdapter:
    def test_can_ingest_pdf(self):
        assert PDFAdapter().can_ingest(Path("doc.pdf"))
        assert not PDFAdapter().can_ingest(Path("doc.txt"))

    def test_ingest_pdf_no_deps(self, temp_cortex, tmp_path):
        """PDF adapter gracefully handles missing PyMuPDF."""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_text("not a real pdf")

        adapter = PDFAdapter()
        result = adapter.ingest(pdf_path, cortex=temp_cortex)

        assert result.errors
        assert "pymupdf" in result.errors[0].lower()


class TestIngestionPipeline:
    def test_routes_csv(self, temp_cortex, tmp_path):
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("a,b\n1,2\n")

        pipeline = IngestionPipeline(temp_cortex)
        result = pipeline.ingest_file(csv_path)

        assert result.memories_imported == 1

    def test_routes_html(self, temp_cortex, tmp_path):
        pytest.importorskip("bs4")
        html_path = tmp_path / "page.html"
        html_path.write_text("<html><body>Hello</body></html>")

        pipeline = IngestionPipeline(temp_cortex)
        result = pipeline.ingest_file(html_path)

        assert result.memories_imported == 1

    def test_no_adapter(self, temp_cortex, tmp_path):
        weird = tmp_path / "data.unknown"
        weird.write_text("???")

        pipeline = IngestionPipeline(temp_cortex)
        result = pipeline.ingest_file(weird)

        assert result.errors
        assert "No adapter" in result.errors[0]

    def test_file_not_found(self, temp_cortex):
        pipeline = IngestionPipeline(temp_cortex)
        result = pipeline.ingest_file(Path("/nonexistent/file.csv"))

        assert result.errors
        assert "not found" in result.errors[0].lower()
