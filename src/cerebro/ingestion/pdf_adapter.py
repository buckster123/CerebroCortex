"""PDF ingestion adapter for CerebroCortex.

Extracts text (chunked), extracts embedded images, and stores everything as
linked memories. Text chunks are stored as SEMANTIC memories; images are
stored as separate memories with IMAGE attachments and linked via PART_OF.

Optional dependency: ``pip install pymupdf`` (also known as ``fitz``).
"""

import logging
import tempfile
from pathlib import Path
from typing import Optional

from cerebro.ingestion.base import IngestionAdapter, IngestionResult
from cerebro.types import LinkType, MemoryType

logger = logging.getLogger(__name__)


class PDFAdapter(IngestionAdapter):
    """Ingest PDFs: text extraction + image extraction + linked memories."""

    def can_ingest(self, path: Path) -> bool:
        return path.suffix.lower() == ".pdf"

    def ingest(
        self,
        path: Path,
        *,
        cortex,
        tags: Optional[list[str]] = None,
        agent_id: str = "CLAUDE",
        session_id: Optional[str] = None,
    ) -> IngestionResult:
        path = Path(path)
        start = __import__("time").time()

        try:
            import fitz  # PyMuPDF
        except ImportError:
            return IngestionResult(
                errors=[
                    "PyMuPDF not installed. Install with: pip install pymupdf"
                ],
                duration_seconds=__import__("time").time() - start,
            )

        tags = tags or []
        parent_tag = f"pdf:{path.stem}"
        all_tags = tags + [parent_tag, "pdf"]

        # 1. Extract text
        text_chunks = self._extract_text_chunks(path, fitz)

        # 2. Store text chunks as memories
        imported = 0
        all_memory_ids: list[str] = []
        for chunk in text_chunks:
            node = cortex.remember(
                content=chunk,
                memory_type=MemoryType.SEMANTIC,
                tags=all_tags,
                agent_id=agent_id,
                session_id=session_id,
            )
            if node:
                imported += 1
                all_memory_ids.append(node.id)
                node.metadata.source_file = str(path.resolve())
                cortex.coordinator.update_node(node)

        # 3. Extract and store images
        images = self._extract_images(path, fitz)
        attachments_created: list[str] = []
        for img_path in images:
            # Use ImageAdapter for consistent image handling
            from cerebro.ingestion.image_adapter import ImageAdapter

            img_adapter = ImageAdapter()
            img_result = img_adapter.ingest(
                img_path,
                cortex=cortex,
                tags=all_tags,
                agent_id=agent_id,
                session_id=session_id,
            )
            if img_result.memories_created:
                img_id = img_result.memories_created[0]
                attachments_created.extend(img_result.attachments_created)
                # Link image to first few text chunks
                for mid in all_memory_ids[:3]:
                    cortex.associate(
                        img_id,
                        mid,
                        LinkType.PART_OF,
                        evidence=f"Extracted from {path.name}",
                    )

        duration = __import__("time").time() - start
        return IngestionResult(
            memories_imported=imported,
            memories_created=all_memory_ids,
            attachments_created=attachments_created,
            duration_seconds=duration,
        )

    def _extract_text_chunks(self, path: Path, fitz_module) -> list[str]:
        """Extract text from PDF and split into semantic chunks."""
        try:
            doc = fitz_module.open(path)
            full_text = "\n\n".join(page.get_text() for page in doc)
            doc.close()
        except Exception as exc:
            logger.warning(f"PDF text extraction failed for {path}: {exc}")
            return []

        if not full_text.strip():
            return []

        # Use semantic chunker if available, otherwise fall back to paragraph merge
        try:
            from cerebro.ingestion.chunker import SemanticChunker
            chunker = SemanticChunker()
            return list(chunker.chunk(full_text))
        except Exception as exc:
            logger.debug(f"Semantic chunking failed for PDF, using fallback: {exc}")

        # Fallback: simple chunking by paragraphs, then merge up to ~500 words
        paragraphs = [p.strip() for p in full_text.split("\n\n") if p.strip()]
        chunks: list[str] = []
        current: list[str] = []
        current_words = 0

        for para in paragraphs:
            para_words = len(para.split())
            if current_words + para_words > 500 and current:
                chunks.append("\n\n".join(current))
                current = [para]
                current_words = para_words
            else:
                current.append(para)
                current_words += para_words

        if current:
            chunks.append("\n\n".join(current))

        return chunks

    def _extract_images(self, path: Path, fitz_module) -> list[Path]:
        """Extract embedded images from PDF to temp files."""
        extracted: list[Path] = []
        try:
            doc = fitz_module.open(path)
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                img_list = page.get_images(full=True)
                for img_index, img in enumerate(img_list, start=1):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    ext = base_image["ext"]
                    # Skip tiny images (likely icons / decorations)
                    if len(image_bytes) < 2048:
                        continue
                    tmp = tempfile.NamedTemporaryFile(
                        suffix=f"_p{page_num}_{img_index}.{ext}",
                        delete=False,
                    )
                    tmp.write(image_bytes)
                    tmp.close()
                    extracted.append(Path(tmp.name))
            doc.close()
        except Exception as exc:
            logger.warning(f"PDF image extraction failed for {path}: {exc}")
        return extracted
