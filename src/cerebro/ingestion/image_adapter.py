"""Image ingestion adapter for CerebroCortex.

Generates a text caption (via local vision model or filename fallback),
optionally extracts OCR text, and stores the image as a memory with an
Attachment record. If the vision sidecar is available, also generates a
vision embedding for cross-modal search.

Optional dependencies (install via ``pip install cerebro-cortex[vision]``):
- ``sentence-transformers`` for CLIP vision embeddings
- ``pillow`` for image handling
- ``pytesseract`` for OCR
"""

import hashlib
import logging
import mimetypes
from pathlib import Path
from typing import Optional

from cerebro.ingestion.base import IngestionAdapter, IngestionResult
from cerebro.models.attachment import Attachment
from cerebro.types import MediaType, MemoryType

logger = logging.getLogger(__name__)


class ImageAdapter(IngestionAdapter):
    """Ingest images: generate caption, OCR, store with attachment + vision embedding."""

    SUPPORTED = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff", ".tif"}

    def can_ingest(self, path: Path) -> bool:
        return path.suffix.lower() in self.SUPPORTED

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

        # 1. Generate caption
        caption = self._caption_image(path)

        # 2. OCR (optional)
        ocr_text = self._ocr_image(path)

        # 3. Build content
        content = caption
        if ocr_text:
            content += f"\n[OCR]: {ocr_text}"

        # 4. Build attachment
        attachment = Attachment(
            mime_type=self._mime_type(path),
            media_type=MediaType.IMAGE,
            file_path=str(path.resolve()),
            original_bytes_hash=self._hash_file(path),
            text_description=caption,
        )

        # 5. Store memory via full pipeline
        node = cortex.remember(
            content=content,
            memory_type=MemoryType.SEMANTIC,
            tags=(tags or []) + ["image", path.suffix.lstrip(".")],
            agent_id=agent_id,
            session_id=session_id,
        )

        if node is None:
            return IngestionResult(
                errors=[f"Gating rejected image: {path.name}"],
                duration_seconds=__import__("time").time() - start,
            )

        # 6. Persist attachment metadata to the memory node
        node.metadata.attachments.append(attachment)
        node.metadata.media_type = MediaType.IMAGE
        node.metadata.source_file = str(path.resolve())
        # Re-sync to SQLite (attachments table) and ChromaDB (metadata)
        cortex.graph._insert_attachment(node.id, attachment)
        cortex.coordinator.update_node(node)

        # 7. Vision embedding (optional sidecar)
        vision_id: Optional[str] = None
        if hasattr(cortex, "_vision_store") and cortex._vision_store is not None:
            vision_id = cortex._vision_store.add_image(
                attachment_id=attachment.id,
                image_path=str(path),
                memory_id=node.id,
            )
            if vision_id:
                attachment.vision_embedding_id = vision_id
                cortex.graph._insert_attachment(node.id, attachment)

        duration = __import__("time").time() - start
        return IngestionResult(
            memories_imported=1,
            memories_created=[node.id],
            attachments_created=[attachment.id],
            duration_seconds=duration,
        )

    # ------------------------------------------------------------------
    # Caption generation
    # ------------------------------------------------------------------

    def _caption_image(self, path: Path) -> str:
        """Generate a text caption for an image.

        Tries (in order):
        1. Ollama local vision model (llava / bakllava)
        2. Filename-based fallback
        """
        caption = self._caption_via_ollama(path)
        if caption:
            return caption
        return f"[Image: {path.name}]"

    def _caption_via_ollama(self, path: Path) -> Optional[str]:
        """Try Ollama vision model for caption generation."""
        try:
            import base64

            import requests
        except ImportError:
            return None

        try:
            with open(path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
        except OSError:
            return None

        # Try common vision models
        for model in ("llava", "bakllava", "llava-phi3"):
            try:
                resp = requests.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": model,
                        "prompt": "Describe this image in one sentence.",
                        "images": [b64],
                        "stream": False,
                    },
                    timeout=60,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    text = data.get("response", "").strip()
                    if text:
                        return text
            except Exception:
                continue
        return None

    # ------------------------------------------------------------------
    # OCR
    # ------------------------------------------------------------------

    def _ocr_image(self, path: Path) -> Optional[str]:
        """Extract text from image using pytesseract (optional)."""
        try:
            import pytesseract
            from PIL import Image

            img = Image.open(path)
            text = pytesseract.image_to_string(img).strip()
            return text if text else None
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _mime_type(path: Path) -> str:
        mt, _ = mimetypes.guess_type(str(path))
        return mt or "application/octet-stream"

    @staticmethod
    def _hash_file(path: Path) -> str:
        h = hashlib.sha256()
        try:
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    h.update(chunk)
            return h.hexdigest()
        except OSError:
            return ""
