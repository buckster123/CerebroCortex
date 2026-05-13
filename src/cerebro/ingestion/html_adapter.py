"""HTML ingestion adapter for CerebroCortex.

Extracts readable text from HTML, strips scripts/styles, handles <img> tags
by ingesting each image via ImageAdapter, and stores the page as a memory.

Optional dependency: ``beautifulsoup4``.
"""

import logging
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse

from cerebro.ingestion.base import IngestionAdapter, IngestionResult
from cerebro.types import MemoryType

logger = logging.getLogger(__name__)


class HTMLAdapter(IngestionAdapter):
    """Ingest HTML files or saved web pages."""

    SUPPORTED = {".html", ".htm"}

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

        try:
            from bs4 import BeautifulSoup
        except ImportError:
            return IngestionResult(
                errors=[
                    "beautifulsoup4 not installed. Install with: pip install beautifulsoup4"
                ],
                duration_seconds=__import__("time").time() - start,
            )

        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            return IngestionResult(
                errors=[f"Failed to read HTML: {exc}"],
                duration_seconds=__import__("time").time() - start,
            )

        soup = BeautifulSoup(raw, "html.parser")

        # Strip script/style/nav/footer tags
        for tag_name in ("script", "style", "nav", "footer", "aside", "noscript"):
            for tag in soup.find_all(tag_name):
                tag.decompose()

        title = ""
        if soup.title and soup.title.string:
            title = soup.title.string.strip()

        # Extract text
        text = soup.get_text(separator="\n", strip=True)
        if not text.strip():
            return IngestionResult(
                errors=["No extractable text found in HTML"],
                duration_seconds=__import__("time").time() - start,
            )

        # Build content with title
        content = text
        if title:
            content = f"# {title}\n\n{text}"

        tags = (tags or []) + ["html"]
        if title:
            tags.append(f"title:{title[:50]}")

        # Store main page memory
        node = cortex.remember(
            content=content,
            memory_type=MemoryType.SEMANTIC,
            tags=tags,
            agent_id=agent_id,
            session_id=session_id,
        )

        if node is None:
            return IngestionResult(
                errors=["Gating rejected HTML content"],
                duration_seconds=__import__("time").time() - start,
            )

        node.metadata.source_file = str(path.resolve())
        cortex.coordinator.update_node(node)

        # Handle <img> tags
        attachments_created: list[str] = []
        base_url = ""
        base_tag = soup.find("base", href=True)
        if base_tag:
            base_url = base_tag["href"]

        for img in soup.find_all("img"):
            src = img.get("src")
            if not src:
                continue
            # Resolve relative URLs/paths
            if base_url:
                src = urljoin(base_url, src)
            img_path = Path(src)
            if not img_path.is_absolute():
                img_path = path.parent / src
            if img_path.exists():
                from cerebro.ingestion.image_adapter import ImageAdapter

                img_result = ImageAdapter().ingest(
                    img_path,
                    cortex=cortex,
                    tags=tags + ["html-image"],
                    agent_id=agent_id,
                    session_id=session_id,
                )
                attachments_created.extend(img_result.attachments_created)

        duration = __import__("time").time() - start
        return IngestionResult(
            memories_imported=1,
            memories_created=[node.id],
            attachments_created=attachments_created,
            duration_seconds=duration,
        )
