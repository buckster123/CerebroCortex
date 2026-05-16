"""Markdown ingestion adapter.

Supports section-based extraction (## headings) or paragraph fallback.
Optional YAML frontmatter for defaults.
"""

import re
import time
from pathlib import Path
from typing import Optional

from cerebro.config import DEFAULT_AGENT_ID
from cerebro.ingestion.base import IngestionAdapter, IngestionResult
from cerebro.ingestion.chunker import SemanticChunker
from cerebro.types import MemoryType


TYPE_MAP: dict[str, MemoryType] = {
    "episodic": MemoryType.EPISODIC,
    "semantic": MemoryType.SEMANTIC,
    "procedural": MemoryType.PROCEDURAL,
    "affective": MemoryType.AFFECTIVE,
    "prospective": MemoryType.PROSPECTIVE,
    "schematic": MemoryType.SCHEMATIC,
}


class MarkdownAdapter(IngestionAdapter):
    """Import memories from Markdown files."""

    SUPPORTED = {".md", ".markdown", ".mdown", ".mkd"}

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
        """Import from a Markdown file."""
        text = path.read_text(encoding="utf-8")
        return self.ingest_text(
            text,
            cortex=cortex,
            tags=tags,
            agent_id=agent_id,
            session_id=session_id,
        )

    def ingest_text(
        self,
        text: str,
        *,
        cortex,
        tags: Optional[list[str]] = None,
        agent_id: str = DEFAULT_AGENT_ID,
        session_id: Optional[str] = None,
    ) -> IngestionResult:
        """Import from Markdown text."""
        from cerebro.ingestion.text_adapter import TextAdapter

        report = IngestionResult()
        start = time.time()

        defaults, body = self._parse_frontmatter(text)
        sections = self._extract_sections(body)

        default_type = TYPE_MAP.get(defaults.get("type", "semantic"), MemoryType.SEMANTIC)
        default_tags = defaults.get("tags", [])
        if isinstance(default_tags, str):
            default_tags = [t.strip() for t in default_tags.split(",")]
        default_agent = defaults.get("agent_id", agent_id)

        # If no headings and we have a semantic chunker, use it for better paragraph chunking
        heading_pattern = re.compile(r"^##\s+(.+)$", re.MULTILINE)
        has_headings = bool(heading_pattern.search(text))

        if not has_headings and self.chunker is not None:
            # Delegate to TextAdapter with semantic chunking for flat markdown
            text_adapter = TextAdapter(chunker=self.chunker)
            return text_adapter.ingest_text(
                body,
                cortex=cortex,
                title=defaults.get("title") or "markdown",
                tags=list(tags or []) + list(default_tags),
                agent_id=default_agent,
                session_id=session_id,
            )

        contents = []
        section_tags = []
        for title, content in sections:
            content = content.strip()
            if not content or len(content) < 3:
                report.memories_skipped += 1
                continue

            full_content = f"{title}: {content}" if title else content
            contents.append(full_content)

            sec_tags = list(default_tags)
            if title:
                tag = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
                if tag and tag not in sec_tags:
                    sec_tags.append(tag)
            section_tags.append(sec_tags)

        # Bulk remember with per-memory tags via individual calls
        # (bulk_remember doesn't support per-item tags, so we loop)
        for full_content, sec_tags in zip(contents, section_tags):
            final_tags = list(tags or []) + sec_tags
            node = cortex.remember(
                content=full_content,
                memory_type=default_type,
                tags=final_tags,
                agent_id=default_agent,
                session_id=session_id,
            )
            if node:
                report.memories_imported += 1
            else:
                report.memories_skipped += 1

        report.duration_seconds = time.time() - start
        return report

    @staticmethod
    def _parse_frontmatter(text: str) -> tuple[dict, str]:
        """Parse YAML frontmatter delimited by --- lines.

        Returns (frontmatter_dict, remaining_text).
        Uses simple key: value parsing to avoid PyYAML dependency.
        """
        text = text.lstrip()
        if not text.startswith("---"):
            return {}, text

        lines = text.split("\n")
        end_idx = None
        for i in range(1, len(lines)):
            if lines[i].strip() == "---":
                end_idx = i
                break

        if end_idx is None:
            return {}, text

        frontmatter = {}
        for line in lines[1:end_idx]:
            line = line.strip()
            if not line or ":" not in line:
                continue
            key, val = line.split(":", 1)
            key = key.strip()
            val = val.strip()

            if val.startswith("[") and val.endswith("]"):
                items = val[1:-1].split(",")
                frontmatter[key] = [item.strip().strip("'\"") for item in items if item.strip()]
            elif val.lower() in ("true", "false"):
                frontmatter[key] = val.lower() == "true"
            else:
                frontmatter[key] = val.strip("'\"")

        body = "\n".join(lines[end_idx + 1:])
        return frontmatter, body

    @staticmethod
    def _extract_sections(text: str) -> list[tuple[str, str]]:
        """Extract sections from markdown text.

        If ## headings exist, each heading starts a section.
        Otherwise, split by blank lines into paragraphs.
        """
        heading_pattern = re.compile(r"^##\s+(.+)$", re.MULTILINE)
        headings = list(heading_pattern.finditer(text))

        if headings:
            sections = []
            for i, match in enumerate(headings):
                title = match.group(1).strip()
                start = match.end()
                end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
                body = text[start:end].strip()
                sections.append((title, body))
            return sections

        paragraphs = re.split(r"\n\s*\n", text)
        return [("", p.strip()) for p in paragraphs if p.strip()]
