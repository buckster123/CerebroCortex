"""Import memories from Markdown files into CerebroCortex.

Supports two formats:

1. Section-based: each ## heading starts a new memory
   ```markdown
   ## Python Type Hints
   Python 3.5+ supports type hints for function parameters and return types.
   Use `def foo(x: int) -> str` syntax.

   ## Error Handling
   Always catch specific exceptions, not bare `except:`.
   ```

2. Paragraph-based: each blank-line-separated paragraph becomes a memory
   (used when no ## headings are found)

Optional YAML frontmatter for defaults:
   ```markdown
   ---
   type: semantic
   tags: [python, programming]
   agent_id: CLAUDE
   ---
   ## ...
   ```
"""

import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from cerebro.cortex import CerebroCortex
from cerebro.models.memory import MemoryMetadata, MemoryNode, StrengthState
from cerebro.types import MemoryType


TYPE_MAP: dict[str, MemoryType] = {
    "episodic": MemoryType.EPISODIC,
    "semantic": MemoryType.SEMANTIC,
    "procedural": MemoryType.PROCEDURAL,
    "affective": MemoryType.AFFECTIVE,
    "prospective": MemoryType.PROSPECTIVE,
    "schematic": MemoryType.SCHEMATIC,
}


@dataclass
class MarkdownImportReport:
    """Report from a Markdown import operation."""
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


class MarkdownImporter:
    """Import memories from Markdown files."""

    def __init__(self, cortex: CerebroCortex):
        self.cortex = cortex

    def import_file(self, path: Path) -> MarkdownImportReport:
        """Import from a Markdown file."""
        text = path.read_text(encoding="utf-8")
        return self.import_text(text)

    def import_text(self, text: str) -> MarkdownImportReport:
        """Import from Markdown text."""
        report = MarkdownImportReport()
        start = time.time()

        # Parse optional YAML frontmatter
        defaults, body = self._parse_frontmatter(text)

        # Extract sections
        sections = self._extract_sections(body)

        default_type = TYPE_MAP.get(
            defaults.get("type", "semantic"), MemoryType.SEMANTIC
        )
        default_tags = defaults.get("tags", [])
        if isinstance(default_tags, str):
            default_tags = [t.strip() for t in default_tags.split(",")]
        default_agent = defaults.get("agent_id", "CLAUDE")

        for title, content in sections:
            content = content.strip()
            if not content or len(content) < 3:
                report.memories_skipped += 1
                continue

            # Use title as first line if present
            if title:
                full_content = f"{title}: {content}"
            else:
                full_content = content

            # Check dedup
            if self.cortex.graph.find_duplicate_content(full_content):
                report.memories_skipped += 1
                continue

            # Build tags: defaults + title-derived tag
            tags = list(default_tags)
            if title:
                tag = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
                if tag and tag not in tags:
                    tags.append(tag)

            now = time.time()
            node = MemoryNode(
                content=full_content,
                metadata=MemoryMetadata(
                    memory_type=default_type,
                    tags=tags,
                    agent_id=default_agent,
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
                report.errors.append(f"Section '{title}': {e}")
                report.memories_skipped += 1

        self.cortex.graph.resync_igraph()
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

            # Parse simple lists: [a, b, c]
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
        # Check for ## headings
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

        # Fallback: paragraph-based splitting
        paragraphs = re.split(r"\n\s*\n", text)
        return [("", p.strip()) for p in paragraphs if p.strip()]
