"""JSON ingestion adapter.

Supports two formats:
1. Simple list of strings or dicts with content, type, tags, salience
2. Full format with nested metadata/strength objects
"""

import json
import time
from pathlib import Path
from typing import Optional

from cerebro.config import DEFAULT_AGENT_ID
from cerebro.ingestion.base import IngestionAdapter, IngestionResult
from cerebro.types import MemoryType


class JSONAdapter(IngestionAdapter):
    """Import memories from JSON files."""

    SUPPORTED = {".json"}

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
        """Import from a JSON file."""
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
        """Import from JSON text."""
        report = IngestionResult()
        start = time.time()

        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            report.errors.append(f"Invalid JSON: {e}")
            report.duration_seconds = time.time() - start
            return report

        if not isinstance(data, list):
            # Accept {"memories": [...]} or {"records": [...]} wrapper
            if isinstance(data, dict):
                data = data.get("memories") or data.get("records") or []
            else:
                report.errors.append("JSON root must be a list or dict")
                report.duration_seconds = time.time() - start
                return report

        contents = []
        meta_overrides = []
        for i, record in enumerate(data):
            if isinstance(record, str):
                contents.append(record)
                meta_overrides.append({})
            elif isinstance(record, dict):
                content = record.get("content", "")
                if not content or len(content) < 3:
                    report.memories_skipped += 1
                    continue
                contents.append(content)
                meta_overrides.append({
                    "memory_type": record.get("type", "semantic"),
                    "tags": record.get("tags", []),
                    "salience": record.get("salience"),
                })
            else:
                report.errors.append(f"Record {i}: expected string or dict")
                report.memories_skipped += 1

        for content, override in zip(contents, meta_overrides):
            mem_type = None
            if override.get("memory_type"):
                try:
                    mem_type = MemoryType(override["memory_type"])
                except ValueError:
                    pass

            rec_tags = list(tags or []) + list(override.get("tags") or [])

            node = cortex.remember(
                content=content,
                memory_type=mem_type,
                tags=rec_tags,
                salience=override.get("salience"),
                agent_id=agent_id,
                session_id=session_id,
            )
            if node:
                report.memories_imported += 1
            else:
                report.memories_skipped += 1

        report.duration_seconds = time.time() - start
        return report
