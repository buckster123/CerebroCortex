"""CSV ingestion adapter for CerebroCortex.

Option A (default): Store each row as a memory (good for small CSVs).
Option B: Store schema description + sample rows (good for large CSVs).

Uses only stdlib csv module — no extra dependencies.
"""

import csv
import logging
from pathlib import Path
from typing import Optional

from cerebro.ingestion.base import IngestionAdapter, IngestionResult
from cerebro.types import MemoryType

logger = logging.getLogger(__name__)

# Threshold: if more than this many rows, switch to schema-only mode
ROW_THRESHOLD = 200


class CSVAdapter(IngestionAdapter):
    """Ingest CSV files as structured row memories or schema summary."""

    def can_ingest(self, path: Path) -> bool:
        return path.suffix.lower() == ".csv"

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
            with open(path, "r", encoding="utf-8", newline="") as f:
                sample = f.read(8192)
                f.seek(0)
                dialect = csv.Sniffer().sniff(sample)
                reader = csv.DictReader(f, dialect=dialect)
                rows = list(reader)
        except Exception as exc:
            return IngestionResult(
                errors=[f"Failed to parse CSV: {exc}"],
                duration_seconds=__import__("time").time() - start,
            )

        if not rows:
            return IngestionResult(
                errors=["CSV file is empty"],
                duration_seconds=__import__("time").time() - start,
            )

        tags = (tags or []) + ["csv", f"file:{path.name}"]
        all_memory_ids: list[str] = []

        if len(rows) > ROW_THRESHOLD:
            # Schema-only mode for large CSVs
            schema_desc = self._build_schema_description(rows, path.name)
            node = cortex.remember(
                content=schema_desc,
                memory_type=MemoryType.SEMANTIC,
                tags=tags + ["schema"],
                agent_id=agent_id,
                session_id=session_id,
            )
            if node:
                node.metadata.source_file = str(path.resolve())
                cortex.coordinator.update_node(node)
                all_memory_ids.append(node.id)
        else:
            # Row-per-memory mode
            for i, row in enumerate(rows):
                # Build a readable representation of the row
                parts = [f"{k}: {v}" for k, v in row.items() if v is not None]
                content = f"[CSV row {i + 1}]\n" + "\n".join(parts)
                node = cortex.remember(
                    content=content,
                    memory_type=MemoryType.SEMANTIC,
                    tags=tags,
                    agent_id=agent_id,
                    session_id=session_id,
                )
                if node:
                    node.metadata.source_file = str(path.resolve())
                    cortex.coordinator.update_node(node)
                    all_memory_ids.append(node.id)

        duration = __import__("time").time() - start
        return IngestionResult(
            memories_imported=len(all_memory_ids),
            memories_created=all_memory_ids,
            duration_seconds=duration,
        )

    @staticmethod
    def _build_schema_description(rows: list[dict], filename: str) -> str:
        """Build a schema description for large CSVs."""
        headers = list(rows[0].keys())
        lines = [
            f"[CSV Schema: {filename}]",
            f"Total rows: {len(rows)}",
            f"Columns: {len(headers)}",
            "",
            "Columns:",
        ]
        for h in headers:
            # Sample a few non-empty values
            samples = [
                str(r[h]) for r in rows[:10] if r.get(h)
            ]
            sample_str = f'  e.g. "{samples[0]}"' if samples else ""
            lines.append(f"  - {h}{sample_str}")
        lines.append("")
        lines.append("Sample rows (first 5):")
        for i, row in enumerate(rows[:5], 1):
            parts = [f"{k}={v}" for k, v in row.items() if v]
            lines.append(f"  Row {i}: {', '.join(parts[:5])}")
        return "\n".join(lines)
