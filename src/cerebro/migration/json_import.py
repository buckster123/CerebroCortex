"""Import memories from generic JSON files into CerebroCortex.

Supports two formats:

1. Simple list of dicts (minimal):
   [
     {"content": "Python is great", "tags": ["python"], "type": "semantic"},
     {"content": "Debug by reading logs first", "type": "procedural"},
     ...
   ]

2. Full CerebroCortex format (with nested metadata/strength):
   [
     {
       "content": "...",
       "metadata": {"memory_type": "semantic", "tags": [...], ...},
       "strength": {"stability": 1.0, ...}
     },
     ...
   ]
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from cerebro.cortex import CerebroCortex
from cerebro.models.memory import MemoryMetadata, MemoryNode, StrengthState
from cerebro.types import MemoryLayer, MemoryType, Visibility


@dataclass
class JSONImportReport:
    """Report from a JSON import operation."""
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


TYPE_SHORTCUTS: dict[str, MemoryType] = {
    "episodic": MemoryType.EPISODIC,
    "semantic": MemoryType.SEMANTIC,
    "procedural": MemoryType.PROCEDURAL,
    "affective": MemoryType.AFFECTIVE,
    "prospective": MemoryType.PROSPECTIVE,
    "schematic": MemoryType.SCHEMATIC,
}


class JSONImporter:
    """Import memories from generic JSON."""

    def __init__(self, cortex: CerebroCortex):
        self.cortex = cortex

    def import_file(self, path: Path) -> JSONImportReport:
        """Import from a JSON file."""
        with open(path) as f:
            data = json.load(f)
        return self.import_data(data)

    def import_data(self, data: list | dict) -> JSONImportReport:
        """Import from parsed JSON data.

        Accepts a list of records or a dict with a "memories" key.
        """
        report = JSONImportReport()
        start = time.time()

        if isinstance(data, dict):
            records = data.get("memories", data.get("records", []))
        elif isinstance(data, list):
            records = data
        else:
            report.errors.append("Expected list or dict with 'memories' key")
            return report

        for i, record in enumerate(records):
            if not isinstance(record, dict):
                report.errors.append(f"Record {i}: not a dict")
                report.memories_skipped += 1
                continue

            content = record.get("content", "").strip()
            if not content or len(content) < 3:
                report.memories_skipped += 1
                continue

            # Check dedup
            if self.cortex.graph.find_duplicate_content(content):
                report.memories_skipped += 1
                continue

            try:
                node = self._record_to_node(record)
                self.cortex.graph.add_node(node)
                report.memories_imported += 1
            except Exception as e:
                report.errors.append(f"Record {i}: {e}")
                report.memories_skipped += 1

        self.cortex.graph.resync_igraph()
        report.duration_seconds = time.time() - start
        return report

    def _record_to_node(self, record: dict) -> MemoryNode:
        """Convert a JSON record to a MemoryNode.

        Handles both simple flat format and full nested format.
        """
        content = record["content"]

        # Full format: nested metadata/strength objects
        if "metadata" in record and isinstance(record["metadata"], dict):
            raw_meta = record["metadata"]
            # Ensure enum values are properly handled
            if "memory_type" in raw_meta and isinstance(raw_meta["memory_type"], str):
                raw_meta["memory_type"] = MemoryType(raw_meta["memory_type"])
            if "visibility" in raw_meta and isinstance(raw_meta["visibility"], str):
                raw_meta["visibility"] = Visibility(raw_meta["visibility"])
            if "layer" in raw_meta and isinstance(raw_meta["layer"], str):
                raw_meta["layer"] = MemoryLayer(raw_meta["layer"])
            metadata = MemoryMetadata(**raw_meta)
        else:
            # Simple format: flat fields
            mem_type_str = record.get("type", record.get("memory_type", "semantic"))
            memory_type = TYPE_SHORTCUTS.get(mem_type_str, MemoryType.SEMANTIC)

            tags = record.get("tags", [])
            if isinstance(tags, str):
                tags = [t.strip() for t in tags.split(",")]

            metadata = MemoryMetadata(
                memory_type=memory_type,
                tags=tags,
                agent_id=record.get("agent_id", "CLAUDE"),
                salience=float(record.get("salience", 0.5)),
                source="import",
            )

        # Strength
        if "strength" in record and isinstance(record["strength"], dict):
            strength = StrengthState(**record["strength"])
        else:
            now = time.time()
            strength = StrengthState(
                access_timestamps=[now],
                access_count=1,
                last_computed_at=now,
            )

        metadata.source = "import"

        return MemoryNode(
            content=content,
            metadata=metadata,
            strength=strength,
        )
