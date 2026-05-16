"""Base classes for ingestion adapters."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from cerebro.cortex import CerebroCortex


@dataclass
class IngestionResult:
    """Report from an ingestion operation."""

    memories_imported: int = 0
    memories_skipped: int = 0
    memories_created: list[str] = field(default_factory=list)
    attachments_created: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "memories_imported": self.memories_imported,
            "memories_skipped": self.memories_skipped,
            "memories_created": self.memories_created,
            "attachments_created": self.attachments_created,
            "errors": self.errors[:20],
            "total_errors": len(self.errors),
            "duration_seconds": round(self.duration_seconds, 2),
        }


class IngestionAdapter(ABC):
    """Base class for file/memory ingestion.

    Implementations must NOT write directly to storage. Instead, they yield
    candidate content/metadata and let the pipeline handle persistence via
    :py:meth:`CerebroCortex.remember` or :py:meth:`CerebroCortex.bulk_remember`.
    """

    @abstractmethod
    def can_ingest(self, path: Path) -> bool:
        """Return True if this adapter handles the given file."""
        ...

    @abstractmethod
    def ingest(
        self,
        path: Path,
        *,
        cortex: "CerebroCortex",
        tags: Optional[list[str]] = None,
        agent_id: str = "CLAUDE",
        session_id: Optional[str] = None,
    ) -> IngestionResult:
        """Ingest a file and return a report.

        Must route persistence through ``cortex.remember()`` or
        ``cortex.bulk_remember()``.
        """
        ...
