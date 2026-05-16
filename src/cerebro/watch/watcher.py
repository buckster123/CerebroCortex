"""FileSystemEventHandler + Observer for auto-ingestion.

Uses watchdog to monitor directories and routes new files through
:cerebro.ingestion.IngestionPipeline:.

State is persisted in ``$DATA_DIR/.cerebro-watch-state.json`` to avoid
re-ingesting unchanged files across restarts.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Optional

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileCreatedEvent, FileMovedEvent

from cerebro.config import DATA_DIR
from cerebro.ingestion import IngestionPipeline

logger = logging.getLogger("cerebro-watch")

_STATE_FILE = DATA_DIR / ".cerebro-watch-state.json"

# File extensions that the ingestion pipeline can handle
_DEFAULT_PATTERNS = [
    "*.md", "*.txt", "*.json",
    "*.pdf", "*.html", "*.csv",
    "*.png", "*.jpg", "*.jpeg", "*.webp",
    "*.py", "*.js", "*.ts", "*.go", "*.rs", "*.java", "*.rb", "*.sh",
]


def _load_state() -> dict:
    if not _STATE_FILE.exists():
        return {}
    try:
        return json.loads(_STATE_FILE.read_text())
    except Exception:
        return {}


def _save_state(state: dict) -> None:
    _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    _STATE_FILE.write_text(json.dumps(state, indent=2) + "\n")


def _file_fingerprint(path: Path) -> str:
    """Return a stable fingerprint for a file (mtime + size)."""
    try:
        stat = path.stat()
        return f"{stat.st_mtime:.6f}:{stat.st_size}"
    except OSError:
        return ""


class _IngestionHandler(FileSystemEventHandler):
    """Handles filesystem events by queueing files for ingestion."""

    def __init__(
        self,
        cortex,
        agent_id: str = "CLAUDE",
        tags: Optional[list[str]] = None,
        patterns: Optional[list[str]] = None,
        delay_seconds: float = 1.0,
    ):
        self.cortex = cortex
        self.agent_id = agent_id
        self.tags = tags or ["auto-ingested"]
        self.patterns = patterns or _DEFAULT_PATTERNS
        self.delay_seconds = delay_seconds
        self.pipeline = IngestionPipeline(cortex)
        self._state = _load_state()
        self._lock = threading.Lock()
        self._pending: dict[str, threading.Timer] = {}

    def _matches_pattern(self, path: Path) -> bool:
        name = path.name.lower()
        for pat in self.patterns:
            # Simple glob-style matching
            suffix = pat.lstrip("*")
            if name.endswith(suffix.lower()):
                return True
        return False

    def _already_processed(self, path: Path) -> bool:
        key = str(path.resolve())
        fp = _file_fingerprint(path)
        if not fp:
            return False
        with self._lock:
            return self._state.get(key) == fp

    def _mark_processed(self, path: Path) -> None:
        key = str(path.resolve())
        fp = _file_fingerprint(path)
        with self._lock:
            self._state[key] = fp
            _save_state(self._state)

    def _ingest(self, path: Path) -> None:
        try:
            if not path.exists():
                return
            if self._already_processed(path):
                logger.debug("Skipping already-ingested file: %s", path)
                return
            report = self.pipeline.ingest_file(
                path, tags=self.tags, agent_id=self.agent_id
            )
            if report.errors:
                logger.warning("Ingestion completed with errors for %s: %s", path, report.errors)
            else:
                logger.info(
                    "Ingested %s → %s memories in %.2fs",
                    path.name,
                    report.memories_created,
                    report.duration_seconds,
                )
            self._mark_processed(path)
        except Exception as exc:
            logger.error("Failed to ingest %s: %s", path, exc)

    def _schedule(self, path: Path) -> None:
        key = str(path.resolve())
        # Cancel any pending timer for this path
        with self._lock:
            old_timer = self._pending.pop(key, None)
        if old_timer is not None:
            old_timer.cancel()

        timer = threading.Timer(self.delay_seconds, self._ingest, args=[path])
        with self._lock:
            self._pending[key] = timer
        timer.start()

    def on_created(self, event: FileCreatedEvent) -> None:
        if event.is_directory:
            return
        path = Path(event.src_path)
        if not self._matches_pattern(path):
            return
        self._schedule(path)

    def on_moved(self, event: FileMovedEvent) -> None:
        if event.is_directory:
            return
        path = Path(event.dest_path)
        if not self._matches_pattern(path):
            return
        self._schedule(path)

    def close(self) -> None:
        with self._lock:
            for timer in self._pending.values():
                timer.cancel()
            self._pending.clear()


class FileWatcher:
    """High-level wrapper around watchdog Observer.

    Example::

        watcher = FileWatcher(cortex)
        watcher.add_directory("~/Dropbox/CerebroInbox")
        watcher.start()
        # ... run forever or until stop() ...
        watcher.stop()
    """

    def __init__(
        self,
        cortex,
        agent_id: Optional[str] = None,
        tags: Optional[list[str]] = None,
        patterns: Optional[list[str]] = None,
        delay_seconds: float = 1.0,
    ):
        self.cortex = cortex
        self.agent_id = agent_id or os.environ.get("CEREBRO_AGENT_ID", "CLAUDE")
        self.tags = tags
        self.patterns = patterns
        self.delay_seconds = delay_seconds
        self._observer: Optional[Observer] = None
        self._handler: Optional[_IngestionHandler] = None
        self._watches: list = []
        self._lock = threading.Lock()
        self._running = False

    def add_directory(self, path: str | Path) -> bool:
        """Add a directory to watch. Returns False if directory doesn't exist."""
        p = Path(path).expanduser().resolve()
        if not p.exists():
            logger.warning("Watch directory does not exist: %s", p)
            return False
        if not p.is_dir():
            logger.warning("Not a directory: %s", p)
            return False

        with self._lock:
            if self._observer is None:
                logger.warning("Watcher not started yet; directory queued for next start")
                return True
            watch = self._observer.schedule(self._handler, str(p), recursive=True)
            self._watches.append(watch)
        logger.info("Watching directory: %s", p)
        return True

    def start(self) -> None:
        """Start the observer thread."""
        with self._lock:
            if self._running:
                return
            self._handler = _IngestionHandler(
                self.cortex,
                agent_id=self.agent_id,
                tags=self.tags,
                patterns=self.patterns,
                delay_seconds=self.delay_seconds,
            )
            self._observer = Observer()
            self._observer.start()
            self._running = True
        logger.info("File watcher started (agent=%s)", self.agent_id)

    def stop(self) -> None:
        """Stop the observer and clean up."""
        with self._lock:
            if not self._running:
                return
            self._running = False
            if self._observer is not None:
                self._observer.stop()
                self._observer.join()
                self._observer = None
            if self._handler is not None:
                self._handler.close()
                self._handler = None
            self._watches.clear()
        logger.info("File watcher stopped")

    def is_running(self) -> bool:
        with self._lock:
            return self._running

    @property
    def watched_count(self) -> int:
        with self._lock:
            return len(self._watches)
