"""Ensure CerebroCortex data directories exist on first run."""

from cerebro.config import DATA_DIR, CHROMA_DIR, EXPORT_DIR


def ensure_data_dirs() -> None:
    """Create data directories if they don't exist."""
    for d in (DATA_DIR, CHROMA_DIR, EXPORT_DIR):
        d.mkdir(parents=True, exist_ok=True)
