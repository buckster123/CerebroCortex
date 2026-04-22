"""Embedder fingerprinting for ChromaDB collections.

Stores a fingerprint of the embedding model that built each collection,
and warns on first access if the currently-configured embedder doesn't
match. Catches the silent-mixed-embedder-vectors class of bug that
bit the v0.2.0 swap from MiniLM to BGE.

Fingerprint shape (stored in collection metadata):
  cc_embedder_name:    str   canonical name returned by EmbeddingFunction.name()
  cc_embedder_dim:     int   embedding dimensionality
  cc_embedder_schema:  int   bump if we change this format

The ChromaDB metadata dict is flat string/int/float/bool only — we fit
that constraint by namespacing keys with cc_ prefix.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
_MISMATCH_WARNED: set[str] = set()  # per-process: don't spam on every call


def _probe_dim(embedding_fn) -> Optional[int]:
    """Best-effort: probe the embedder for its output dimension.

    Tries (in order): .dimension property, one-shot embed of 'x'.
    Returns None if both fail.
    """
    dim = getattr(embedding_fn, "dimension", None)
    if isinstance(dim, int) and dim > 0:
        return dim
    try:
        vecs = embedding_fn(["x"])
        if vecs and hasattr(vecs[0], "shape"):
            return int(vecs[0].shape[0])
    except Exception as e:
        logger.debug(f"embedder dim probe failed: {e}")
    return None


def fingerprint_for(embedding_fn) -> dict[str, Any]:
    """Build a fingerprint dict for a live EmbeddingFunction."""
    name = getattr(embedding_fn, "name", None)
    name_str = name() if callable(name) else (name or "unknown")
    dim = _probe_dim(embedding_fn) or 0
    return {
        "cc_embedder_name": name_str,
        "cc_embedder_dim": dim,
        "cc_embedder_schema": SCHEMA_VERSION,
    }


def extract_fingerprint(collection_metadata: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """Pull an embedder fingerprint out of collection metadata, if present."""
    if not collection_metadata:
        return None
    if "cc_embedder_name" not in collection_metadata:
        return None
    return {
        "cc_embedder_name": collection_metadata.get("cc_embedder_name", "unknown"),
        "cc_embedder_dim": int(collection_metadata.get("cc_embedder_dim", 0)),
        "cc_embedder_schema": int(collection_metadata.get("cc_embedder_schema", 0)),
    }


def merge_into_metadata(metadata: dict[str, Any], fingerprint: dict[str, Any]) -> dict[str, Any]:
    """Return a new metadata dict with fingerprint keys layered on top.

    Note: intended for collection *creation*. For .modify() on existing collections,
    see strip_immutable_for_modify() — ChromaDB 1.x rejects hnsw:* keys on modify.
    """
    out = dict(metadata or {})
    out.update(fingerprint)
    return out


def strip_immutable_for_modify(metadata: dict[str, Any]) -> dict[str, Any]:
    """Filter metadata to only keys ChromaDB accepts in collection.modify().

    ChromaDB 1.x raises ValueError on .modify() if metadata contains hnsw:* keys
    (distance function / construction params are immutable once the collection
    exists). Safe set = anything that is not hnsw:-prefixed.
    """
    return {k: v for k, v in (metadata or {}).items() if not k.startswith("hnsw:")}


def check_match(collection_name: str,
                stored: Optional[dict[str, Any]],
                current: dict[str, Any]) -> bool:
    """Compare a stored fingerprint against the current embedder.

    Returns True on match (or if no stored fingerprint exists — legacy collection).
    On mismatch, logs a WARNING exactly once per (collection, mismatch-pair) per process.

    Does NOT raise — a warning is the right primitive here. Forcing hard-fail would
    break all dream cycles on every embedder swap; a warning with clear remediation
    steps gives the operator the choice.
    """
    if stored is None:
        # Legacy collection with no fingerprint — not an error, just unknown provenance.
        # Consider calling backfill_fingerprint after initial boot to upgrade.
        return True

    if stored.get("cc_embedder_schema", 0) > SCHEMA_VERSION:
        logger.warning(
            f"collection {collection_name!r} was built with a newer fingerprint schema "
            f"(v{stored['cc_embedder_schema']} > v{SCHEMA_VERSION}); upgrade CerebroCortex"
        )
        return False

    stored_name = stored.get("cc_embedder_name", "")
    stored_dim = stored.get("cc_embedder_dim", 0)
    current_name = current.get("cc_embedder_name", "")
    current_dim = current.get("cc_embedder_dim", 0)

    if stored_name == current_name and stored_dim == current_dim:
        return True

    key = f"{collection_name}|{stored_name}|{current_name}"
    if key not in _MISMATCH_WARNED:
        _MISMATCH_WARNED.add(key)
        msg_lines = [
            f"EMBEDDER FINGERPRINT MISMATCH on collection {collection_name!r}:",
            f"  stored:  {stored_name}  (dim={stored_dim})",
            f"  current: {current_name}  (dim={current_dim})",
            "",
            "Retrieval relevance on this collection will be DEGRADED or meaningless.",
            "Options:",
            "  1) Re-embed the corpus with the current embedder:",
            "       PYTHONPATH=src ./venv/bin/python scripts/reembed_collections.py",
            "  2) Revert embedder config to the one that built this collection.",
            "  3) If you intended this swap, run `cerebro doctor audit --fix-fingerprint`",
            "     to update the fingerprint (NOT the vectors) — only safe if you re-embedded",
            "     out-of-band or accept stale vectors.",
        ]
        logger.warning("\n".join(msg_lines))
    return False
