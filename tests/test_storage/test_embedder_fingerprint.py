"""Tests for embedder fingerprinting."""
from __future__ import annotations

import numpy as np

from cerebro.storage.embedder_fingerprint import (
    SCHEMA_VERSION,
    check_match,
    extract_fingerprint,
    fingerprint_for,
    merge_into_metadata,
    strip_immutable_for_modify,
)


class _FakeEF:
    """Minimal fake that looks like an EmbeddingFunction."""
    def __init__(self, name: str, dim: int):
        self._name = name
        self._dim = dim

    def name(self) -> str:  # matches the callable-or-attr shape embeddings.py uses
        return self._name

    @property
    def dimension(self) -> int:
        return self._dim

    def __call__(self, texts):
        return [np.zeros(self._dim, dtype=np.float32) for _ in texts]


# --------------- fingerprint_for ---------------

def test_fingerprint_for_basic():
    ef = _FakeEF("sbert:all-MiniLM-L6-v2", 384)
    fp = fingerprint_for(ef)
    assert fp["cc_embedder_name"] == "sbert:all-MiniLM-L6-v2"
    assert fp["cc_embedder_dim"] == 384
    assert fp["cc_embedder_schema"] == SCHEMA_VERSION


def test_fingerprint_for_falls_back_to_probe_when_no_dimension():
    class NoDim:
        def name(self): return "weird"
        def __call__(self, texts): return [np.zeros(256, dtype=np.float32) for _ in texts]
    fp = fingerprint_for(NoDim())
    assert fp["cc_embedder_dim"] == 256


def test_fingerprint_for_string_name_attr():
    class StringName:
        name = "not-callable"
        dimension = 128
    fp = fingerprint_for(StringName())
    assert fp["cc_embedder_name"] == "not-callable"
    assert fp["cc_embedder_dim"] == 128


# --------------- extract_fingerprint ---------------

def test_extract_none_when_missing():
    assert extract_fingerprint(None) is None
    assert extract_fingerprint({}) is None
    assert extract_fingerprint({"hnsw:space": "cosine"}) is None  # no cc_* keys


def test_extract_returns_canonical_shape():
    fp = extract_fingerprint({
        "hnsw:space": "cosine",
        "cc_embedder_name": "sbert:foo",
        "cc_embedder_dim": 384,
        "cc_embedder_schema": 1,
    })
    assert fp == {"cc_embedder_name": "sbert:foo", "cc_embedder_dim": 384, "cc_embedder_schema": 1}


def test_extract_coerces_numeric_strings():
    # ChromaDB stores metadata as Any; tolerate string-int degenerate cases
    fp = extract_fingerprint({
        "cc_embedder_name": "foo",
        "cc_embedder_dim": "384",
        "cc_embedder_schema": "1",
    })
    assert fp["cc_embedder_dim"] == 384
    assert fp["cc_embedder_schema"] == 1


# --------------- merge_into_metadata + strip_immutable_for_modify ---------------

def test_merge_preserves_hnsw_and_adds_fingerprint():
    base = {"hnsw:space": "cosine"}
    fp = {"cc_embedder_name": "x", "cc_embedder_dim": 1, "cc_embedder_schema": 1}
    merged = merge_into_metadata(base, fp)
    assert merged["hnsw:space"] == "cosine"
    assert merged["cc_embedder_name"] == "x"


def test_merge_is_not_mutating():
    base = {"hnsw:space": "cosine"}
    fp = {"cc_embedder_name": "x"}
    merged = merge_into_metadata(base, fp)
    assert "cc_embedder_name" not in base  # didn't mutate caller


def test_strip_removes_only_hnsw():
    meta = {
        "hnsw:space": "cosine",
        "hnsw:construction_ef": 100,
        "cc_embedder_name": "keep",
        "user_custom_tag": "keep",
    }
    stripped = strip_immutable_for_modify(meta)
    assert "hnsw:space" not in stripped
    assert "hnsw:construction_ef" not in stripped
    assert stripped["cc_embedder_name"] == "keep"
    assert stripped["user_custom_tag"] == "keep"


def test_strip_handles_none():
    assert strip_immutable_for_modify(None) == {}


# --------------- check_match ---------------

def test_check_match_legacy_collection_returns_true():
    # No stored fingerprint (pre-v0.3.0 collection) -> "don't know, assume ok"
    current = {"cc_embedder_name": "x", "cc_embedder_dim": 1, "cc_embedder_schema": 1}
    assert check_match("coll", None, current) is True


def test_check_match_returns_true_on_exact_match():
    fp = {"cc_embedder_name": "x", "cc_embedder_dim": 384, "cc_embedder_schema": 1}
    assert check_match("coll", fp, fp) is True


def test_check_match_returns_false_on_name_mismatch(caplog):
    stored = {"cc_embedder_name": "sbert:MiniLM", "cc_embedder_dim": 384, "cc_embedder_schema": 1}
    current = {"cc_embedder_name": "openai_compat:bge-int8", "cc_embedder_dim": 384, "cc_embedder_schema": 1}
    with caplog.at_level("WARNING"):
        assert check_match("test_name_mismatch_coll", stored, current) is False
    joined = " ".join(r.message for r in caplog.records)
    assert "EMBEDDER FINGERPRINT MISMATCH" in joined
    assert "sbert:MiniLM" in joined
    assert "openai_compat:bge-int8" in joined
    assert "reembed_collections.py" in joined  # actionable remediation


def test_check_match_returns_false_on_dim_mismatch():
    stored = {"cc_embedder_name": "same", "cc_embedder_dim": 384, "cc_embedder_schema": 1}
    current = {"cc_embedder_name": "same", "cc_embedder_dim": 768, "cc_embedder_schema": 1}
    assert check_match("dim_mismatch_coll", stored, current) is False


def test_check_match_warns_only_once_per_pair(caplog):
    stored = {"cc_embedder_name": "a", "cc_embedder_dim": 1, "cc_embedder_schema": 1}
    current = {"cc_embedder_name": "b", "cc_embedder_dim": 1, "cc_embedder_schema": 1}
    with caplog.at_level("WARNING"):
        check_match("once_coll_42", stored, current)
        check_match("once_coll_42", stored, current)
        check_match("once_coll_42", stored, current)
    # Expect exactly one MISMATCH warning for this collection+pair in the capture
    mismatches = [r for r in caplog.records if "EMBEDDER FINGERPRINT MISMATCH on collection 'once_coll_42'" in r.message]
    assert len(mismatches) == 1


def test_check_match_detects_newer_schema(caplog):
    stored = {"cc_embedder_name": "x", "cc_embedder_dim": 1, "cc_embedder_schema": SCHEMA_VERSION + 99}
    current = {"cc_embedder_name": "x", "cc_embedder_dim": 1, "cc_embedder_schema": SCHEMA_VERSION}
    with caplog.at_level("WARNING"):
        assert check_match("future_coll", stored, current) is False
    assert any("newer fingerprint schema" in r.message for r in caplog.records)
