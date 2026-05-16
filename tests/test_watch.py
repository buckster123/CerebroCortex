"""Tests for cerebro.watch file watcher module."""

import json
import tempfile
import time
from pathlib import Path

import pytest

from cerebro.watch import FileWatcher
from cerebro.watch.watcher import _file_fingerprint, _load_state, _save_state


def test_file_fingerprint(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("hello")
    fp = _file_fingerprint(f)
    assert fp != ""
    assert ":" in fp


def test_state_persistence(tmp_path, monkeypatch):
    from cerebro.watch import watcher
    monkeypatch.setattr(watcher, "_STATE_FILE", tmp_path / "watch_state.json")
    state = {"a": "1", "b": "2"}
    _save_state(state)
    loaded = _load_state()
    assert loaded == state


def test_file_watcher_lifecycle(multi_agent_cortex):
    """Test starting and stopping the watcher."""
    ctx = multi_agent_cortex["cortex"]
    watcher = FileWatcher(ctx)
    assert not watcher.is_running()
    watcher.start()
    assert watcher.is_running()
    watcher.stop()
    assert not watcher.is_running()


def test_file_watcher_add_directory(multi_agent_cortex, tmp_path):
    ctx = multi_agent_cortex["cortex"]
    watcher = FileWatcher(ctx)
    watcher.start()
    d = tmp_path / "watch_dir"
    d.mkdir()
    assert watcher.add_directory(d)
    assert watcher.watched_count == 1
    watcher.stop()


def test_file_watcher_add_nonexistent(multi_agent_cortex):
    ctx = multi_agent_cortex["cortex"]
    watcher = FileWatcher(ctx)
    watcher.start()
    assert not watcher.add_directory("/nonexistent/path/12345")
    watcher.stop()


def test_file_watcher_ingest_txt(multi_agent_cortex, tmp_path, monkeypatch):
    """End-to-end: drop a file and verify it gets ingested."""
    ctx = multi_agent_cortex["cortex"]
    from cerebro.watch import watcher as watcher_mod
    monkeypatch.setattr(watcher_mod, "_STATE_FILE", tmp_path / "watch_state.json")

    watch_dir = tmp_path / "inbox"
    watch_dir.mkdir()

    fw = FileWatcher(ctx, agent_id="TEST", tags=["watcher-test"], delay_seconds=0.2)
    fw.start()
    fw.add_directory(watch_dir)

    # Drop a file
    test_file = watch_dir / "note.txt"
    test_file.write_text("This is a test memory from the watcher.")

    # Wait for ingestion
    time.sleep(0.8)

    fw.stop()

    # Verify memory was created
    results = ctx.recall("test memory from the watcher", top_k=5)
    contents = [n.content for n, _ in results]
    assert any("test memory from the watcher" in c for c in contents)
