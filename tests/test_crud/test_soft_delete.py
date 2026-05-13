"""Tests for soft-delete, versioning, and trash can (Phase C)."""

import tempfile
from pathlib import Path

import pytest

from cerebro.cortex import CerebroCortex


@pytest.fixture
def crud_cortex():
    with tempfile.TemporaryDirectory() as d:
        ctx = CerebroCortex(
            db_path=Path(d) / "test.db",
            chroma_dir=Path(d) / "chroma",
        )
        ctx.initialize()
        yield ctx
        ctx.close()


class TestSoftDelete:
    def test_delete_soft_by_default(self, crud_cortex):
        node = crud_cortex.remember("This is a detailed memory about soft deletion testing in CerebroCortex")
        assert node is not None
        ok = crud_cortex.delete_memory(node.id)
        assert ok is True

        # Active query should not find it
        assert crud_cortex.get_memory(node.id) is None

        # Deleted query should find it
        deleted = crud_cortex.list_deleted()
        assert len(deleted) == 1
        assert deleted[0].id == node.id
        assert deleted[0].metadata.deleted_at is not None

    def test_delete_hard(self, crud_cortex):
        node = crud_cortex.remember("Hard deletion test with unique content to avoid gating")
        ok = crud_cortex.delete_memory(node.id, hard=True)
        assert ok is True
        assert crud_cortex.get_memory(node.id) is None
        assert crud_cortex.list_deleted() == []

    def test_restore_memory(self, crud_cortex):
        node = crud_cortex.remember("Memory to be restored after soft deletion")
        crud_cortex.delete_memory(node.id)
        assert crud_cortex.get_memory(node.id) is None

        ok = crud_cortex.restore_memory(node.id)
        assert ok is True
        restored = crud_cortex.get_memory(node.id)
        assert restored is not None
        assert restored.metadata.deleted_at is None

    def test_purge_memory(self, crud_cortex):
        node = crud_cortex.remember("Memory to be purged permanently from the system")
        crud_cortex.delete_memory(node.id)  # soft delete first
        ok = crud_cortex.purge_memory(node.id)
        assert ok is True
        assert crud_cortex.list_deleted() == []
        assert crud_cortex.get_memory(node.id, include_deleted=True) is None

    def test_purge_all_deleted(self, crud_cortex):
        n1 = crud_cortex.remember("Old trash memory about pineapple recipes for purge testing")
        n2 = crud_cortex.remember("Old trash memory about quantum computing for purge testing")
        assert n1 is not None and n2 is not None
        assert n1.id != n2.id
        crud_cortex.delete_memory(n1.id)
        crud_cortex.delete_memory(n2.id)
        count = crud_cortex.purge_all_deleted(older_than_days=-1)
        assert count == 2
        assert crud_cortex.list_deleted() == []


class TestVersioning:
    def test_update_creates_version(self, crud_cortex):
        node = crud_cortex.remember("Versioned content memory for testing snapshot functionality")
        assert node is not None
        versions_before = crud_cortex.get_memory_versions(node.id)
        assert len(versions_before) == 0

        crud_cortex.update_memory(node.id, content="Updated content with different text for versioning")
        versions_after = crud_cortex.get_memory_versions(node.id)
        assert len(versions_after) == 1
        assert versions_after[0]["content"] == "Versioned content memory for testing snapshot functionality"

    def test_restore_version(self, crud_cortex):
        node = crud_cortex.remember("Original content for version restore testing")
        assert node is not None
        crud_cortex.update_memory(node.id, content="Changed content for restore testing")
        versions = crud_cortex.get_memory_versions(node.id)
        vid = versions[0]["id"]

        restored = crud_cortex.restore_version(vid)
        assert restored is not None
        assert restored.content == "Original content for version restore testing"

    def test_update_without_content_change_no_version(self, crud_cortex):
        node = crud_cortex.remember("No change memory for testing metadata-only updates")
        assert node is not None
        crud_cortex.update_memory(node.id, salience=0.9)
        versions = crud_cortex.get_memory_versions(node.id)
        assert len(versions) == 0


class TestBulkOperations:
    def test_bulk_delete(self, crud_cortex):
        n1 = crud_cortex.remember("Bulk delete test memory about Jupiter's moons and atmospheric composition")
        n2 = crud_cortex.remember("Bulk delete test memory about deep sea volcanic vents and extremophile bacteria")
        assert n1 is not None and n2 is not None
        assert n1.id != n2.id
        deleted = crud_cortex.bulk_delete([n1.id, n2.id])
        assert len(deleted) == 2
        assert crud_cortex.get_memory(n1.id) is None
        assert crud_cortex.get_memory(n2.id) is None

    def test_bulk_update_visibility(self, crud_cortex):
        from cerebro.types import Visibility

        n1 = crud_cortex.remember("Visibility test memory one")
        n2 = crud_cortex.remember("Visibility test memory two")
        assert n1 is not None and n2 is not None
        updated = crud_cortex.bulk_update_visibility(
            [n1.id, n2.id], Visibility.PRIVATE
        )
        assert len(updated) == 2
        assert crud_cortex.get_memory(n1.id).metadata.visibility == Visibility.PRIVATE

    def test_export_json(self, crud_cortex):
        n1 = crud_cortex.remember("Export test memory for JSON serialization")
        assert n1 is not None
        output = crud_cortex.export_memories([n1.id], fmt="json")
        assert "Export test memory" in output
        assert n1.id in output


class TestThreadManagement:
    def test_list_threads_empty(self, crud_cortex):
        threads = crud_cortex.list_threads()
        assert threads == []

    def test_thread_crud(self, crud_cortex):
        n1 = crud_cortex.remember("thread msg 1", session_id="sess-1")
        # Manually set conversation_thread via raw SQL since remember() doesn't expose it
        crud_cortex._graph.conn.execute(
            "UPDATE memory_nodes SET conversation_thread = ? WHERE id = ?",
            ("thread-42", n1.id),
        )
        crud_cortex._graph.conn.commit()

        threads = crud_cortex.list_threads()
        assert len(threads) == 1
        assert threads[0]["thread_id"] == "thread-42"

        mems = crud_cortex.get_thread_memories("thread-42")
        assert len(mems) == 1

        count = crud_cortex.prune_thread("thread-42")
        assert count == 1
        assert crud_cortex.get_memory(n1.id) is None
