"""Tag management engine for CerebroCortex.

Handles tag CRUD across the memory graph, including rename, merge, delete,
and counting. All operations sync back to ChromaDB metadata.
"""

import json
import logging
from typing import Optional

from cerebro.storage.graph_store import GraphStore
from cerebro.storage.coordinator import StorageCoordinator

logger = logging.getLogger(__name__)


class TagManager:
    """Manage tags across all memories."""

    def __init__(self, graph: GraphStore):
        self._graph = graph

    def list_tags(self, agent_id: Optional[str] = None) -> dict[str, int]:
        """Return {tag: count} across all active memories."""
        if agent_id:
            rows = self._graph.conn.execute(
                "SELECT tags_json FROM memory_nodes WHERE deleted_at IS NULL AND agent_id = ?",
                (agent_id,),
            ).fetchall()
        else:
            rows = self._graph.conn.execute(
                "SELECT tags_json FROM memory_nodes WHERE deleted_at IS NULL"
            ).fetchall()

        counts: dict[str, int] = {}
        for r in rows:
            tags = json.loads(r["tags_json"])
            for t in tags:
                counts[t] = counts.get(t, 0) + 1
        return counts

    def rename_tag(self, old: str, new: str, agent_id: Optional[str] = None) -> int:
        """Rename a tag everywhere. Returns number of memories updated."""
        where = "deleted_at IS NULL AND tags_json LIKE ?"
        params: list = [f'%"{old}"%']
        if agent_id:
            where += " AND agent_id = ?"
            params.append(agent_id)

        rows = self._graph.conn.execute(
            f"SELECT id, tags_json FROM memory_nodes WHERE {where}", params
        ).fetchall()

        updated = 0
        for r in rows:
            tags = json.loads(r["tags_json"])
            if old in tags:
                tags[tags.index(old)] = new
                self._graph.conn.execute(
                    "UPDATE memory_nodes SET tags_json = ? WHERE id = ?",
                    (json.dumps(tags), r["id"]),
                )
                updated += 1

        if updated:
            self._graph.conn.commit()
            logger.info(f"Renamed tag '{old}' -> '{new}' in {updated} memories")
        return updated

    def merge_tags(self, source_tags: list[str], target_tag: str, agent_id: Optional[str] = None) -> int:
        """Merge multiple tags into one. Returns number of memories updated."""
        where = "deleted_at IS NULL AND (" + " OR ".join(["tags_json LIKE ?"] * len(source_tags)) + ")"
        params = [f'%"{t}"%' for t in source_tags]
        if agent_id:
            where += " AND agent_id = ?"
            params.append(agent_id)

        rows = self._graph.conn.execute(
            f"SELECT id, tags_json FROM memory_nodes WHERE {where}", params
        ).fetchall()

        updated = 0
        for r in rows:
            tags = json.loads(r["tags_json"])
            changed = False
            for st in source_tags:
                if st in tags and st != target_tag:
                    tags.remove(st)
                    changed = True
            if changed and target_tag not in tags:
                tags.append(target_tag)
            if changed:
                self._graph.conn.execute(
                    "UPDATE memory_nodes SET tags_json = ? WHERE id = ?",
                    (json.dumps(tags), r["id"]),
                )
                updated += 1

        if updated:
            self._graph.conn.commit()
            logger.info(f"Merged tags {source_tags} -> '{target_tag}' in {updated} memories")
        return updated

    def delete_tag(self, tag: str, agent_id: Optional[str] = None) -> int:
        """Remove a tag from all memories. Returns number of memories updated."""
        where = "deleted_at IS NULL AND tags_json LIKE ?"
        params: list = [f'%"{tag}"%']
        if agent_id:
            where += " AND agent_id = ?"
            params.append(agent_id)

        rows = self._graph.conn.execute(
            f"SELECT id, tags_json FROM memory_nodes WHERE {where}", params
        ).fetchall()

        updated = 0
        for r in rows:
            tags = json.loads(r["tags_json"])
            if tag in tags:
                tags.remove(tag)
                self._graph.conn.execute(
                    "UPDATE memory_nodes SET tags_json = ? WHERE id = ?",
                    (json.dumps(tags), r["id"]),
                )
                updated += 1

        if updated:
            self._graph.conn.commit()
            logger.info(f"Deleted tag '{tag}' from {updated} memories")
        return updated
