"""StorageCoordinator — single gateway for all persistent writes.

Guarantees:
- SQLite is the source of truth (SOT) and is always written first.
- ChromaDB is treated as a rebuildable vector index.
- If ChromaDB write fails, the node ID is queued for backfill on next startup.
- All writes are idempotent where possible (e.g. duplicate content strengthens
  existing memory instead of creating a new node).

Replaces the scattered dual-write pattern:
    self._graph.add_node(node)
    self._chroma.add_node(coll, node)
"""

import json
import logging
import time
from typing import Optional

from cerebro.config import (
    COLLECTION_KNOWLEDGE,
    COLLECTION_MEMORIES,
    COLLECTION_SESSIONS,
    NEAR_DEDUP_THRESHOLD,
)
from cerebro.models.memory import MemoryNode
from cerebro.storage.chroma_store import ChromaStore
from cerebro.storage.graph_store import GraphStore
from cerebro.types import MemoryType

logger = logging.getLogger(__name__)


class StorageCoordinator:
    """Coordinates writes across SQLite (graph) and ChromaDB (vector)."""

    def __init__(self, graph: GraphStore, vector: ChromaStore):
        self._graph = graph
        self._vector = vector
        self._pending_backfill: set[str] = set()

    # ========================================================================
    # Collection routing
    # ========================================================================

    @staticmethod
    def collection_for_type(memory_type: MemoryType) -> str:
        """Determine which ChromaDB collection a memory belongs in.

        This static method was migrated from CerebroCortex to avoid circular
        imports in engines that need collection routing.
        """
        if memory_type in (MemoryType.SEMANTIC, MemoryType.SCHEMATIC):
            return COLLECTION_KNOWLEDGE
        elif memory_type == MemoryType.EPISODIC:
            return COLLECTION_SESSIONS
        else:  # PROCEDURAL, PROSPECTIVE, AFFECTIVE
            return COLLECTION_MEMORIES

    # ========================================================================
    # Node lifecycle
    # ========================================================================

    def store_node(
        self,
        node: MemoryNode,
        collection: Optional[str] = None,
        skip_chroma: bool = False,
    ) -> Optional[MemoryNode]:
        """Persist a MemoryNode to SQLite (SOT) and ChromaDB (index).

        Steps:
        1. Check exact duplicate via content hash. If found, strengthen the
           existing memory's access record and return it.
        2. Insert into SQLite.
        3. Insert into ChromaDB (if not skipped). On failure, queue for
           backfill and log a warning.

        Args:
            node: The memory node to store.
            collection: Target ChromaDB collection. Auto-detected from
                node.metadata.memory_type if omitted.
            skip_chroma: If True, only write to SQLite. Used during bulk
                imports where ChromaDB is populated in a second pass.

        Returns:
            The stored MemoryNode, or the existing node if deduplicated.
            None should not normally occur, but is typed for safety.
        """
        # -- Deduplication: exact hash match ---------------------------------
        existing_id = self._graph.find_duplicate_content(node.content)
        if existing_id:
            existing = self._graph.get_node(existing_id)
            if existing:
                from cerebro.activation.strength import record_access

                now = time.time()
                new_strength = record_access(existing.strength, now)
                self._graph.update_node_strength(existing_id, new_strength)
                logger.debug(f"Deduplicated content; strengthened {existing_id}")
                return existing

        # -- Near-duplicate detection (vector similarity) ----------------------
        coll = collection or self.collection_for_type(node.metadata.memory_type)
        if NEAR_DEDUP_THRESHOLD < 1.0:
            near_dup = self._find_near_duplicate(node, coll)
            if near_dup:
                # Merge tags, boost salience slightly
                merged_tags = list(set(near_dup.metadata.tags) | set(node.metadata.tags or []))
                self._graph.update_node_metadata(
                    near_dup.id,
                    tags=merged_tags,
                    salience=min(1.0, near_dup.metadata.salience + 0.02),
                )
                # Create a link noting the near-duplicate relationship
                try:
                    from cerebro.models.link import AssociativeLink
                    from cerebro.types import LinkType
                    link = AssociativeLink(
                        source_id=near_dup.id,
                        target_id=node.id,
                        link_type=LinkType.SUPPORTS,
                        weight=0.5,
                        evidence="Near-duplicate content detected at ingestion",
                    )
                    self._graph.add_link(link)
                except Exception:
                    pass  # Link creation is best-effort
                logger.debug(f"Near-duplicate detected; merged into {near_dup.id}")
                return near_dup

        # -- SQLite first (SOT) ----------------------------------------------
        self._graph.add_node(node)

        # -- ChromaDB second (index) -----------------------------------------
        if not skip_chroma:
            try:
                self._vector.add_node(coll, node)
            except Exception as exc:
                logger.warning(
                    f"ChromaDB write failed for {node.id} (queued for backfill): {exc}"
                )
                self._pending_backfill.add(node.id)

        return node

    def update_node(
        self,
        node: MemoryNode,
        collection: Optional[str] = None,
        old_collection: Optional[str] = None,
    ) -> bool:
        """Update an existing node across both backends.

        Handles collection migration when memory_type has changed.

        Args:
            node: The updated MemoryNode.
            collection: Current target ChromaDB collection.
            old_collection: Previous collection (if type changed, the old
                collection needs a delete).

        Returns:
            True if both backends were updated successfully.
        """
        coll = collection or self.collection_for_type(node.metadata.memory_type)

        # SQLite metadata updates are done via individual field updates in
        # GraphStore; the caller (cortex.update_memory) already handles that.
        # Here we only ensure ChromaDB is in sync.

        # Collection migration
        if old_collection is not None and old_collection != coll:
            try:
                self._vector.delete(old_collection, [node.id])
            except Exception as exc:
                logger.warning(f"Failed to delete {node.id} from old collection {old_collection}: {exc}")

        try:
            self._vector.update_node(coll, node)
            return True
        except Exception as exc:
            logger.warning(f"ChromaDB update failed for {node.id}: {exc}")
            self._pending_backfill.add(node.id)
            return False

    def delete_node(
        self,
        node_id: str,
        collection: Optional[str] = None,
        soft: bool = True,
    ) -> bool:
        """Delete a node from both backends.

        Args:
            node_id: Memory ID to delete.
            collection: ChromaDB collection containing the node.
            soft: If True (default), mark deleted_at in SQLite instead of hard delete.

        Returns:
            True if the node was found and removed/marked.
        """
        # If collection unknown, try to find it from SQLite
        if collection is None:
            row = self._graph.conn.execute(
                "SELECT memory_type FROM memory_nodes WHERE id = ?", (node_id,)
            ).fetchone()
            if row:
                try:
                    mt = MemoryType(row["memory_type"])
                    collection = self.collection_for_type(mt)
                except ValueError:
                    collection = COLLECTION_MEMORIES

        # SQLite first
        deleted = self._graph.delete_node(node_id, soft=soft)
        if not deleted:
            return False

        # ChromaDB second (best effort)
        if collection:
            try:
                self._vector.delete(collection, [node_id])
            except Exception as exc:
                logger.warning(f"ChromaDB delete failed for {node_id}: {exc}")

        return True

    # ========================================================================
    # Backfill
    # ========================================================================

    def backfill_pending(self) -> dict[str, int]:
        """Retry ChromaDB writes for queued node IDs.

        Typically called once during CerebroCortex.initialize().

        Returns:
            Dict with counts, e.g. {"backfilled": N, "failed": N}.
        """
        if not self._pending_backfill:
            return {"backfilled": 0, "failed": 0}

        backfilled = 0
        failed = 0
        still_pending: set[str] = set()

        for node_id in self._pending_backfill:
            node = self._graph.get_node(node_id)
            if node is None:
                # Node was deleted from SQLite; nothing to backfill
                continue

            coll = self.collection_for_type(node.metadata.memory_type)
            try:
                self._vector.add_node(coll, node)
                backfilled += 1
            except Exception as exc:
                logger.warning(f"Backfill failed for {node_id}: {exc}")
                failed += 1
                still_pending.add(node_id)

        self._pending_backfill = still_pending
        logger.info(f"Backfill complete: {backfilled} succeeded, {failed} failed, "
                    f"{len(still_pending)} still pending")
        return {"backfilled": backfilled, "failed": failed}

    def queue_for_backfill(self, node_id: str) -> None:
        """Explicitly queue a node ID for ChromaDB backfill."""
        self._pending_backfill.add(node_id)

    # ========================================================================
    # Bulk helpers
    # ========================================================================

    def bulk_store_nodes(
        self,
        nodes: list[MemoryNode],
        skip_chroma: bool = False,
    ) -> list[Optional[MemoryNode]]:
        """Store multiple nodes efficiently.

        Used by ingestion adapters. Deduplication is still checked per node.
        """
        return [
            self.store_node(node, skip_chroma=skip_chroma)
            for node in nodes
        ]

    # ========================================================================
    # Near-duplicate detection
    # ========================================================================

    def _find_near_duplicate(
        self, node: MemoryNode, collection: str
    ) -> Optional[MemoryNode]:
        """Search ChromaDB for a very similar existing memory.

        Returns the existing MemoryNode if similarity exceeds
        NEAR_DEDUP_THRESHOLD, else None.
        """
        try:
            results = self._vector.search(collection, node.content, n_results=1)
            if results and results[0].get("similarity") is not None:
                similarity = results[0]["similarity"]
                if similarity >= NEAR_DEDUP_THRESHOLD:
                    existing_id = results[0]["id"]
                    return self._graph.get_node(existing_id)
        except Exception as exc:
            logger.debug(f"Near-duplicate search failed: {exc}")
        return None
