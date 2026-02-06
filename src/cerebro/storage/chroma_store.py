"""ChromaDB vector store for CerebroCortex.

Handles semantic vector search over memory content. Works alongside
the GraphStore (SQLite+igraph) which handles the associative network.

Key differences from Neo-Cortex:
- 3 collections instead of 6 (type is a metadata filter)
- Richer metadata (valence, salience, memory_type)
- Document IDs match GraphStore node IDs for cross-referencing
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from cerebro.config import ALL_COLLECTIONS, CHROMA_DIR, EMBEDDING_DIM
from cerebro.models.memory import MemoryMetadata, MemoryNode
from cerebro.storage.base import VectorStore
from cerebro.storage.embeddings import get_embedding_function

logger = logging.getLogger(__name__)


class ChromaStore(VectorStore):
    """ChromaDB backend for semantic vector search."""

    def __init__(self, persist_path: Optional[Path] = None):
        self._persist_path = persist_path or CHROMA_DIR
        self._client = None
        self._embedding_fn = None
        self._collections: dict[str, Any] = {}

    def _get_client(self):
        if self._client is None:
            import chromadb
            from chromadb.config import Settings

            self._persist_path.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=str(self._persist_path),
                settings=Settings(anonymized_telemetry=False),
            )
            logger.info(f"ChromaDB initialized at {self._persist_path}")
        return self._client

    def _get_embedding_fn(self):
        if self._embedding_fn is None:
            self._embedding_fn = get_embedding_function("auto")
        return self._embedding_fn

    def _get_collection(self, name: str):
        if name not in self._collections:
            client = self._get_client()
            embedding_fn = self._get_embedding_fn()
            self._collections[name] = client.get_or_create_collection(
                name=name,
                embedding_function=embedding_fn,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collections[name]

    def initialize(self) -> None:
        """Initialize all collections."""
        for coll_name in ALL_COLLECTIONS:
            self._get_collection(coll_name)
        logger.info(f"Initialized {len(ALL_COLLECTIONS)} ChromaDB collections")

    # =========================================================================
    # Metadata conversion
    # =========================================================================

    @staticmethod
    def node_to_metadata(node: MemoryNode) -> dict[str, Any]:
        """Convert a MemoryNode to flat ChromaDB metadata."""
        meta = node.metadata
        return {
            "agent_id": meta.agent_id,
            "visibility": meta.visibility.value if hasattr(meta.visibility, "value") else str(meta.visibility),
            "layer": meta.layer.value if hasattr(meta.layer, "value") else str(meta.layer),
            "memory_type": meta.memory_type.value if hasattr(meta.memory_type, "value") else str(meta.memory_type),
            "valence": meta.valence.value if hasattr(meta.valence, "value") else str(meta.valence),
            "arousal": meta.arousal,
            "salience": meta.salience,
            "tags_json": json.dumps(meta.tags),
            "concepts_json": json.dumps(meta.concepts),
            "episode_id": meta.episode_id or "",
            "session_id": meta.session_id or "",
            "conversation_thread": meta.conversation_thread or "",
            "source": meta.source,
            "created_at": node.created_at.isoformat(),
            "access_count": node.strength.access_count,
        }

    @staticmethod
    def result_to_dict(
        doc_id: str,
        content: str,
        metadata: dict[str, Any],
        distance: Optional[float] = None,
        collection: Optional[str] = None,
    ) -> dict[str, Any]:
        """Convert a ChromaDB result to a result dict."""
        similarity = round(1 - distance, 4) if distance is not None else None
        return {
            "id": doc_id,
            "content": content,
            "metadata": metadata,
            "similarity": similarity,
            "collection": collection,
        }

    # =========================================================================
    # CRUD operations
    # =========================================================================

    def add_node(self, collection: str, node: MemoryNode) -> str:
        """Add a MemoryNode to a collection."""
        coll = self._get_collection(collection)
        coll.add(
            ids=[node.id],
            documents=[node.content],
            metadatas=[self.node_to_metadata(node)],
        )
        return node.id

    def add(self, collection: str, ids: list[str], documents: list[str],
            metadatas: list[dict[str, Any]]) -> list[str]:
        """Raw add to collection."""
        coll = self._get_collection(collection)
        coll.add(ids=ids, documents=documents, metadatas=metadatas)
        return ids

    def search(
        self,
        collection: str,
        query: str,
        n_results: int = 10,
        where: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """Semantic search. Returns list of result dicts with similarity scores."""
        coll = self._get_collection(collection)

        # Embed query ourselves and pass as query_embeddings to avoid
        # ChromaDB 1.4+ embedding function protocol mismatches
        ef = self._get_embedding_fn()
        query_embedding = ef([query])[0]  # Single embedding as numpy array or list

        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding.tolist() if hasattr(query_embedding, 'tolist') else query_embedding],
            "n_results": min(n_results, coll.count()) if coll.count() > 0 else 1,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        try:
            results = coll.query(**kwargs)
        except Exception as e:
            logger.error(f"ChromaDB search failed: {e}")
            return []

        records = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                content = results["documents"][0][i] if results["documents"] else ""
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                distance = results["distances"][0][i] if results["distances"] else 1.0
                records.append(self.result_to_dict(
                    doc_id, content, metadata, distance, collection
                ))

        return records

    def get(self, collection: str, ids: list[str]) -> list[dict[str, Any]]:
        """Get documents by ID."""
        if not ids:
            return []
        coll = self._get_collection(collection)
        results = coll.get(ids=ids, include=["documents", "metadatas"])

        records = []
        if results["ids"]:
            for i, doc_id in enumerate(results["ids"]):
                content = results["documents"][i] if results["documents"] else ""
                metadata = results["metadatas"][i] if results["metadatas"] else {}
                records.append(self.result_to_dict(
                    doc_id, content, metadata, collection=collection
                ))
        return records

    def update(self, collection: str, ids: list[str], documents: list[str],
               metadatas: list[dict[str, Any]]) -> bool:
        """Update existing documents."""
        coll = self._get_collection(collection)
        try:
            coll.update(ids=ids, documents=documents, metadatas=metadatas)
            return True
        except Exception as e:
            logger.error(f"ChromaDB update failed: {e}")
            return False

    def update_node(self, collection: str, node: MemoryNode) -> bool:
        """Update a MemoryNode in the collection."""
        return self.update(
            collection,
            ids=[node.id],
            documents=[node.content],
            metadatas=[self.node_to_metadata(node)],
        )

    def delete(self, collection: str, ids: list[str]) -> bool:
        """Delete documents by ID."""
        if not ids:
            return True
        coll = self._get_collection(collection)
        try:
            coll.delete(ids=ids)
            return True
        except Exception as e:
            logger.error(f"ChromaDB delete failed: {e}")
            return False

    def count(self, collection: str) -> int:
        """Count documents in collection."""
        coll = self._get_collection(collection)
        return coll.count()

    def count_all(self) -> dict[str, int]:
        """Count documents across all collections."""
        return {name: self.count(name) for name in ALL_COLLECTIONS}
