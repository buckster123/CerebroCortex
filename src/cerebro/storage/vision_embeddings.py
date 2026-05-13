"""Vision embedding sidecar for cross-modal search (Path b).

Text embeddings and vision embeddings live in **separate vector spaces**.
- Every image memory gets a **text embedding** of its caption/description
  (goes into normal ChromaDB collections).
- Every image memory **optionally** gets a **vision embedding** stored in a
  dedicated sidecar collection.
- At recall time, text query searches text vectors. If ``include_vision=True``,
  the query is also encoded via a text-to-vision bridge model (e.g. CLIP text
  encoder) and the vision sidecar is searched. Results are merged by ID.

Requires ``pip install cerebro-cortex[vision]`` (sentence-transformers + pillow).
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

VISION_COLLECTION_NAME = "cerebro_vision"

if TYPE_CHECKING:
    from chromadb.api import ClientAPI


class VisionEmbeddingFunction:
    """Wrapper for a vision embedding model (CLIP-style via sentence-transformers).

    Falls back to None if sentence-transformers is not installed.
    """

    def __init__(self, model_name: str = "clip-ViT-B-32"):
        self.model_name = model_name
        self._model: Any = None

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.model_name)
                logger.info(f"Vision model loaded: {self.model_name}")
            except ImportError as exc:
                logger.warning(
                    "sentence-transformers not installed; vision search disabled. "
                    f"Install with: pip install cerebro-cortex[vision] ({exc})"
                )
                raise
        return self._model

    def embed_image(self, image_path: str | Path) -> np.ndarray:
        """Encode an image into a vision embedding vector."""
        model = self._load()
        emb = model.encode(str(image_path), convert_to_numpy=True)
        return emb.astype(np.float32)

    def embed_text_for_vision(self, text: str) -> np.ndarray:
        """Encode text using the SAME vision model for cross-modal search."""
        model = self._load()
        emb = model.encode(text, convert_to_numpy=True)
        return emb.astype(np.float32)

    @property
    def dimension(self) -> int:
        """Embedding dimension. Defaults to 512 for CLIP ViT-B-32."""
        return 512

    @property
    def available(self) -> bool:
        """Quick check without triggering model load."""
        try:
            import sentence_transformers  # noqa: F401
            return True
        except ImportError:
            return False


class VisionVectorStore:
    """ChromaDB wrapper for the vision sidecar collection."""

    def __init__(
        self,
        chroma_client: "ClientAPI",
        embedding_fn: Optional[VisionEmbeddingFunction] = None,
    ):
        self._client = chroma_client
        self._embedding_fn = embedding_fn
        self._collection: Any = None

    def initialize(self) -> bool:
        """Create/get the vision sidecar collection.

        Returns True if initialized successfully, False if vision deps missing.
        """
        if self._embedding_fn is None:
            logger.info("VisionVectorStore: no embedding function, skipping init")
            return False
        try:
            self._collection = self._client.get_or_create_collection(
                name=VISION_COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(f"Vision sidecar collection ready: {VISION_COLLECTION_NAME}")
            return True
        except Exception as exc:
            logger.warning(f"Failed to initialize vision collection: {exc}")
            return False

    def add_image(
        self,
        attachment_id: str,
        image_path: str | Path,
        memory_id: str,
    ) -> Optional[str]:
        """Add a vision embedding for an image attachment.

        Returns the attachment_id on success, None on failure.
        """
        if self._collection is None or self._embedding_fn is None:
            return None
        try:
            embedding = self._embedding_fn.embed_image(image_path)
            self._collection.add(
                ids=[attachment_id],
                embeddings=[embedding.tolist()],
                metadatas=[{"memory_id": memory_id, "source": "vision_sidecar"}],
            )
            logger.debug(f"Vision embedding stored for {attachment_id}")
            return attachment_id
        except Exception as exc:
            logger.error(f"Vision embedding failed for {attachment_id}: {exc}")
            return None

    def search_by_text(self, query: str, n_results: int = 10) -> list[dict[str, Any]]:
        """Search vision collection using a text query (cross-modal)."""
        if self._collection is None or self._embedding_fn is None:
            return []
        try:
            query_emb = self._embedding_fn.embed_text_for_vision(query)
            results = self._collection.query(
                query_embeddings=[query_emb.tolist()],
                n_results=n_results,
                include=["metadatas", "distances"],
            )
            return self._unpack_results(results)
        except Exception as exc:
            logger.warning(f"Vision text search failed: {exc}")
            return []

    def search_by_image(
        self,
        image_path: str | Path,
        n_results: int = 10,
    ) -> list[dict[str, Any]]:
        """Search vision collection using an image query (image-to-image)."""
        if self._collection is None or self._embedding_fn is None:
            return []
        try:
            query_emb = self._embedding_fn.embed_image(image_path)
            results = self._collection.query(
                query_embeddings=[query_emb.tolist()],
                n_results=n_results,
                include=["metadatas", "distances"],
            )
            return self._unpack_results(results)
        except Exception as exc:
            logger.warning(f"Vision image search failed: {exc}")
            return []

    @staticmethod
    def _unpack_results(results: dict[str, Any]) -> list[dict[str, Any]]:
        """Unpack ChromaDB query results into structured records."""
        records: list[dict[str, Any]] = []
        if not results.get("ids") or not results["ids"][0]:
            return records
        for i, aid in enumerate(results["ids"][0]):
            records.append({
                "attachment_id": aid,
                "memory_id": results["metadatas"][0][i].get("memory_id"),
                "distance": results["distances"][0][i],
            })
        return records

    def count(self) -> int:
        if self._collection is None:
            return 0
        return self._collection.count()
