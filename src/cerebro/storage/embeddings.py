"""Embedding functions for CerebroCortex.

Ported from Neo-Cortex with minimal changes. Priority order:
1. sentence-transformers (primary, CPU-based, works everywhere)
2. Ollama (alternative, if running locally)
3. Fallback hash (last resort, no semantic meaning)

All implementations return numpy arrays for ChromaDB 1.4+ compatibility.
"""

import hashlib
import logging
from typing import Protocol

import numpy as np

from cerebro.config import (
    EMBEDDING_DIM,
    OLLAMA_BASE_URL,
    OLLAMA_EMBEDDING_MODEL,
    SBERT_MODEL,
)

logger = logging.getLogger(__name__)


class EmbeddingFunction(Protocol):
    """Protocol for embedding functions."""

    def name(self) -> str: ...
    def embed(self, texts: list[str]) -> list[np.ndarray]: ...
    def embed_query(self, text: str) -> np.ndarray: ...
    def __call__(self, input: list[str]) -> list[np.ndarray]: ...


class SentenceTransformerEmbeddings:
    """Primary embedding function using sentence-transformers."""

    def __init__(self, model_name: str = SBERT_MODEL):
        self.model_name = model_name
        self._model = None
        self._name = f"sbert:{model_name}"

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading sentence-transformer model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def name(self) -> str:
        return self._name

    @property
    def dimension(self) -> int:
        return EMBEDDING_DIM

    def __call__(self, input: list[str]) -> list[np.ndarray]:
        return self.embed(input)

    def embed(self, texts: list[str]) -> list[np.ndarray]:
        model = self._load_model()
        embeddings = model.encode(texts, convert_to_numpy=True)
        return [row.astype(np.float32) for row in embeddings]

    def embed_query(self, text: str = None, *, input: str = None) -> np.ndarray:
        query = text if text is not None else input
        if query is None:
            raise ValueError("Must provide text or input")
        model = self._load_model()
        embedding = model.encode(query, convert_to_numpy=True)
        return embedding.astype(np.float32)


class OllamaEmbeddings:
    """Alternative embedding function using Ollama API."""

    def __init__(self, model: str = OLLAMA_EMBEDDING_MODEL, base_url: str = OLLAMA_BASE_URL):
        self.model = model
        self.base_url = base_url
        self._dimension = None
        self._name = f"ollama:{model}"

    def name(self) -> str:
        return self._name

    @property
    def dimension(self) -> int:
        return self._dimension or 768

    def __call__(self, input: list[str]) -> list[np.ndarray]:
        return self.embed(input)

    def embed(self, texts: list[str]) -> list[np.ndarray]:
        import httpx
        embeddings = []
        for text in texts:
            try:
                embeddings.append(self._embed_single(text))
            except Exception as e:
                logger.error(f"Ollama embed failed: {e}")
                if self._dimension:
                    embeddings.append(np.zeros(self._dimension, dtype=np.float32))
                else:
                    raise
        return embeddings

    def _embed_single(self, text: str) -> np.ndarray:
        import httpx
        with httpx.Client(timeout=60.0) as client:
            response = client.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model, "prompt": text},
            )
            response.raise_for_status()
            embedding = response.json()["embedding"]
            if self._dimension is None:
                self._dimension = len(embedding)
            return np.array(embedding, dtype=np.float32)

    def embed_query(self, text: str = None, *, input: str = None) -> np.ndarray:
        query = text if text is not None else input
        if query is None:
            raise ValueError("Must provide text or input")
        return self._embed_single(query)


class FallbackEmbeddings:
    """Hash-based pseudo-embeddings when nothing else is available.

    Returns deterministic numpy arrays for ChromaDB 1.4+ compatibility.
    No semantic meaning — only use as last resort.
    """

    def __init__(self, dimension: int = EMBEDDING_DIM):
        self._dimension = dimension
        self._name = "fallback"
        logger.warning("Using fallback embeddings - no semantic search available")

    def name(self) -> str:
        return self._name

    @property
    def dimension(self) -> int:
        return self._dimension

    def __call__(self, input: list[str]) -> list[np.ndarray]:
        return self.embed(input)

    def embed(self, texts: list[str]) -> list[np.ndarray]:
        return [self._hash_embed(text) for text in texts]

    def _hash_embed(self, text: str) -> np.ndarray:
        values = np.empty(self._dimension, dtype=np.float32)
        for i in range(self._dimension):
            hash_input = f"{text}:{i}".encode()
            hash_val = int(hashlib.md5(hash_input).hexdigest(), 16)
            values[i] = ((hash_val % 10000) / 5000) - 1
        return values

    def embed_query(self, text: str = None, *, input: str = None) -> np.ndarray:
        query = text if text is not None else input
        if query is None:
            raise ValueError("Must provide text or input")
        return self._hash_embed(query)


def check_sentence_transformers_available() -> bool:
    try:
        import sentence_transformers  # noqa: F401
        return True
    except ImportError:
        return False


def check_ollama_available() -> bool:
    try:
        import httpx
        with httpx.Client(timeout=5.0) as client:
            response = client.get(f"{OLLAMA_BASE_URL}/api/tags")
            if response.status_code != 200:
                return False
            models = response.json().get("models", [])
            model_names = [m.get("name", "").split(":")[0] for m in models]
            return OLLAMA_EMBEDDING_MODEL.split(":")[0] in model_names
    except Exception:
        return False


def get_embedding_function(prefer: str = "auto") -> EmbeddingFunction:
    """Get the best available embedding function."""
    if prefer == "auto":
        if check_sentence_transformers_available():
            logger.info("Using sentence-transformers embeddings")
            return SentenceTransformerEmbeddings()
        if check_ollama_available():
            logger.info("Using Ollama embeddings")
            return OllamaEmbeddings()
        logger.warning("No embedding model available, using fallback")
        return FallbackEmbeddings()
    elif prefer == "sbert":
        return SentenceTransformerEmbeddings()
    elif prefer == "ollama":
        return OllamaEmbeddings()
    elif prefer == "fallback":
        return FallbackEmbeddings()
    else:
        raise ValueError(f"Unknown embedding preference: {prefer}")
