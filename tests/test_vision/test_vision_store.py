"""Tests for vision embedding sidecar (Phase B)."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from cerebro.cortex import CerebroCortex
from cerebro.storage.vision_embeddings import (
    VisionEmbeddingFunction,
    VisionVectorStore,
    VISION_COLLECTION_NAME,
)


@pytest.fixture
def vision_cortex():
    """CerebroCortex with vision support initialized."""
    with tempfile.TemporaryDirectory() as d:
        ctx = CerebroCortex(
            db_path=Path(d) / "test.db",
            chroma_dir=Path(d) / "chroma",
        )
        ctx.initialize()
        yield ctx
        ctx.close()


class TestVisionEmbeddingFunction:
    @pytest.fixture(autouse=True)
    def check_clip_loadable(self):
        pytest.importorskip("sentence_transformers")
        vef = VisionEmbeddingFunction()
        try:
            vef._load()
        except Exception as exc:
            pytest.skip(f"CLIP model not loadable in this environment: {exc}")

    def test_available(self):
        vef = VisionEmbeddingFunction()
        # sentence-transformers is in base deps, so should be available
        assert vef.available is True

    def test_embed_text(self):
        vef = VisionEmbeddingFunction()
        emb = vef.embed_text_for_vision("a photo of a dog")
        assert isinstance(emb, np.ndarray)
        assert emb.dtype == np.float32
        assert emb.shape == (vef.dimension,)

    def test_dimension(self):
        vef = VisionEmbeddingFunction()
        assert vef.dimension == 512


class TestVisionVectorStore:
    @pytest.fixture(autouse=True)
    def check_clip_loadable(self):
        """Skip vision vector store tests if CLIP model can't load."""
        pytest.importorskip("sentence_transformers")
        vef = VisionEmbeddingFunction()
        try:
            vef._load()
        except Exception as exc:
            pytest.skip(f"CLIP model not loadable in this environment: {exc}")

    def test_initialize_without_embedding_fn(self, vision_cortex):
        client = vision_cortex._chroma._get_client()
        store = VisionVectorStore(client, embedding_fn=None)
        assert store.initialize() is False
        assert store.count() == 0

    def test_initialize_with_embedding_fn(self, vision_cortex):
        client = vision_cortex._chroma._get_client()
        vef = VisionEmbeddingFunction()
        store = VisionVectorStore(client, embedding_fn=vef)
        assert store.initialize() is True

    def test_add_and_search_by_text(self, vision_cortex):
        client = vision_cortex._chroma._get_client()
        vef = VisionEmbeddingFunction()
        store = VisionVectorStore(client, embedding_fn=vef)
        store.initialize()

        # Add a dummy embedding directly (avoid needing real image)
        dummy_emb = np.random.rand(512).astype(np.float32)
        store._collection.add(
            ids=["att_test_01"],
            embeddings=[dummy_emb.tolist()],
            metadatas=[{"memory_id": "mem_test_01", "source": "vision_sidecar"}],
        )

        # Search with text query
        results = store.search_by_text("random query", n_results=5)
        assert len(results) == 1
        assert results[0]["attachment_id"] == "att_test_01"
        assert results[0]["memory_id"] == "mem_test_01"

    def test_search_by_image_missing_file(self, vision_cortex):
        client = vision_cortex._chroma._get_client()
        vef = VisionEmbeddingFunction()
        store = VisionVectorStore(client, embedding_fn=vef)
        store.initialize()

        results = store.search_by_image("/nonexistent/image.png")
        assert results == []

    def test_count(self, vision_cortex):
        client = vision_cortex._chroma._get_client()
        vef = VisionEmbeddingFunction()
        store = VisionVectorStore(client, embedding_fn=vef)
        store.initialize()
        assert store.count() == 0


class TestCerebroCortexVisionInit:
    def test_vision_store_attribute(self, vision_cortex):
        assert hasattr(vision_cortex, "_vision_store")

    def test_vision_store_initialized(self, vision_cortex):
        pytest.importorskip("sentence_transformers")
        # If sentence-transformers is available, _vision_store should be set
        # (could be None if model download fails, but usually it's a VisionVectorStore)
        assert vision_cortex._vision_store is not None
