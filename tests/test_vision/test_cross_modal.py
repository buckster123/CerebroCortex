"""Tests for cross-modal recall integration (Phase B)."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from cerebro.cortex import CerebroCortex
from cerebro.models.attachment import Attachment
from cerebro.storage.vision_embeddings import VisionEmbeddingFunction
from cerebro.types import MediaType


@pytest.fixture
def cross_modal_cortex():
    """CerebroCortex with a text memory and a vision-backed image memory."""
    with tempfile.TemporaryDirectory() as d:
        ctx = CerebroCortex(
            db_path=Path(d) / "test.db",
            chroma_dir=Path(d) / "chroma",
        )
        ctx.initialize()

        # Store a regular text memory
        text_node = ctx.remember(
            content="Python decorators are a powerful metaprogramming tool",
            tags=["python", "decorators"],
        )

        # Store an image memory (simulated, no actual image file needed for recall test)
        image_node = ctx.remember(
            content="[Image: screenshot of login bug with red error banner]",
            tags=["image", "screenshot", "bug"],
        )
        if image_node:
            image_node.metadata.media_type = MediaType.IMAGE
            image_node.metadata.attachments.append(
                Attachment(
                    mime_type="image/png",
                    media_type=MediaType.IMAGE,
                    file_path="/tmp/fake_screenshot.png",
                    text_description="screenshot of login bug with red error banner",
                )
            )
            ctx.graph._insert_attachment(image_node.id, image_node.metadata.attachments[0])
            ctx.coordinator.update_node(image_node)

        # Seed vision sidecar with a dummy embedding for the image memory
        if image_node and ctx._vision_store is not None and ctx._vision_store._collection is not None:
            dummy_emb = np.random.rand(512).astype(np.float32)
            ctx._vision_store._collection.add(
                ids=["att_test_crossmodal"],
                embeddings=[dummy_emb.tolist()],
                metadatas=[{"memory_id": image_node.id, "source": "vision_sidecar"}],
            )

        yield {
            "cortex": ctx,
            "text_node": text_node,
            "image_node": image_node,
        }
        ctx.close()


class TestCrossModalRecall:
    @pytest.fixture(autouse=True)
    def check_clip_loadable(self):
        pytest.importorskip("sentence_transformers")
        vef = VisionEmbeddingFunction()
        try:
            vef._load()
        except Exception as exc:
            pytest.skip(f"CLIP model not loadable in this environment: {exc}")

    def test_recall_without_vision(self, cross_modal_cortex):
        ctx = cross_modal_cortex["cortex"]
        results = ctx.recall("python decorators", top_k=5)
        assert len(results) > 0
        # Should find the text memory
        assert any("decorators" in r[0].content for r in results)

    def test_recall_with_vision_flag_no_vision_store(self, cross_modal_cortex):
        ctx = cross_modal_cortex["cortex"]
        # Even with include_vision=True, should work gracefully
        results = ctx.recall("screenshot", top_k=5, include_vision=True)
        # If vision store is available, it may or may not find results
        # The key is that it doesn't crash
        assert isinstance(results, list)

    def test_recall_boosts_existing_scores(self, cross_modal_cortex):
        ctx = cross_modal_cortex["cortex"]
        image_node = cross_modal_cortex["image_node"]

        if image_node is None:
            pytest.skip("Image node was gated out")
        if ctx._vision_store is None or ctx._vision_store._collection is None:
            pytest.skip("Vision sidecar not available")

        # Mock: inject the image memory ID into vision results by adding
        # a matching embedding for the query text
        try:
            query_emb = ctx._vision_store._embedding_fn.embed_text_for_vision("screenshot")
        except Exception as exc:
            pytest.skip(f"CLIP model not loadable in this environment: {exc}")
        ctx._vision_store._collection.add(
            ids=["att_query_match"],
            embeddings=[query_emb.tolist()],
            metadatas=[{"memory_id": image_node.id, "source": "vision_sidecar"}],
        )

        results = ctx.recall("screenshot", top_k=5, include_vision=True)
        assert isinstance(results, list)
        # Image memory should be findable
        ids = [r[0].id for r in results]
        assert image_node.id in ids
