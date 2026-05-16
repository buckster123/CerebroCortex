"""Tests for near-duplicate detection at ingestion time."""

import pytest

from cerebro.config import NEAR_DEDUP_THRESHOLD


class TestFindNearDuplicates:
    """Test CerebroCortex.find_near_duplicates() public API."""

    def test_exact_duplicate_found(self, multi_agent_cortex):
        ctx = multi_agent_cortex["cortex"]
        content = "Python is a versatile programming language for data science"
        node = ctx.remember(content, agent_id="CLAUDE")
        assert node is not None

        # Exact same content should be a near-duplicate
        matches = ctx.find_near_duplicates(content, threshold=0.95, top_k=5)
        assert len(matches) >= 1
        assert matches[0]["id"] == node.id
        assert matches[0]["similarity"] >= 0.95

    def test_similar_content_found(self, multi_agent_cortex):
        ctx = multi_agent_cortex["cortex"]
        original = "Machine learning models require large amounts of training data"
        node = ctx.remember(original, agent_id="CLAUDE")
        assert node is not None

        # Very similar content
        similar = "Machine learning models need large amounts of training data"
        matches = ctx.find_near_duplicates(similar, threshold=0.90, top_k=5)
        assert len(matches) >= 1
        # Should find the original
        ids = [m["id"] for m in matches]
        assert node.id in ids

    def test_different_content_no_match(self, multi_agent_cortex):
        ctx = multi_agent_cortex["cortex"]
        ctx.remember("The quick brown fox jumps over the lazy dog", agent_id="CLAUDE")

        matches = ctx.find_near_duplicates(
            "Quantum computing uses qubits instead of classical bits",
            threshold=0.95,
            top_k=5,
        )
        # Should find nothing at high threshold
        assert len(matches) == 0

    def test_agent_visibility_filter(self, multi_agent_cortex):
        ctx = multi_agent_cortex["cortex"]
        alice_content = "Alice's private note about neural networks"
        alice_node = ctx.remember(
            alice_content,
            agent_id="ALICE",
            visibility="private",
        )
        assert alice_node is not None

        # BOB should not see ALICE's private memory as a duplicate
        matches = ctx.find_near_duplicates(
            alice_content,
            threshold=0.90,
            agent_id="BOB",
            top_k=5,
        )
        ids = [m["id"] for m in matches]
        assert alice_node.id not in ids

        # ALICE should see her own private memory
        matches_alice = ctx.find_near_duplicates(
            alice_content,
            threshold=0.90,
            agent_id="ALICE",
            top_k=5,
        )
        ids_alice = [m["id"] for m in matches_alice]
        assert alice_node.id in ids_alice


class TestDedupAtIngestion:
    """Test that deduplication happens during ingestion pipeline."""

    def test_exact_duplicate_skipped(self, multi_agent_cortex):
        ctx = multi_agent_cortex["cortex"]
        from cerebro.ingestion import IngestionPipeline

        pipeline = IngestionPipeline(ctx)
        content = "Docker containers are lightweight virtualization units"

        # First ingestion
        report1 = pipeline.ingest_text(content, agent_id="CLAUDE")
        assert report1.memories_imported >= 1

        # Second ingestion of exact same content
        report2 = pipeline.ingest_text(content, agent_id="CLAUDE")
        # Should be skipped by thalamus exact dedup
        assert report2.memories_skipped >= 1

    def test_ingestion_result_counts(self, multi_agent_cortex):
        ctx = multi_agent_cortex["cortex"]
        from cerebro.ingestion import IngestionPipeline

        pipeline = IngestionPipeline(ctx)
        report = pipeline.ingest_text(
            "This is a test document for counting ingestion results accurately",
            agent_id="CLAUDE",
        )
        # Should have at least 1 imported, 0 skipped for fresh content
        assert report.memories_imported >= 1
        assert report.duration_seconds >= 0
