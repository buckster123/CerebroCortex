"""Shared test fixtures for CerebroCortex."""

import tempfile
from pathlib import Path

import pytest

from cerebro.models.memory import MemoryMetadata, MemoryNode
from cerebro.types import EmotionalValence, MemoryType


@pytest.fixture
def temp_dir():
    """Temporary directory for test data."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def graph_store(temp_dir):
    """GraphStore with temporary SQLite database."""
    from cerebro.storage.graph_store import GraphStore

    store = GraphStore(db_path=temp_dir / "test.db")
    store.initialize()
    yield store
    store.close()


@pytest.fixture
def chroma_store(temp_dir):
    """ChromaStore with temporary directory."""
    from cerebro.storage.chroma_store import ChromaStore

    store = ChromaStore(persist_path=temp_dir / "chroma")
    store.initialize()
    yield store


@pytest.fixture
def sample_memories():
    """Pre-built sample memories for testing."""
    return [
        MemoryNode(
            id="mem_test_semantic_01",
            content="Python lists are mutable sequences that support indexing and slicing",
            metadata=MemoryMetadata(
                memory_type=MemoryType.SEMANTIC,
                tags=["python", "data-structures"],
                salience=0.6,
            ),
        ),
        MemoryNode(
            id="mem_test_episodic_01",
            content="Deployed the app using Docker on RPi5, encountered port conflict on 8080",
            metadata=MemoryMetadata(
                memory_type=MemoryType.EPISODIC,
                tags=["deployment", "docker", "rpi5"],
                valence=EmotionalValence.NEGATIVE,
                arousal=0.7,
                salience=0.8,
            ),
        ),
        MemoryNode(
            id="mem_test_procedural_01",
            content="When debugging async issues: 1) Check event loop state, 2) Verify all awaits, 3) Look for blocking calls",
            metadata=MemoryMetadata(
                memory_type=MemoryType.PROCEDURAL,
                tags=["debugging", "async", "workflow"],
                salience=0.9,
            ),
        ),
        MemoryNode(
            id="mem_test_affective_01",
            content="The OAuth2 breakthrough felt amazing after 3 hours of debugging JWT issues",
            metadata=MemoryMetadata(
                memory_type=MemoryType.AFFECTIVE,
                valence=EmotionalValence.POSITIVE,
                arousal=0.9,
                salience=0.85,
            ),
        ),
        MemoryNode(
            id="mem_test_prospective_01",
            content="Need to revisit auth hardening: JWT secret should be 32+ bytes for production",
            metadata=MemoryMetadata(
                memory_type=MemoryType.PROSPECTIVE,
                tags=["security", "auth", "todo"],
                salience=0.7,
            ),
        ),
    ]
