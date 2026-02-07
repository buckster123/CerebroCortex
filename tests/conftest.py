"""Shared test fixtures for CerebroCortex."""

import tempfile
from pathlib import Path

import pytest

from cerebro.cortex import CerebroCortex
from cerebro.models.memory import MemoryMetadata, MemoryNode
from cerebro.types import EmotionalValence, MemoryType, Visibility


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


@pytest.fixture
def multi_agent_cortex():
    """CerebroCortex with ALICE and BOB agents and scoped memories.

    Creates:
    - ALICE SHARED memory (visible to all)
    - ALICE PRIVATE memory (visible only to ALICE)
    - BOB PRIVATE memory (visible only to BOB)
    - ALICE THREAD memory (visible to conversation_thread="thread-42")
    """
    with tempfile.TemporaryDirectory() as d:
        ctx = CerebroCortex(
            db_path=Path(d) / "scope_test.db",
            chroma_dir=Path(d) / "chroma",
        )
        ctx.initialize()

        # Store memories with different visibility
        alice_shared = ctx.remember(
            content="Alice shared knowledge about Python decorators and metaprogramming patterns",
            tags=["python", "decorators"],
            agent_id="ALICE",
            visibility=Visibility.SHARED,
            salience=0.8,
        )
        alice_private = ctx.remember(
            content="Alice private secret about her internal configuration and API keys",
            tags=["config", "secret"],
            agent_id="ALICE",
            visibility=Visibility.PRIVATE,
            salience=0.8,
        )
        bob_private = ctx.remember(
            content="Bob private notes about his specialized debugging approach and tools",
            tags=["debugging", "tools"],
            agent_id="BOB",
            visibility=Visibility.PRIVATE,
            salience=0.8,
        )

        # Manually create a THREAD-scoped memory via graph store
        thread_node = MemoryNode(
            id="mem_scope_thread_01",
            content="Thread conversation about deployment strategy between Alice and Bob",
            metadata=MemoryMetadata(
                memory_type=MemoryType.SEMANTIC,
                tags=["deployment", "strategy"],
                agent_id="ALICE",
                visibility=Visibility.THREAD,
                conversation_thread="thread-42",
                salience=0.8,
            ),
        )
        ctx.graph.add_node(thread_node)
        coll = ctx._collection_for_type(thread_node.metadata.memory_type)
        ctx.vector.add_node(coll, thread_node)

        yield {
            "cortex": ctx,
            "alice_shared": alice_shared,
            "alice_private": alice_private,
            "bob_private": bob_private,
            "thread_node": thread_node,
        }
        ctx.close()
