"""Tests for memory models."""

import json
from datetime import datetime

from cerebro.models.memory import MemoryMetadata, MemoryNode, StrengthState
from cerebro.types import EmotionalValence, MemoryLayer, MemoryType, Visibility


class TestMemoryNode:
    def test_default_creation(self):
        node = MemoryNode(content="Test content")
        assert node.content == "Test content"
        assert node.id.startswith("mem_")
        assert node.metadata.memory_type == MemoryType.SEMANTIC
        assert node.metadata.layer == MemoryLayer.WORKING
        assert node.metadata.visibility == Visibility.SHARED
        assert node.metadata.valence == EmotionalValence.NEUTRAL
        assert node.metadata.salience == 0.5
        assert node.strength.stability == 1.0
        assert node.strength.difficulty == 5.0
        assert node.strength.access_count == 0

    def test_custom_metadata(self):
        node = MemoryNode(
            content="Async debugging strategy",
            metadata=MemoryMetadata(
                memory_type=MemoryType.PROCEDURAL,
                layer=MemoryLayer.LONG_TERM,
                tags=["async", "debugging"],
                valence=EmotionalValence.POSITIVE,
                salience=0.9,
                agent_id="RESEARCHER",
            ),
        )
        assert node.metadata.memory_type == MemoryType.PROCEDURAL
        assert node.metadata.layer == MemoryLayer.LONG_TERM
        assert "async" in node.metadata.tags
        assert node.metadata.salience == 0.9
        assert node.metadata.agent_id == "RESEARCHER"

    def test_strength_state_defaults(self):
        s = StrengthState()
        assert s.stability == 1.0
        assert s.difficulty == 5.0
        assert s.access_count == 0
        assert s.access_timestamps == []
        assert s.compressed_count == 0
        assert s.last_retrievability == 1.0

    def test_strength_state_bounds(self):
        s = StrengthState(stability=0.01, difficulty=1.0)
        assert s.stability == 0.01
        assert s.difficulty == 1.0

    def test_json_serialization(self):
        node = MemoryNode(
            content="Test",
            metadata=MemoryMetadata(tags=["a", "b"], concepts=["python"]),
        )
        data = node.model_dump()
        assert data["content"] == "Test"
        assert data["metadata"]["tags"] == ["a", "b"]

        json_str = node.model_dump_json()
        assert "Test" in json_str

    def test_unique_ids(self):
        n1 = MemoryNode(content="A")
        n2 = MemoryNode(content="B")
        assert n1.id != n2.id

    def test_all_memory_types(self):
        for mt in MemoryType:
            node = MemoryNode(
                content=f"Test {mt.value}",
                metadata=MemoryMetadata(memory_type=mt),
            )
            assert node.metadata.memory_type == mt
