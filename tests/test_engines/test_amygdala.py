"""Tests for the AffectEngine (amygdala)."""

from cerebro.engines.amygdala import AffectEngine
from cerebro.models.memory import MemoryMetadata, MemoryNode
from cerebro.types import EmotionalValence, MemoryType


class TestEmotionAnalysis:
    def test_positive_content(self, graph_store):
        engine = AffectEngine(graph_store)
        valence, arousal, adj = engine.analyze_emotion(
            "The deployment was a great success! Everything works perfectly."
        )
        assert valence == EmotionalValence.POSITIVE
        assert adj > 0

    def test_negative_content(self, graph_store):
        engine = AffectEngine(graph_store)
        valence, arousal, adj = engine.analyze_emotion(
            "The system crashed with a terrible error. Complete failure."
        )
        assert valence == EmotionalValence.NEGATIVE
        assert adj > 0

    def test_neutral_content(self, graph_store):
        engine = AffectEngine(graph_store)
        valence, arousal, adj = engine.analyze_emotion(
            "Python version 3.12 was released on October 2."
        )
        assert valence == EmotionalValence.NEUTRAL

    def test_mixed_content(self, graph_store):
        engine = AffectEngine(graph_store)
        valence, arousal, adj = engine.analyze_emotion(
            "The bug was terrible but we solved it and the solution was great."
        )
        assert valence == EmotionalValence.MIXED

    def test_high_arousal_markers(self, graph_store):
        engine = AffectEngine(graph_store)
        _, low_arousal, _ = engine.analyze_emotion("It was okay.")
        _, high_arousal, _ = engine.analyze_emotion(
            "Incredible! A breakthrough! This is urgent and critical!"
        )
        assert high_arousal > low_arousal

    def test_negative_boosts_salience_more(self, graph_store):
        engine = AffectEngine(graph_store)
        _, _, neg_adj = engine.analyze_emotion("Error! Crash! Bug! Failure!")
        _, _, pos_adj = engine.analyze_emotion("Great! Success! Works! Solved!")
        # Negative outcomes should boost salience more (learn from mistakes)
        assert neg_adj >= pos_adj


class TestApplyEmotion:
    def test_updates_node_metadata(self, graph_store):
        engine = AffectEngine(graph_store)
        node = MemoryNode(
            id="mem_emo1",
            content="The breakthrough was amazing after hours of work!",
            metadata=MemoryMetadata(salience=0.5),
        )

        enriched = engine.apply_emotion(node)
        assert enriched.metadata.valence == EmotionalValence.POSITIVE
        assert enriched.metadata.salience >= 0.5  # should increase
        assert enriched.id == node.id  # same node

    def test_preserves_content(self, graph_store):
        engine = AffectEngine(graph_store)
        node = MemoryNode(id="mem_emo2", content="Some emotional content that felt wrong")
        enriched = engine.apply_emotion(node)
        assert enriched.content == node.content


class TestAffectiveLinks:
    def test_creates_affective_links(self, graph_store):
        engine = AffectEngine(graph_store)

        # Add some positive memories
        for i in range(3):
            graph_store.add_node(MemoryNode(
                id=f"mem_aff_{i}",
                content=f"Positive memory {i}",
                metadata=MemoryMetadata(
                    valence=EmotionalValence.POSITIVE,
                    arousal=0.7,
                    salience=0.6,
                ),
            ))

        # New positive memory should link to existing ones
        new_node = MemoryNode(
            id="mem_aff_new",
            content="Another positive moment",
            metadata=MemoryMetadata(
                valence=EmotionalValence.POSITIVE,
                arousal=0.8,
                salience=0.7,
            ),
        )
        graph_store.add_node(new_node)

        created = engine.create_affective_links(new_node)
        assert len(created) >= 1


class TestReprocessing:
    def test_reprocess_boosts_negative(self, graph_store):
        engine = AffectEngine(graph_store)
        node = MemoryNode(
            id="mem_rp1",
            content="Tried the new approach",
            metadata=MemoryMetadata(salience=0.5),
        )
        graph_store.add_node(node)

        result = engine.reprocess_emotion(
            "mem_rp1", EmotionalValence.NEGATIVE,
        )
        assert result is True

        # Check salience increased
        updated = graph_store.get_node("mem_rp1")
        assert updated.metadata.salience > 0.5

    def test_reprocess_nonexistent(self, graph_store):
        engine = AffectEngine(graph_store)
        result = engine.reprocess_emotion(
            "mem_nonexistent", EmotionalValence.POSITIVE,
        )
        assert result is False
