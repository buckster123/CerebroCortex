"""AffectEngine - Emotional processing and salience modulation.

The amygdala: processes emotional content, adjusts salience based on
outcomes, creates affective links between emotionally similar memories.

Responsibilities:
- Compute emotional valence and arousal from content
- Adjust salience based on emotional intensity
- Create affective links between emotionally related memories
- Track emotional trajectories in episodes
- Boost salience for negative outcomes (learn from mistakes)
"""

import json
from typing import Optional

from cerebro.models.memory import MemoryNode
from cerebro.storage.graph_store import GraphStore
from cerebro.types import EmotionalValence, LinkType


# Emotion keyword mappings
POSITIVE_MARKERS = {
    "amazing", "breakthrough", "excellent", "great", "perfect",
    "solved", "success", "works", "love", "beautiful", "happy",
    "excited", "wonderful", "fantastic",
}
NEGATIVE_MARKERS = {
    "bug", "broken", "crash", "error", "fail", "frustrat",
    "terrible", "wrong", "hate", "awful", "disappoint", "stuck",
    "confused", "impossible", "nightmare",
}
HIGH_AROUSAL_MARKERS = {
    "!", "urgent", "critical", "panic", "incredible", "shocking",
    "breakthrough", "eureka", "finally", "nightmare", "disaster",
}


class AffectEngine:
    """Processes emotional dimensions of memories."""

    def __init__(self, graph: GraphStore):
        self._graph = graph

    def analyze_emotion(self, content: str) -> tuple[EmotionalValence, float, float]:
        """Analyze emotional content of text.

        Returns:
            (valence, arousal, salience_adjustment)
            - valence: positive/negative/neutral/mixed
            - arousal: 0.0 (calm) to 1.0 (excited)
            - salience_adjustment: how much to adjust salience (+/- delta)
        """
        lower = content.lower()

        pos_count = sum(1 for m in POSITIVE_MARKERS if m in lower)
        neg_count = sum(1 for m in NEGATIVE_MARKERS if m in lower)
        arousal_count = sum(1 for m in HIGH_AROUSAL_MARKERS if m in lower)

        # Determine valence
        if pos_count > 0 and neg_count > 0:
            valence = EmotionalValence.MIXED
        elif pos_count > neg_count:
            valence = EmotionalValence.POSITIVE
        elif neg_count > pos_count:
            valence = EmotionalValence.NEGATIVE
        else:
            valence = EmotionalValence.NEUTRAL

        # Compute arousal
        emotion_intensity = pos_count + neg_count
        arousal = min(0.3 + emotion_intensity * 0.15 + arousal_count * 0.1, 1.0)

        # Salience adjustment: negative outcomes boost salience more (learn from mistakes)
        salience_adj = 0.0
        if neg_count > 0:
            salience_adj += min(neg_count * 0.1, 0.3)  # negative gets bigger boost
        if pos_count > 0:
            salience_adj += min(pos_count * 0.05, 0.15)
        if arousal_count > 0:
            salience_adj += min(arousal_count * 0.05, 0.1)

        return valence, arousal, salience_adj

    def apply_emotion(self, node: MemoryNode) -> MemoryNode:
        """Apply emotional analysis to a memory node, updating its metadata.

        Returns a new MemoryNode with updated valence, arousal, and salience.
        """
        valence, arousal, salience_adj = self.analyze_emotion(node.content)

        # Only override if not explicitly set (allow user overrides)
        new_valence = valence
        new_arousal = max(node.metadata.arousal, arousal)  # take the higher arousal
        new_salience = max(0.1, min(1.0, node.metadata.salience + salience_adj))

        return MemoryNode(
            id=node.id,
            content=node.content,
            metadata=node.metadata.model_copy(update={
                "valence": new_valence,
                "arousal": new_arousal,
                "salience": new_salience,
            }),
            strength=node.strength,
            created_at=node.created_at,
            last_accessed_at=node.last_accessed_at,
            promoted_at=node.promoted_at,
            link_count=node.link_count,
        )

    def create_affective_links(
        self,
        node: MemoryNode,
        max_links: int = 3,
    ) -> list[str]:
        """Link this memory to others with similar emotional profile.

        Finds memories with matching valence and creates affective links.
        Only links to memories the owning agent can access (scope-aware).
        """
        created = []
        valence = node.metadata.valence

        # Build scope filter to prevent cross-agent PRIVATE links
        from cerebro.cortex import _scope_sql
        scope_clause, scope_params = _scope_sql(node.metadata.agent_id)

        # Query for memories with same valence
        rows = self._graph.conn.execute(
            f"""SELECT id, arousal, salience FROM memory_nodes
            WHERE id != ? AND valence = ?{scope_clause}
            ORDER BY salience DESC
            LIMIT ?""",
            (node.id, valence.value if isinstance(valence, EmotionalValence) else valence,
             *scope_params, max_links),
        ).fetchall()

        for row in rows:
            # Weight based on arousal similarity
            arousal_diff = abs(node.metadata.arousal - row["arousal"])
            weight = max(0.3, 0.7 - arousal_diff * 0.5)

            link_id = self._graph.ensure_link(
                node.id, row["id"], LinkType.AFFECTIVE,
                weight=weight, source="encoding",
                evidence=f"Shared {valence.value if isinstance(valence, EmotionalValence) else valence} valence",
            )
            created.append(link_id)

        return created

    def reprocess_emotion(
        self,
        node_id: str,
        outcome: EmotionalValence,
        salience_boost: float = 0.0,
    ) -> bool:
        """Reprocess a memory's emotional state based on outcomes.

        Called during dream consolidation or when outcomes are known.
        Negative outcomes boost salience (learn from mistakes).
        """
        node = self._graph.get_node(node_id)
        if not node:
            return False

        # Negative outcomes get salience boost
        adjustment = salience_boost
        if outcome == EmotionalValence.NEGATIVE:
            adjustment += 0.15

        new_salience = max(0.1, min(1.0, node.metadata.salience + adjustment))

        return self._graph.update_node_metadata(
            node_id,
            valence=outcome.value if isinstance(outcome, EmotionalValence) else outcome,
            salience=new_salience,
        )

    def get_emotional_summary(
        self,
        agent_id: Optional[str] = None,
        conversation_thread: Optional[str] = None,
    ) -> dict[str, int]:
        """Get a breakdown of memories by emotional valence."""
        from cerebro.cortex import _scope_sql
        scope_clause, scope_params = _scope_sql(agent_id, conversation_thread)

        rows = self._graph.conn.execute(
            f"SELECT valence, COUNT(*) as c FROM memory_nodes WHERE 1=1{scope_clause} GROUP BY valence",
            scope_params,
        ).fetchall()
        return {row["valence"]: row["c"] for row in rows}
