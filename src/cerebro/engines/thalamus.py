"""GatingEngine - Thalamic sensory gating.

The thalamus: filters incoming information, decides what is worth
remembering, assigns initial memory parameters.

Responsibilities:
- Deduplication (reject exact or near-exact duplicates)
- Initial layer assignment (sensory/working/long_term)
- Initial salience estimation
- Memory type classification
- Noise filtering (too short, too generic)
"""

import time
from typing import Optional

from cerebro.activation.decay import compute_current_retrievability
from cerebro.activation.strength import record_access
from cerebro.config import DEFAULT_AGENT_ID, LAYER_CONFIG
from cerebro.models.memory import MemoryMetadata, MemoryNode, StrengthState
from cerebro.storage.graph_store import GraphStore
from cerebro.types import MemoryLayer, MemoryType, Visibility


# Minimum content length to be worth storing
MIN_CONTENT_LENGTH = 10

# Keywords that boost salience
HIGH_SALIENCE_KEYWORDS = {
    "important", "critical", "bug", "fix", "error", "breakthrough",
    "discovery", "remember", "never", "always", "warning", "danger",
    "lesson", "learned", "insight",
}


class GatingEngine:
    """Decides what incoming information becomes a memory and how."""

    def __init__(self, graph: GraphStore):
        self._graph = graph

    def evaluate_input(
        self,
        content: str,
        memory_type: Optional[MemoryType] = None,
        tags: Optional[list[str]] = None,
        salience: Optional[float] = None,
        agent_id: str = DEFAULT_AGENT_ID,
        session_id: Optional[str] = None,
        visibility: Visibility = Visibility.SHARED,
    ) -> Optional[MemoryNode]:
        """Evaluate incoming content and create a MemoryNode if it passes gating.

        This is the main entry point for the "should I remember this?" decision.

        Returns:
            MemoryNode ready for storage, or None if gated out.
        """
        # Gate 1: minimum content length
        if len(content.strip()) < MIN_CONTENT_LENGTH:
            return None

        # Gate 2: deduplication
        existing_id = self._graph.find_duplicate_content(content)
        if existing_id:
            # Strengthen existing memory instead of creating duplicate
            existing = self._graph.get_node(existing_id)
            if existing:
                new_strength = record_access(existing.strength, time.time())
                self._graph.update_node_strength(existing_id, new_strength)
            return None

        # Classify and score
        resolved_type = memory_type or self._classify_type(content)
        resolved_salience = salience if salience is not None else self._estimate_salience(content, tags)
        initial_layer = self._assign_layer(resolved_salience, resolved_type)

        metadata = MemoryMetadata(
            agent_id=agent_id,
            visibility=visibility,
            layer=initial_layer,
            memory_type=resolved_type,
            tags=tags or [],
            session_id=session_id,
            salience=resolved_salience,
        )

        node = MemoryNode(
            content=content,
            metadata=metadata,
            strength=StrengthState(stability=self._initial_stability(resolved_salience)),
        )

        return node

    def _classify_type(self, content: str) -> MemoryType:
        """Heuristic memory type classification.

        Real classification would use LLM, but this provides a fast default.
        """
        lower = content.lower()

        # Procedural: steps, instructions, workflows
        if any(marker in lower for marker in [
            "step 1", "1)", "first,", "when you", "how to",
            "workflow", "procedure", "algorithm", "strategy",
        ]):
            return MemoryType.PROCEDURAL

        # Affective: emotional content
        if any(marker in lower for marker in [
            "felt", "feeling", "amazing", "frustrat", "excit",
            "disappoint", "breakthrough", "terrible", "love", "hate",
        ]):
            return MemoryType.AFFECTIVE

        # Prospective: future intentions
        if any(marker in lower for marker in [
            "need to", "should", "todo", "plan to", "will",
            "going to", "revisit", "later", "eventually",
        ]):
            return MemoryType.PROSPECTIVE

        # Episodic: temporal/narrative content
        if any(marker in lower for marker in [
            "then", "after", "before", "yesterday", "today",
            "session", "deployed", "tried", "encountered",
        ]):
            return MemoryType.EPISODIC

        # Default: semantic (facts, knowledge)
        return MemoryType.SEMANTIC

    def _estimate_salience(self, content: str, tags: Optional[list[str]] = None) -> float:
        """Estimate how important/memorable this content is."""
        score = 0.5  # baseline

        lower = content.lower()

        # Keyword boost
        keyword_hits = sum(1 for kw in HIGH_SALIENCE_KEYWORDS if kw in lower)
        score += min(keyword_hits * 0.1, 0.3)

        # Length bonus (longer content tends to be more substantive)
        if len(content) > 200:
            score += 0.1
        elif len(content) < 30:
            score -= 0.1

        # Tag bonus (user explicitly tagged = more intentional)
        if tags:
            score += min(len(tags) * 0.05, 0.15)

        # Question marks = might be worth answering later
        if "?" in content:
            score += 0.05

        # Exclamation = emphasis
        if "!" in content:
            score += 0.05

        return max(0.1, min(1.0, score))

    def _assign_layer(self, salience: float, memory_type: MemoryType) -> MemoryLayer:
        """Decide initial memory layer based on salience and type."""
        # High-value types go straight to working memory
        if memory_type in (MemoryType.PROCEDURAL, MemoryType.SCHEMATIC):
            return MemoryLayer.WORKING

        # High salience -> working, low -> sensory
        if salience >= 0.7:
            return MemoryLayer.WORKING
        elif salience >= 0.4:
            return MemoryLayer.WORKING
        else:
            return MemoryLayer.SENSORY

    def _initial_stability(self, salience: float) -> float:
        """Set initial FSRS stability based on salience.

        Higher salience memories start with higher stability.
        """
        # Range: 0.5 to 3.0 days based on salience
        return 0.5 + salience * 2.5
