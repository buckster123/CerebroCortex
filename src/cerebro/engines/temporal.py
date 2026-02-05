"""SemanticEngine - Semantic knowledge management.

The temporal lobe: manages factual/conceptual memories, extracts concepts,
creates semantic links between related knowledge.

Responsibilities:
- Concept extraction from content
- Semantic link creation between conceptually related memories
- Contradiction detection between facts
- Knowledge clustering by concept
"""

import json
import re
from typing import Optional

from cerebro.models.memory import MemoryNode
from cerebro.storage.graph_store import GraphStore
from cerebro.types import LinkType, MemoryType


# Common words to exclude from concept extraction
STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after", "above",
    "below", "between", "out", "off", "over", "under", "again", "further",
    "then", "once", "here", "there", "when", "where", "why", "how", "all",
    "each", "every", "both", "few", "more", "most", "other", "some",
    "such", "no", "nor", "not", "only", "own", "same", "so", "than",
    "too", "very", "just", "because", "but", "and", "or", "if", "while",
    "that", "this", "these", "those", "it", "its", "i", "you", "he",
    "she", "we", "they", "me", "him", "her", "us", "them", "my", "your",
    "his", "our", "their", "what", "which", "who", "whom",
}


class SemanticEngine:
    """Manages semantic knowledge: concepts, facts, and their relationships."""

    def __init__(self, graph: GraphStore):
        self._graph = graph

    def extract_concepts(self, content: str, max_concepts: int = 10) -> list[str]:
        """Extract key concepts from text content.

        Uses word frequency and simple heuristics. LLM-based extraction
        would be done at the cortex level for higher quality.
        """
        # Tokenize: split on non-alphanumeric, lowercase
        words = re.findall(r'[a-z][a-z0-9_-]+', content.lower())

        # Filter stop words and very short words
        meaningful = [w for w in words if w not in STOP_WORDS and len(w) > 2]

        # Count frequencies
        freq: dict[str, int] = {}
        for w in meaningful:
            freq[w] = freq.get(w, 0) + 1

        # Also extract multi-word concepts (bigrams of capitalized words from original)
        bigrams = re.findall(r'[A-Z][a-z]+\s+[A-Z][a-z]+', content)
        for bg in bigrams:
            key = bg.lower()
            freq[key] = freq.get(key, 0) + 2  # boost bigrams

        # Sort by frequency, take top N
        sorted_concepts = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        return [c[0] for c in sorted_concepts[:max_concepts]]

    def create_semantic_links(
        self,
        node: MemoryNode,
        max_links: int = 5,
    ) -> list[str]:
        """Create semantic links based on shared concepts.

        Links the given memory to existing memories that share concepts.
        """
        concepts = set(node.metadata.concepts) if node.metadata.concepts else set()
        if not concepts:
            concepts = set(self.extract_concepts(node.content))

        if not concepts:
            return []

        created = []

        # Find memories sharing concepts via SQLite JSON search
        for concept in concepts:
            rows = self._graph.conn.execute(
                """SELECT id, concepts_json FROM memory_nodes
                WHERE id != ? AND concepts_json LIKE ?
                LIMIT ?""",
                (node.id, f'%"{concept}"%', max_links),
            ).fetchall()

            for row in rows:
                other_concepts = set(json.loads(row["concepts_json"]))
                overlap = concepts & other_concepts
                if overlap:
                    weight = min(0.3 + 0.15 * len(overlap), 0.9)
                    link_id = self._graph.ensure_link(
                        node.id, row["id"], LinkType.SEMANTIC,
                        weight=weight, source="encoding",
                        evidence=f"Shared concepts: {', '.join(overlap)}",
                    )
                    created.append(link_id)

        return created

    def find_contradictions(
        self,
        node: MemoryNode,
        candidates: Optional[list[str]] = None,
    ) -> list[tuple[str, str]]:
        """Detect potential contradictions with existing memories.

        This is a heuristic check. The Dream Engine does deeper LLM-based
        contradiction detection.

        Returns:
            List of (memory_id, reason) tuples for potential contradictions.
        """
        contradictions = []
        content_lower = node.content.lower()

        # Look for negation patterns: "X is not Y" vs "X is Y"
        negation_patterns = [
            (r"(\w+) (?:is|are) not (\w+)", r"\1 (?:is|are) \2"),
            (r"(\w+) (?:doesn't|don't|does not|do not) (\w+)", r"\1 (?:does|do) \2"),
            (r"never (\w+)", r"always \1"),
        ]

        # Get candidate memories to check against
        if candidates:
            check_ids = candidates
        else:
            # Check against memories sharing concepts
            concepts = node.metadata.concepts or self.extract_concepts(node.content, 5)
            check_ids = []
            for concept in concepts[:3]:
                rows = self._graph.conn.execute(
                    """SELECT id FROM memory_nodes
                    WHERE id != ? AND memory_type = 'semantic'
                    AND concepts_json LIKE ?
                    LIMIT 10""",
                    (node.id, f'%"{concept}"%'),
                ).fetchall()
                check_ids.extend(r["id"] for r in rows)

        # Simple keyword contradiction: if one says "not X" and other says "X"
        for check_id in set(check_ids):
            other = self._graph.get_node(check_id)
            if not other or not other.content:
                continue
            other_lower = other.content.lower()

            # Check for direct negation patterns
            if ("not " in content_lower and "not " not in other_lower) or \
               ("not " not in content_lower and "not " in other_lower):
                # Rough check: share enough words to be about same topic?
                words_a = set(re.findall(r'[a-z]+', content_lower))
                words_b = set(re.findall(r'[a-z]+', other_lower))
                overlap = (words_a & words_b) - STOP_WORDS
                if len(overlap) >= 3:
                    contradictions.append((check_id, f"Potential negation conflict, shared terms: {', '.join(list(overlap)[:5])}"))

        return contradictions

    def get_concept_cluster(self, concept: str, max_results: int = 20) -> list[str]:
        """Get all memory IDs related to a concept."""
        rows = self._graph.conn.execute(
            """SELECT id FROM memory_nodes
            WHERE concepts_json LIKE ?
            ORDER BY salience DESC
            LIMIT ?""",
            (f'%"{concept}"%', max_results),
        ).fetchall()
        return [r["id"] for r in rows]

    def enrich_node(self, node: MemoryNode) -> MemoryNode:
        """Enrich a node with extracted concepts if not already present."""
        if node.metadata.concepts:
            return node

        concepts = self.extract_concepts(node.content)
        return MemoryNode(
            id=node.id,
            content=node.content,
            metadata=node.metadata.model_copy(update={"concepts": concepts}),
            strength=node.strength,
            created_at=node.created_at,
            last_accessed_at=node.last_accessed_at,
            promoted_at=node.promoted_at,
            link_count=node.link_count,
        )
