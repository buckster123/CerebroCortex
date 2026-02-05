"""Core enums and type definitions for CerebroCortex."""

from enum import Enum


class MemoryType(str, Enum):
    """The six modalities of memory."""
    EPISODIC = "episodic"        # temporal sequences with context
    SEMANTIC = "semantic"        # facts, concepts, relationships
    PROCEDURAL = "procedural"    # strategies, workflows, patterns
    AFFECTIVE = "affective"      # emotional markers, reactions
    PROSPECTIVE = "prospective"  # future intentions, deferred plans
    SCHEMATIC = "schematic"      # abstractions extracted from episodes


class LinkType(str, Enum):
    """Types of associative links between memories."""
    TEMPORAL = "temporal"         # A happened before/after B
    CAUSAL = "causal"             # A caused B
    SEMANTIC = "semantic"         # A is conceptually related to B
    AFFECTIVE = "affective"      # A evokes similar emotion as B
    CONTEXTUAL = "contextual"    # A and B share context (project/session)
    CONTRADICTS = "contradicts"  # A conflicts with B
    SUPPORTS = "supports"        # A provides evidence for B
    DERIVED_FROM = "derived_from"  # B was abstracted/extracted from A
    PART_OF = "part_of"          # A is a step within episode B


class MemoryLayer(str, Enum):
    """Memory durability layers, inspired by multi-store memory model."""
    SENSORY = "sensory"      # minutes to hours
    WORKING = "working"      # hours to days
    LONG_TERM = "long_term"  # days to months
    CORTEX = "cortex"        # permanent


class Visibility(str, Enum):
    """Memory visibility/sharing scope."""
    PRIVATE = "private"  # agent-only
    SHARED = "shared"    # all agents
    THREAD = "thread"    # cross-agent dialogue


class EmotionalValence(str, Enum):
    """Emotional coloring of a memory."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"


class DreamPhase(str, Enum):
    """Phases of the Dream Engine consolidation cycle."""
    SWS_REPLAY = "sws_replay"
    PATTERN_EXTRACTION = "pattern_extraction"
    SCHEMA_FORMATION = "schema_formation"
    EMOTIONAL_REPROCESSING = "emotional_reprocessing"
    PRUNING = "pruning"
    REM_RECOMBINATION = "rem_recombination"
