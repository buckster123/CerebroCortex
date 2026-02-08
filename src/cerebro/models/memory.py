"""Core memory node model - the fundamental unit of CerebroCortex."""

from datetime import datetime
from typing import Optional
import uuid

from pydantic import BaseModel, Field

from cerebro.types import EmotionalValence, MemoryLayer, MemoryType, Visibility


class StrengthState(BaseModel):
    """FSRS + ACT-R hybrid strength parameters for a single memory.

    Tracks both the FSRS stability/difficulty (spaced-repetition scheduling)
    and ACT-R access history (power-law base-level activation).
    """
    # FSRS parameters
    stability: float = Field(default=1.0, ge=0.01, description="Days until R=90%")
    difficulty: float = Field(default=5.0, ge=1.0, le=10.0, description="Intrinsic difficulty 1-10")

    # Access history for ACT-R base-level activation B(t) = ln(Σ t_k^{-d})
    access_timestamps: list[float] = Field(
        default_factory=list,
        description="Unix timestamps of accesses (most recent 50 kept individually)"
    )
    access_count: int = Field(default=0, ge=0)

    # Compressed history for old accesses (beyond MAX_STORED_TIMESTAMPS)
    compressed_count: int = Field(default=0, ge=0, description="Number of compressed old accesses")
    compressed_avg_interval: float = Field(
        default=0.0, ge=0.0,
        description="Average interval in seconds between compressed accesses"
    )

    # Cached computed values (recomputed on access/decay tick)
    last_retrievability: float = Field(default=1.0, ge=0.0, le=1.0)
    last_activation: float = 0.0
    last_computed_at: Optional[float] = None  # unix timestamp


class MemoryMetadata(BaseModel):
    """Rich metadata for a memory node."""
    # Identity and scope
    agent_id: str = "CLAUDE"
    visibility: Visibility = Visibility.SHARED
    layer: MemoryLayer = MemoryLayer.WORKING
    memory_type: MemoryType = MemoryType.SEMANTIC

    # Content classification
    tags: list[str] = Field(default_factory=list)
    concepts: list[str] = Field(default_factory=list, description="Extracted key concepts")

    # Temporal context
    session_id: Optional[str] = None
    conversation_thread: Optional[str] = None
    episode_id: Optional[str] = None

    # Relationships (for thread/agent compatibility)
    responding_to: list[str] = Field(default_factory=list)
    related_agents: list[str] = Field(default_factory=list)
    recipient: Optional[str] = Field(default=None, description="Target agent for directed messages")

    # Emotional dimension (Amygdala)
    valence: EmotionalValence = EmotionalValence.NEUTRAL
    arousal: float = Field(default=0.5, ge=0.0, le=1.0, description="Calm(0) to excited(1)")
    salience: float = Field(default=0.5, ge=0.0, le=1.0, description="Importance/significance")

    # Source tracking (Metacognition)
    source: str = Field(
        default="user_input",
        description="Origin: user_input, llm_generation, consolidation, dream, import"
    )
    derived_from: list[str] = Field(
        default_factory=list,
        description="Parent memory IDs this was derived from"
    )


class MemoryNode(BaseModel):
    """The fundamental unit of CerebroCortex memory.

    Stored across three backends:
    - ChromaDB: content + embedding + flat metadata for vector search
    - SQLite: full metadata + strength state + graph node
    - igraph: graph vertex for fast traversal
    """
    id: str = Field(default_factory=lambda: f"mem_{uuid.uuid4().hex[:12]}")
    content: str

    metadata: MemoryMetadata = Field(default_factory=MemoryMetadata)
    strength: StrengthState = Field(default_factory=StrengthState)

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now)
    last_accessed_at: Optional[datetime] = None
    promoted_at: Optional[datetime] = None

    # Cached graph info (populated from graph store, not stored in ChromaDB)
    link_count: int = Field(default=0, description="Cached count of associative links")

    # Search result field (populated during recall, not persisted)
    similarity: Optional[float] = Field(default=None, exclude=True)

    model_config = {"json_encoders": {datetime: lambda v: v.isoformat()}}
