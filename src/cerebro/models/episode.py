"""Episode model - temporal sequences of related memories."""

from datetime import datetime
from typing import Optional
import uuid

from pydantic import BaseModel, Field

from cerebro.types import EmotionalValence


class EpisodeStep(BaseModel):
    """A single step in an episodic sequence."""
    memory_id: str
    position: int = Field(ge=0, description="Order in sequence (0-based)")
    timestamp: datetime = Field(default_factory=datetime.now)
    role: str = Field(
        default="event",
        description="Role in the episode: event, context, outcome, reflection"
    )


class Episode(BaseModel):
    """A temporal sequence of related memories.

    Episodes capture the narrative structure of experiences:
    what happened, in what order, and with what outcome.
    The Dream Engine processes episodes to extract schemas and patterns.
    """
    id: str = Field(default_factory=lambda: f"ep_{uuid.uuid4().hex[:12]}")
    title: Optional[str] = None
    steps: list[EpisodeStep] = Field(default_factory=list)

    # Context
    session_id: Optional[str] = None
    agent_id: str = "CLAUDE"
    tags: list[str] = Field(default_factory=list)

    # Temporal bounds
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None

    # Emotional summary
    overall_valence: EmotionalValence = EmotionalValence.NEUTRAL
    peak_arousal: float = Field(default=0.5, ge=0.0, le=1.0)

    # Consolidation state (set by Dream Engine)
    consolidated: bool = False
    schema_extracted: bool = False

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now)

    model_config = {"json_encoders": {datetime: lambda v: v.isoformat()}}
