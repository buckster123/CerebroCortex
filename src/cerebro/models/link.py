"""Associative link model - typed, weighted edges in the memory graph."""

from datetime import datetime
from typing import Optional
import uuid

from pydantic import BaseModel, Field

from cerebro.types import LinkType


class AssociativeLink(BaseModel):
    """A typed, weighted edge in the associative memory graph.

    Links represent relationships between memories discovered through:
    - Encoding: created when memories are stored in context
    - Retrieval: strengthened when memories are co-activated (Hebbian learning)
    - Dream consolidation: created/discovered by the Dream Engine
    - Explicit: user/agent explicitly declares a relationship
    """
    id: str = Field(default_factory=lambda: f"link_{uuid.uuid4().hex[:12]}")
    source_id: str = Field(description="Source MemoryNode ID")
    target_id: str = Field(description="Target MemoryNode ID")
    link_type: LinkType

    # Strength (Hebbian: strengthened by co-activation)
    weight: float = Field(default=0.5, ge=0.0, le=1.0)

    # Usage tracking
    created_at: datetime = Field(default_factory=datetime.now)
    last_activated: Optional[datetime] = None
    activation_count: int = Field(default=0, ge=0)

    # Provenance
    source: str = Field(
        default="system",
        description="How this link was created: encoding, retrieval, dream_sws, "
                    "dream_pattern, dream_rem, user, migration"
    )
    evidence: Optional[str] = Field(
        default=None,
        description="Human-readable explanation of why this link exists"
    )

    model_config = {"json_encoders": {datetime: lambda v: v.isoformat()}}
