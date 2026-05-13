"""Agent profile model."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_serializer


class AgentProfile(BaseModel):
    """Profile for a registered agent in CerebroCortex."""
    id: str
    display_name: str
    generation: int = Field(default=0, description="-1=origin, 0=primary, 1+=descendant")
    lineage: str = "Unknown"
    specialization: str = "General"
    origin_story: Optional[str] = None
    color: str = "#888888"
    symbol: str = "A"
    registered_at: datetime = Field(default_factory=datetime.now)

    @field_serializer("registered_at")
    def serialize_datetime(self, value: datetime) -> str:
        return value.isoformat()
