"""Attachment model for multimodal memory support."""

from datetime import datetime
from typing import Optional
import uuid

from pydantic import BaseModel, Field, field_serializer

from cerebro.types import MediaType


class Attachment(BaseModel):
    """A media attachment linked to a memory.

    Represents an external file (image, PDF, audio, etc.) that is associated
    with a memory node. The actual file bytes are not stored in the database;
    only metadata, a local path/URI, and optional vision embedding references.
    """

    id: str = Field(default_factory=lambda: f"att_{uuid.uuid4().hex[:12]}")
    mime_type: str = "application/octet-stream"
    media_type: MediaType = MediaType.UNKNOWN
    file_path: Optional[str] = None  # local path or URI
    original_bytes_hash: Optional[str] = None  # sha256 of original bytes
    text_description: Optional[str] = None  # LLM/OCR-generated caption
    vision_embedding_id: Optional[str] = None  # ID in vision sidecar collection
    created_at: datetime = Field(default_factory=datetime.now)

    @field_serializer("created_at")
    def serialize_datetime(self, value: datetime) -> str:
        return value.isoformat() if value else ""
