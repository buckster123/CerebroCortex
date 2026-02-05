"""Activation and recall result models."""

from typing import Any

from pydantic import BaseModel, Field


class ActivationResult(BaseModel):
    """Result of computing activation for a single memory."""
    memory_id: str

    # ACT-R components
    base_level_activation: float = Field(description="B(t) = ln(Σ t_k^{-d})")
    associative_activation: float = Field(
        default=0.0, description="Σ w_j * s_ij from spreading activation"
    )
    total_activation: float = Field(description="A(t) = B(t) + associative + salience_bonus")

    # FSRS components
    fsrs_stability: float
    fsrs_difficulty: float
    retrievability: float = Field(description="R(t,S) = (1 + t/9S)^{-1}")

    # Combined
    recall_probability: float = Field(
        description="Final recall probability combining activation and retrievability"
    )

    # Debug info
    time_since_last_access_hours: float = 0.0
    contributing_links: list[dict[str, Any]] = Field(default_factory=list)


class SpreadingActivationResult(BaseModel):
    """Result of spreading activation from a query."""
    seed_ids: list[str] = Field(description="Initial vector search result IDs")
    activated: dict[str, float] = Field(
        default_factory=dict,
        description="memory_id -> activation score for all activated memories"
    )
    total_nodes_visited: int = 0
    activation_waves: int = Field(default=0, description="Number of hops performed")


class RecallResult(BaseModel):
    """A single memory returned from a recall query, with scoring breakdown."""
    memory_id: str
    content: str
    memory_type: str
    layer: str

    # Scoring components
    vector_similarity: float = 0.0
    activation_score: float = 0.0
    retrievability: float = 0.0
    salience: float = 0.0
    final_score: float = 0.0

    # Metadata
    tags: list[str] = Field(default_factory=list)
    valence: str = "neutral"
    agent_id: str = "CLAUDE"
    created_at: str = ""
    access_count: int = 0
