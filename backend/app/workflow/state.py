"""LangGraph state definition for the per-variant recommendation workflow.

Each node in the graph reads from and writes to this typed dict, ensuring
a clear contract between agents (editor, critic, refiner).
"""

from __future__ import annotations

from typing import Any, TypedDict

from app.models import AuditEntry


class RecommendationState(TypedDict, total=False):
    """State that flows through the per-variant LangGraph sub-graph.

    Fields are grouped into *immutable inputs* (set once before the
    graph starts) and *mutable state* (updated by agents each iteration).
    """

    # -- Immutable inputs (set once at graph entry) -----------------------
    original_image_b64: str
    recommendation_id: str
    recommendation_title: str
    recommendation_description: str
    recommendation_type: str
    brand_guidelines_text: str  # JSON-serialised BrandGuidelines
    runtime_config: Any  # RuntimeConfig instance (passed through, not serialised)

    # -- Mutable state (updated by agents across iterations) --------------
    plan: str
    edit_prompt: str
    plan_approved: bool
    plan_feedback: str
    plan_revision_count: int
    edited_image_b64: str
    evaluation_passed: bool
    evaluation_score: float
    evaluation_feedback: str
    refiner_feedback: str
    attempt: int
    max_attempts: int
    audit_trail: list[AuditEntry]
    status: str  # "accepted" | "max_retries_exceeded"
