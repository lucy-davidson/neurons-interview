"""Pydantic domain models for jobs, recommendations, and API responses.

These models serve double duty: they define the REST API contract (via
FastAPI's automatic schema generation) and hold in-memory job state.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field

# ── Request / Response ──────────────────────────────────────────────


class Recommendation(BaseModel):
    """A single textual recommendation for improving a marketing creative."""

    id: str
    title: str
    description: str
    type: str


class BrandGuidelines(BaseModel):
    """Constraints the agentic workflow must respect during editing."""

    protected_regions: list[str] = Field(default_factory=list)
    typography: str = ""
    aspect_ratio: str = ""
    brand_elements: str = ""


class JobRequest(BaseModel):
    """Body sent by the client when creating a new job."""

    recommendations: list[Recommendation]
    brand_guidelines: BrandGuidelines


class JobStatus(str, Enum):
    """Lifecycle states for a background job."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class AuditEntry(BaseModel):
    """Immutable record of a single agent action within the workflow."""

    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    agent: str
    action: str
    detail: str


class VariantResult(BaseModel):
    """Outcome of one variant through the edit-evaluate-refine loop."""

    variant_id: str
    variant_title: str
    variant_description: str
    status: str  # "accepted" | "max_retries_exceeded"
    attempts: int
    edited_image_b64: str | None = None
    evaluation_score: float | None = None
    evaluation_feedback: str | None = None
    audit_trail: list[AuditEntry] = Field(default_factory=list)
    # Provider tracking for experiment comparison
    text_provider: str = ""
    image_provider: str = ""
    text_model: str = ""
    image_model: str = ""


class RecommendationResult(BaseModel):
    """Container for all variant results belonging to one recommendation."""

    recommendation_id: str
    recommendation_title: str
    variants: list[VariantResult] = Field(default_factory=list)


class JobSummary(BaseModel):
    """Lightweight projection used by the job-list endpoint."""

    job_id: str
    status: JobStatus
    created_at: str
    num_recommendations: int
    error: str | None = None


class JobListResponse(BaseModel):
    """Envelope returned by ``GET /api/v1/jobs``."""

    jobs: list[JobSummary]


class JobResponse(BaseModel):
    """Full job detail returned by ``GET /api/v1/jobs/{job_id}``."""

    job_id: str
    status: JobStatus
    created_at: str
    results: list[RecommendationResult] = Field(default_factory=list)
    error: str | None = None
    warnings: list[str] = Field(default_factory=list)


# ── Internal job record ─────────────────────────────────────────────


class Job(BaseModel):
    """Server-side job record (not exposed directly via the API)."""

    job_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    status: JobStatus = JobStatus.PENDING
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    original_image_b64: str = ""
    recommendations: list[Recommendation] = Field(default_factory=list)
    brand_guidelines: BrandGuidelines = Field(default_factory=BrandGuidelines)
    results: list[RecommendationResult] = Field(default_factory=list)
    error: str | None = None

    def to_response(self) -> JobResponse:
        """Project this internal record into the public API shape."""
        return JobResponse(
            job_id=self.job_id,
            status=self.status,
            created_at=self.created_at,
            results=self.results,
            error=self.error,
        )
