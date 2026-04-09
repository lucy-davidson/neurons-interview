"""FastAPI application -- async visual-recommendations backend.

Exposes a REST API for submitting image-recommendation jobs, polling
their status, and retrieving edited image variants.  Jobs are processed
asynchronously via a LangGraph agentic workflow.

Persistence: uses PostgreSQL when DATABASE_URL is set, otherwise falls
back to an in-memory dict (suitable for local development only).
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import pathlib
import random
import uuid
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator

from app.config import settings, ConfigOverrides, RuntimeConfig
from app import db, metrics
from app.models import (
    BrandGuidelines,
    Job,
    JobListResponse,
    JobRequest,
    JobResponse,
    JobStatus,
    JobSummary,
    Recommendation,
    RecommendationResult,
)
from PIL import Image as PILImage

from app.services.image import validate_image
from app.services.llm import check_openai_health, check_gemini_health, check_claude_health, get_rate_limit_status
from app.workflow.graph import run_all_recommendations

# ── Structured logging ─────────────────────────────────────────────

# Write JSON logs to a persistent file (if /app/logs is mounted)
_log_handlers: list[logging.Handler] = [logging.StreamHandler()]  # stdout
_log_dir = "/app/logs"
if os.path.isdir(_log_dir):
    _file_handler = logging.FileHandler(f"{_log_dir}/backend.log")
    _log_handlers.append(_file_handler)

logging.basicConfig(
    level=settings.log_level,
    handlers=_log_handlers,
    format="%(message)s",  # structlog handles formatting
)

structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer()
        if settings.log_level == "DEBUG"
        else structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(
        logging.getLevelName(settings.log_level)
    ),
    logger_factory=structlog.stdlib.LoggerFactory(),
)

logger = structlog.get_logger()

# ── In-memory fallback (used when DATABASE_URL is not set) ──────────

_jobs: dict[str, Job] = {}

# ── App lifecycle ───────────────────────────────────────────────────


JOB_HEARTBEAT_INTERVAL = 30  # seconds between heartbeat updates
JOB_TIMEOUT = 300  # seconds before a job with no heartbeat is considered dead
CLEANUP_INTERVAL = 60  # seconds between stale-job cleanup sweeps

_cleanup_task: asyncio.Task | None = None


async def _recover_orphaned_jobs() -> None:
    """Mark any 'running' or 'pending' jobs as 'failed' on startup.

    These are jobs from a previous process that died mid-flight.
    Their in-memory state is gone, so they can never complete.
    """
    if not db.is_enabled():
        return
    orphaned = await db.list_jobs()
    count = 0
    for j in orphaned:
        if j["status"] in ("running", "pending"):
            await db.update_job_status(
                j["job_id"], "failed",
                error="Server restarted while job was processing",
            )
            count += 1
    if count:
        logger.warning("recovered_orphaned_jobs", count=count)


DEPENDENCY_CHECK_INTERVAL = 30  # seconds between dependency health checks

_dependency_task: asyncio.Task | None = None


async def _dependency_health_loop() -> None:
    """Periodically check dependency health and update Prometheus gauges."""
    while True:
        try:
            openai_h, gemini_h, claude_h, db_h = await asyncio.gather(
                check_openai_health(),
                check_gemini_health(),
                check_claude_health(),
                db.check_health(),
            )
            for name, result in [
                ("openai", openai_h), ("gemini", gemini_h),
                ("claude", claude_h), ("postgresql", db_h),
            ]:
                is_up = 1.0 if result.get("status") == "healthy" else 0.0
                metrics.dependency_up.labels(dependency=name).set(is_up)
                metrics.dependency_latency.labels(dependency=name).set(result.get("latency_ms", 0))
        except Exception as exc:
            logger.error("dependency_health_check_error", error=str(exc))
        await asyncio.sleep(DEPENDENCY_CHECK_INTERVAL)


async def _stale_job_cleanup_loop() -> None:
    """Periodically sweep for jobs whose heartbeat has gone stale."""
    while True:
        await asyncio.sleep(CLEANUP_INTERVAL)
        try:
            await db.fail_stale_jobs(timeout_seconds=JOB_TIMEOUT)
        except Exception as exc:
            logger.error("cleanup_loop_error", error=str(exc))


async def _heartbeat_loop(job_id: str, stop_event: asyncio.Event) -> None:
    """Update the job's heartbeat timestamp until stop_event is set."""
    while not stop_event.is_set():
        await db.touch_heartbeat(job_id)
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=JOB_HEARTBEAT_INTERVAL)
        except asyncio.TimeoutError:
            pass  # Normal -- just loop and send another heartbeat


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise database on startup, close on shutdown."""
    global _cleanup_task, _dependency_task
    logger.info("backend_starting")
    await db.init_db()
    await _recover_orphaned_jobs()
    # Start background loops
    if db.is_enabled():
        _cleanup_task = asyncio.create_task(_stale_job_cleanup_loop())
    _dependency_task = asyncio.create_task(_dependency_health_loop())
    yield
    if _cleanup_task:
        _cleanup_task.cancel()
    if _dependency_task:
        _dependency_task.cancel()
    await db.close_db()
    logger.info("backend_shutdown")


app = FastAPI(
    title="Visual Recommendations API",
    version="1.0.0",
    description=(
        "Agentic workflow for generating visual variants of marketing "
        "creatives based on textual recommendations."
    ),
    lifespan=lifespan,
)

# Prometheus auto-instrumentation for all endpoints
Instrumentator().instrument(app).expose(app, endpoint="/metrics")

_cors_origins: list[str] = (
    ["*"] if settings.cors_origins == "*"
    else [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_CONTENT_TYPES: set[str] = {"image/png", "image/jpeg", "image/webp"}


# ── Shared validation helpers ──────────────────────────────────────


MAX_GUIDELINES_TEXT_LENGTH = 5000


def _validate_brand_guidelines(guidelines: BrandGuidelines) -> None:
    """Validate brand guidelines field lengths. Raises HTTPException on failure."""
    for field_name, value in [
        ("typography", guidelines.typography),
        ("aspect_ratio", guidelines.aspect_ratio),
        ("brand_elements", guidelines.brand_elements),
    ]:
        if len(value) > MAX_GUIDELINES_TEXT_LENGTH:
            raise HTTPException(
                status_code=400,
                detail=f"Brand guidelines '{field_name}' ({len(value)} chars) "
                f"exceeds {MAX_GUIDELINES_TEXT_LENGTH} character limit.",
            )
    for i, region in enumerate(guidelines.protected_regions):
        if len(region) > MAX_GUIDELINES_TEXT_LENGTH:
            raise HTTPException(
                status_code=400,
                detail=f"Protected region #{i + 1} ({len(region)} chars) "
                f"exceeds {MAX_GUIDELINES_TEXT_LENGTH} character limit.",
            )


def _validate_recommendations(recommendations: list[Recommendation]) -> None:
    """Validate recommendation fields. Raises HTTPException on failure."""
    seen_ids: set[str] = set()
    for rec in recommendations:
        # Whitespace-only / empty checks
        if not rec.id.strip():
            raise HTTPException(status_code=400, detail="Recommendation ID cannot be empty.")
        if not rec.title.strip():
            raise HTTPException(
                status_code=400,
                detail=f"Recommendation '{rec.id}' title cannot be empty.",
            )
        if not rec.description.strip():
            raise HTTPException(
                status_code=400,
                detail=f"Recommendation '{rec.id}' description cannot be empty.",
            )
        # Duplicate IDs
        if rec.id in seen_ids:
            raise HTTPException(
                status_code=400,
                detail=f"Duplicate recommendation ID '{rec.id}'. Each recommendation must have a unique ID.",
            )
        seen_ids.add(rec.id)
        # Text length
        if len(rec.description) > settings.max_recommendation_text_length:
            raise HTTPException(
                status_code=400,
                detail=f"Recommendation '{rec.id}' description ({len(rec.description)} chars) "
                f"exceeds {settings.max_recommendation_text_length} character limit.",
            )


async def _validate_image(image: UploadFile) -> tuple[bytes, list[str]]:
    """Validate an uploaded image file. Returns (image_bytes, warnings).

    Raises HTTPException on validation failure.
    """
    if image.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image type '{image.content_type}'. "
            f"Allowed: PNG, JPEG, WebP.",
        )

    image_data = await image.read()

    if len(image_data) == 0:
        raise HTTPException(status_code=400, detail="Image file is empty.")

    if len(image_data) > settings.max_image_size_bytes:
        limit_mb = settings.max_image_size_bytes // (1024 * 1024)
        actual_mb = round(len(image_data) / (1024 * 1024), 1)
        raise HTTPException(
            status_code=400,
            detail=f"Image file size ({actual_mb} MB) exceeds {limit_mb} MB limit.",
        )

    if not validate_image(image_data):
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")

    warnings: list[str] = []
    try:
        _img = PILImage.open(io.BytesIO(image_data))
        w, h = _img.size
        if w > settings.max_image_dimension or h > settings.max_image_dimension:
            raise HTTPException(
                status_code=400,
                detail=f"Image dimensions ({w}x{h}) exceed {settings.max_image_dimension}px limit.",
            )
        if _img.mode == "RGBA":
            warnings.append(
                "Image has an alpha channel (RGBA). Some image providers handle "
                "transparency inconsistently. Consider converting to RGB."
            )
    except HTTPException:
        raise
    except Exception:
        pass  # PIL validation already handled above

    return image_data, warnings


# ── Rollout traffic routing ─────────────────────────────────────────


def _pick_rollout_config(configs: list[dict]) -> dict | None:
    """Select a rollout config using weighted random selection."""
    if not configs:
        return None
    total = sum(c["weight"] for c in configs)
    if total <= 0:
        return None
    r = random.random() * total
    cumulative = 0.0
    for c in configs:
        cumulative += c["weight"]
        if r <= cumulative:
            return c
    return configs[-1]


# ── Helper: persist results after each variant ─────────────────────


def _results_to_dicts(results: list[RecommendationResult]) -> list[dict]:
    """Serialise the current results list to plain dicts for DB storage."""
    return [r.model_dump(mode="json") for r in results]


async def _persist_results(job_id: str, results: list[RecommendationResult]) -> None:
    """Write current results to the database (no-op if DB is disabled)."""
    if db.is_enabled():
        await db.update_job_results(job_id, _results_to_dicts(results))


# ── Background task runner ──────────────────────────────────────────


async def _process_job(job: Job) -> None:
    """Run the full agentic workflow in the background for *job*.

    Mutates *job* in-place AND persists state to the database so that
    both the polling endpoint and crash recovery work correctly.
    A heartbeat loop runs concurrently to signal liveness.
    """
    # Bind job_id to all log lines within this task
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(job_id=job.job_id)
    metrics.jobs_in_progress.inc()

    # Start heartbeat loop so the cleanup sweep knows we're alive
    heartbeat_stop = asyncio.Event()
    heartbeat_task = None
    if db.is_enabled():
        heartbeat_task = asyncio.create_task(_heartbeat_loop(job.job_id, heartbeat_stop))

    try:
        job.status = JobStatus.RUNNING
        if db.is_enabled():
            await db.update_job_status(job.job_id, "running")
            await db.touch_heartbeat(job.job_id)  # Initial heartbeat
        logger.info(
            "job_processing_started",
            num_recommendations=len(job.recommendations),
        )

        # Build runtime config from snapshot (per-job overrides) or global defaults
        rc = None
        if job.config_snapshot:
            rc = RuntimeConfig(ConfigOverrides(**{
                k: v for k, v in job.config_snapshot.items()
                if k in ConfigOverrides.model_fields
            }))
        else:
            rc = RuntimeConfig()

        # If rollout configs are active, pass them so each variant can
        # independently pick a config. This means a single job shows
        # variants from different models side by side.
        active_rollout = None
        if job.rollout_config_name and db.is_enabled():
            active_rollout = await db.get_active_rollout_configs()

        await run_all_recommendations(
            image_b64=job.original_image_b64,
            recommendations=job.recommendations,
            brand_guidelines=job.brand_guidelines,
            max_attempts=rc.max_retries,
            job_results=job.results,
            on_variant_complete=lambda: _persist_results(job.job_id, job.results),
            runtime_config=rc,
            rollout_configs=active_rollout,
        )
        job.status = JobStatus.COMPLETED
        metrics.jobs_completed.labels(status="completed").inc()
        if db.is_enabled():
            await db.update_job_status(job.job_id, "completed")
            await db.update_job_results(job.job_id, _results_to_dicts(job.results))
        logger.info("job_completed")

    except Exception as exc:
        logger.error("job_failed", error=str(exc), exc_info=True)
        job.status = JobStatus.FAILED
        job.error = str(exc)
        metrics.jobs_completed.labels(status="failed").inc()
        if db.is_enabled():
            await db.update_job_status(job.job_id, "failed", error=str(exc))

    finally:
        # Stop the heartbeat loop
        heartbeat_stop.set()
        if heartbeat_task:
            await heartbeat_task
        metrics.jobs_in_progress.dec()
        structlog.contextvars.clear_contextvars()


# ── Endpoints ───────────────────────────────────────────────────────


@app.get("/api/v1/jobs", response_model=JobListResponse)
async def list_jobs_endpoint() -> JobListResponse:
    """Return a lightweight summary of every known job."""
    if db.is_enabled():
        rows = await db.list_jobs()
        summaries = [
            JobSummary(
                job_id=r["job_id"],
                status=r["status"],
                created_at=r["created_at"],
                num_recommendations=r["num_recommendations"],
                error=r.get("error"),
            )
            for r in rows
        ]
    else:
        summaries = [
            JobSummary(
                job_id=job.job_id,
                status=job.status,
                created_at=job.created_at,
                num_recommendations=len(job.recommendations),
                error=job.error,
            )
            for job in _jobs.values()
        ]
    return JobListResponse(jobs=summaries)


@app.post("/api/v1/jobs", response_model=JobResponse, status_code=202)
async def create_job(
    image: UploadFile = File(...),
    request_body: UploadFile = File(...),
) -> JobResponse:
    """Submit a new visual-recommendation job.

    Accepts a multipart form with:
      - ``image``: the marketing creative (PNG/JPEG/WebP).
      - ``request_body``: a JSON file matching the ``JobRequest`` schema.

    Returns 202 immediately; the job is processed asynchronously.
    """
    # -- Validate image ------------------------------------------------------
    image_data, _warnings = await _validate_image(image)

    # -- Parse and validate request body -----------------------------------
    body = await request_body.read()
    try:
        request = JobRequest.model_validate_json(body)
    except Exception:
        raise HTTPException(
            status_code=422,
            detail="Invalid request body. Expected JSON with 'recommendations' (array) and 'brand_guidelines' (object).",
        )

    if not request.recommendations:
        raise HTTPException(status_code=400, detail="At least one recommendation is required.")

    # -- Validate recommendation count ------------------------------------
    if len(request.recommendations) > settings.max_recommendations:
        raise HTTPException(
            status_code=400,
            detail=f"Too many recommendations ({len(request.recommendations)}). Maximum is {settings.max_recommendations}.",
        )

    # -- Validate recommendation fields -----------------------------------
    _validate_recommendations(request.recommendations)
    _validate_brand_guidelines(request.brand_guidelines)

    image_b64 = base64.b64encode(image_data).decode()

    # Build runtime config from overrides (if any)
    overrides = None
    if request.config_overrides:
        try:
            overrides = ConfigOverrides(**{
                k: v for k, v in request.config_overrides.items()
                if k in ConfigOverrides.model_fields
            })
        except Exception:
            raise HTTPException(
                status_code=400,
                detail="Invalid config_overrides. Allowed fields: "
                + ", ".join(sorted(ConfigOverrides.model_fields.keys())),
            )
        errors = overrides.validate_values()
        if errors:
            raise HTTPException(status_code=400, detail="; ".join(errors))
    # Rollout routing: if no explicit overrides, pick from active rollout configs
    rollout_config_name = None
    if overrides is None and db.is_enabled():
        active_configs = await db.get_active_rollout_configs()
        if active_configs:
            selected = _pick_rollout_config(active_configs)
            if selected:
                rollout_config_name = selected["name"]
                overrides = ConfigOverrides(**{
                    k: v for k, v in selected["config"].items()
                    if k in ConfigOverrides.model_fields
                })
                logger.info("rollout_config_selected", config_name=rollout_config_name)

    rc = RuntimeConfig(overrides)
    config_snapshot = rc.snapshot()

    job = Job(
        original_image_b64=image_b64,
        recommendations=request.recommendations,
        brand_guidelines=request.brand_guidelines,
        config_snapshot=config_snapshot,
        rollout_config_name=rollout_config_name,
    )

    # Persist to both in-memory store and database
    _jobs[job.job_id] = job
    if db.is_enabled():
        await db.create_job(
            job_id=job.job_id,
            status=job.status.value,
            created_at=job.created_at,
            original_image_b64=image_b64,
            recommendations=[r.model_dump() for r in request.recommendations],
            brand_guidelines=request.brand_guidelines.model_dump(),
            config_snapshot=config_snapshot,
            rollout_config_name=rollout_config_name,
        )

    # Fire-and-forget: the polling endpoint exposes progress
    asyncio.create_task(_process_job(job))

    metrics.jobs_submitted.inc()
    log_kwargs: dict = {
        "job_id": job.job_id,
        "num_recommendations": len(request.recommendations),
    }
    if _warnings:
        log_kwargs["warnings"] = _warnings
    logger.info("job_created", **log_kwargs)

    resp = job.to_response()
    if _warnings:
        resp.warnings = _warnings
    return resp


@app.get("/api/v1/jobs/{job_id}", response_model=JobResponse)
async def get_job(job_id: str) -> JobResponse:
    """Poll job status and retrieve results (including partial results while running).

    Checks the in-memory store first (for active jobs), then falls back
    to the database (for jobs from previous sessions).
    Adds live warnings (e.g. rate limiting) for running jobs.
    """
    job = _jobs.get(job_id)
    if job:
        resp = job.to_response()
        # Add live warnings for running jobs
        if job.status in (JobStatus.RUNNING, JobStatus.PENDING):
            rl = get_rate_limit_status()
            if rl["any_limited"]:
                limited = [
                    f"{name} ({info['current_rpm']}/{info['default_rpm']} RPM)"
                    for name, info in rl["providers"].items()
                    if info["is_limited"]
                ]
                resp.warnings.append(
                    "Rate limited (" + ", ".join(limited) + ") — generation may take longer than usual."
                )
        # Add per-recommendation warnings for all-failed
        for rec in resp.results:
            variants = rec.variants or []
            if variants and all(v.status == "max_retries_exceeded" for v in variants):
                total_attempts = sum(v.attempts for v in variants)
                resp.warnings.append(
                    f'All {len(variants)} variant(s) for "{rec.recommendation_title}" '
                    f"failed after {total_attempts} total attempts. "
                    "Try rephrasing the recommendation to be more specific or visually concrete."
                )
        return resp

    if db.is_enabled():
        row = await db.get_job(job_id)
        if row:
            return JobResponse(
                job_id=row["job_id"],
                status=row["status"],
                created_at=row["created_at"],
                results=[
                    RecommendationResult.model_validate(r)
                    for r in row["results"]
                ],
                error=row.get("error"),
            )

    raise HTTPException(status_code=404, detail="Job not found")


@app.get("/api/v1/jobs/{job_id}/image/{variant_id}")
async def get_result_image(job_id: str, variant_id: str) -> JSONResponse:
    """Return the edited image for a specific variant as base-64 JSON."""
    job = _jobs.get(job_id)
    results = job.results if job else None

    if results is None and db.is_enabled():
        row = await db.get_job(job_id)
        if row and row["status"] == "completed":
            results = [
                RecommendationResult.model_validate(r)
                for r in row["results"]
            ]

    if results is None:
        raise HTTPException(status_code=404, detail="Job not found")

    for result in results:
        for variant in result.variants:
            if variant.variant_id == variant_id:
                if not variant.edited_image_b64:
                    raise HTTPException(status_code=404, detail="No image available for this variant")
                return JSONResponse(
                    content={"image_b64": variant.edited_image_b64},
                    media_type="application/json",
                )

    raise HTTPException(status_code=404, detail="Variant not found in job results")


@app.get("/api/v1/metrics/summary")
async def metrics_summary() -> dict:
    """Return a JSON summary of key system metrics for the monitor dashboard."""
    return {
        "jobs_submitted": metrics.jobs_submitted._value.get(),
        "jobs_completed": metrics.jobs_completed.labels(status="completed")._value.get(),
        "jobs_failed": metrics.jobs_completed.labels(status="failed")._value.get(),
        "jobs_in_progress": metrics.jobs_in_progress._value.get(),
        "variants_accepted": metrics.variants_processed.labels(status="accepted")._value.get(),
        "variants_failed": metrics.variants_processed.labels(status="max_retries_exceeded")._value.get(),
        "parse_failures": sum(
            metrics.llm_parse_failures.labels(agent=a)._value.get()
            for a in ["ideator", "idea_critic", "critic"]
        ),
        "agent_invocations": {
            agent: metrics.agent_invocations.labels(agent=agent)._value.get()
            for agent in ["ideator", "idea_critic", "editor", "critic", "refiner"]
        },
    }


@app.post("/api/v1/jobs/{job_id}/cancel")
async def cancel_job(job_id: str) -> dict:
    """Mark a job as cancelled. The background task will check this and stop."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job.status == JobStatus.COMPLETED:
        return {"status": "already_completed", "message": "Job has already completed and cannot be cancelled."}
    if job.status == JobStatus.FAILED:
        return {"status": "already_failed", "message": "Job has already failed."}
    job.status = JobStatus.FAILED
    job.error = "Cancelled by user"
    if db.is_enabled():
        await db.update_job_status(job_id, "failed", error="Cancelled by user")
    logger.info("job_cancelled", job_id=job_id)
    return {"status": "cancelled"}


MAX_FEEDBACK_COMMENT_LENGTH = 2000


@app.post("/api/v1/jobs/{job_id}/feedback")
async def submit_feedback(job_id: str, body: dict) -> dict:
    """Record user feedback on a specific variant.

    Body: {"variant_id": "...", "feedback_type": "thumbs_up"|"thumbs_down"|"selected"|"refinement", "comment": "..."}
    """
    variant_id = body.get("variant_id", "")
    feedback_type = body.get("feedback_type", "")
    comment = body.get("comment")

    if feedback_type not in ("thumbs_up", "thumbs_down", "selected", "refinement"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid feedback_type '{feedback_type}'. "
            "Must be one of: thumbs_up, thumbs_down, selected, refinement.",
        )
    if not variant_id:
        raise HTTPException(status_code=400, detail="variant_id is required.")
    if comment and len(comment) > MAX_FEEDBACK_COMMENT_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Comment too long ({len(comment)} chars). Maximum is {MAX_FEEDBACK_COMMENT_LENGTH}.",
        )

    # Look up the job — check in-memory first, then DB
    job = _jobs.get(job_id)
    if not job and db.is_enabled():
        row = await db.get_job(job_id)
        if not row:
            raise HTTPException(status_code=404, detail="Job not found.")
    elif not job:
        raise HTTPException(status_code=404, detail="Job not found.")

    # Find the variant and extract context for the feedback record
    text_provider = image_provider = text_model = image_model = recommendation_type = None
    evaluation_score = None
    variant_found = False

    results = job.results if job else [
        RecommendationResult.model_validate(r) for r in (row.get("results") or [])
    ]
    for rec in results:
        for v in rec.variants:
            if v.variant_id == variant_id:
                variant_found = True
                text_provider = v.text_provider
                image_provider = v.image_provider
                text_model = v.text_model
                image_model = v.image_model
                evaluation_score = v.evaluation_score
                recommendation_type = rec.recommendation_id
                break

    if not variant_found:
        raise HTTPException(
            status_code=404,
            detail=f"Variant '{variant_id}' not found in job '{job_id}'.",
        )

    if db.is_enabled():
        await db.save_feedback(
            job_id=job_id,
            variant_id=variant_id,
            feedback_type=feedback_type,
            comment=comment,
            text_provider=text_provider,
            image_provider=image_provider,
            text_model=text_model,
            image_model=image_model,
            evaluation_score=evaluation_score,
            recommendation_type=recommendation_type,
        )

    logger.info(
        "feedback_recorded",
        job_id=job_id,
        variant_id=variant_id,
        feedback_type=feedback_type,
    )
    return {"status": "recorded"}


@app.get("/api/v1/feedback/stats")
async def feedback_stats() -> dict:
    """Return aggregate feedback statistics."""
    if db.is_enabled():
        return await db.get_feedback_stats()
    return {}


@app.get("/api/v1/feedback/calibration")
async def feedback_calibration() -> dict:
    """Analyse how well the critic's scores correlate with user feedback.

    Returns score distributions by sentiment, per-provider satisfaction
    rates, threshold analysis, and per-recommendation-type breakdown.
    """
    if not db.is_enabled():
        raise HTTPException(status_code=503, detail="Database not configured. Calibration requires stored feedback data.")
    result = await db.get_feedback_calibration()
    if not result:
        return {
            "score_distribution": {},
            "provider_satisfaction": [],
            "threshold_analysis": [],
            "by_recommendation_type": [],
            "total_feedback_with_scores": 0,
            "message": "No feedback data with evaluation scores found.",
        }
    return result


@app.get("/api/v1/experiments/comparison")
async def provider_comparison() -> dict:
    """Aggregate variant outcomes by provider for experiment comparison.

    Scans all completed jobs and groups variants by their text_provider
    and image_provider to show comparative performance.
    """
    from collections import defaultdict

    # Gather all variants from in-memory jobs and database
    all_variants: list[dict] = []
    for job in _jobs.values():
        if job.status != JobStatus.COMPLETED:
            continue
        for rec in job.results:
            for v in rec.variants:
                all_variants.append(v.model_dump())

    if db.is_enabled():
        db_jobs = await db.list_jobs()
        for summary in db_jobs:
            if summary["status"] != "completed":
                continue
            if summary["job_id"] in _jobs:
                continue  # Already counted from in-memory
            row = await db.get_job(summary["job_id"])
            if row:
                for rec in row["results"]:
                    for v in rec.get("variants", []):
                        all_variants.append(v)

    # Aggregate by text provider
    by_text: dict[str, dict] = defaultdict(lambda: {
        "count": 0, "accepted": 0, "total_score": 0.0, "total_attempts": 0,
        "scores": [],
    })
    by_image: dict[str, dict] = defaultdict(lambda: {
        "count": 0, "accepted": 0, "total_score": 0.0, "total_attempts": 0,
    })

    for v in all_variants:
        tp = v.get("text_provider", "unknown") or "unknown"
        ip = v.get("image_provider", "unknown") or "unknown"
        score = v.get("evaluation_score") or 0.0
        attempts = v.get("attempts", 1)
        accepted = v.get("status") == "accepted"

        by_text[tp]["count"] += 1
        by_text[tp]["accepted"] += int(accepted)
        by_text[tp]["total_score"] += score
        by_text[tp]["total_attempts"] += attempts
        by_text[tp]["scores"].append(score)

        by_image[ip]["count"] += 1
        by_image[ip]["accepted"] += int(accepted)
        by_image[ip]["total_score"] += score
        by_image[ip]["total_attempts"] += attempts

    # Compute averages
    result = {"total_variants": len(all_variants), "by_text_provider": {}, "by_image_provider": {}}

    for provider, data in by_text.items():
        n = data["count"]
        scores = data["scores"]
        result["by_text_provider"][provider] = {
            "variants": n,
            "accepted": data["accepted"],
            "acceptance_rate": round(data["accepted"] / n, 2) if n else 0,
            "avg_score": round(data["total_score"] / n, 3) if n else 0,
            "avg_attempts": round(data["total_attempts"] / n, 1) if n else 0,
            "min_score": round(min(scores), 3) if scores else 0,
            "max_score": round(max(scores), 3) if scores else 0,
        }

    for provider, data in by_image.items():
        n = data["count"]
        result["by_image_provider"][provider] = {
            "variants": n,
            "accepted": data["accepted"],
            "acceptance_rate": round(data["accepted"] / n, 2) if n else 0,
            "avg_score": round(data["total_score"] / n, 3) if n else 0,
            "avg_attempts": round(data["total_attempts"] / n, 1) if n else 0,
        }

    return result


@app.get("/api/v1/health/dependencies")
async def dependency_health() -> dict:
    """Check connectivity to all external dependencies.

    Tests each provider and the database with a lightweight call,
    returning status and latency for each.
    """
    import asyncio
    openai_check, gemini_check, claude_check, db_check = await asyncio.gather(
        check_openai_health(),
        check_gemini_health(),
        check_claude_health(),
        db.check_health(),
    )

    def _role(provider_name: str) -> str:
        roles = []
        if settings.text_provider == provider_name:
            roles.append("text/vision")
        if settings.image_provider == provider_name:
            roles.append("image")
        return ", ".join(roles) if roles else "fallback"

    deps = {
        "openai": {
            **openai_check,
            "role": _role("openai"),
            "model": settings.openai_vision_model if settings.text_provider == "openai" else settings.openai_image_model,
        },
        "gemini": {
            **gemini_check,
            "role": _role("gemini"),
            "model": settings.gemini_image_model if settings.image_provider == "gemini" else settings.gemini_vision_model,
        },
        "claude": {
            **claude_check,
            "role": _role("claude"),
            "model": settings.claude_vision_model,
        },
        "postgresql": db_check,
    }

    all_healthy = all(
        d.get("status") in ("healthy", "not_configured", "inactive")
        for d in deps.values()
    )
    deps["overall"] = "healthy" if all_healthy else "degraded"

    return deps


@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness probe -- includes database status when configured."""
    result = {"status": "ok"}
    if settings.database_url:
        result["database"] = "connected" if db.is_enabled() else "disconnected"
    return result


@app.get("/api/v1/thumb/{job_id}/{variant_id}")
async def get_thumbnail(job_id: str, variant_id: str):
    """Return a small JPEG thumbnail for a variant image."""
    from fastapi.responses import Response

    job = _jobs.get(job_id)
    results = job.results if job else None

    if results is None and db.is_enabled():
        row = await db.get_job(job_id)
        if row:
            results = [RecommendationResult.model_validate(r) for r in row["results"]]

    if results is None:
        raise HTTPException(status_code=404, detail="Job not found")

    for rec in results:
        for v in rec.variants:
            if v.variant_id == variant_id and v.edited_image_b64:
                img = PILImage.open(io.BytesIO(base64.b64decode(v.edited_image_b64)))
                img.thumbnail((300, 300))
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=50)
                return Response(content=buf.getvalue(), media_type="image/jpeg")

    raise HTTPException(status_code=404, detail="Variant not found")


# ── Experiment endpoints ────────────────────────────────────────────


@app.post("/api/v1/experiments", status_code=201)
async def create_experiment(body: dict) -> dict:
    """Create a new experiment with named config variations.

    Body: {"name": "...", "description": "...", "base_config": {...},
           "variations": [{"name": "...", "config": {...}}, ...]}
    """
    name = (body.get("name") or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="Experiment name is required.")
    variations = body.get("variations", [])
    if not variations:
        raise HTTPException(status_code=400, detail="At least one variation is required.")
    seen_names: set[str] = set()
    for v in variations:
        vname = (v.get("name") or "").strip()
        if not vname:
            raise HTTPException(status_code=400, detail="Each variation must have a name.")
        if vname in seen_names:
            raise HTTPException(status_code=400, detail=f"Duplicate variation name '{vname}'.")
        seen_names.add(vname)
        # Validate config overrides if present
        config = v.get("config", {})
        if config:
            try:
                ov = ConfigOverrides(**{k: val for k, val in config.items() if k in ConfigOverrides.model_fields})
                errors = ov.validate_values()
                if errors:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Variation '{vname}': {'; '.join(errors)}",
                    )
            except HTTPException:
                raise
            except Exception:
                pass  # Unknown fields are ignored

    experiment_id = uuid.uuid4().hex
    base_config = body.get("base_config", {})
    description = body.get("description")

    if db.is_enabled():
        await db.create_experiment(
            experiment_id=experiment_id,
            name=name,
            description=description,
            base_config=base_config,
            variations=variations,
        )

    logger.info("experiment_created", experiment_id=experiment_id, name=name, num_variations=len(variations))
    return {
        "experiment_id": experiment_id,
        "name": name,
        "variations": len(variations),
    }


@app.get("/api/v1/experiments")
async def list_experiments() -> dict:
    """List all experiments."""
    if db.is_enabled():
        experiments = await db.list_experiments()
        return {"experiments": experiments}
    return {"experiments": []}


@app.get("/api/v1/experiments/{experiment_id}")
async def get_experiment(experiment_id: str) -> dict:
    """Get experiment details including config variations."""
    if not db.is_enabled():
        raise HTTPException(status_code=404, detail="Database not configured")
    exp = await db.get_experiment(experiment_id)
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return exp


@app.post("/api/v1/experiments/{experiment_id}/run", status_code=202)
async def run_experiment(
    experiment_id: str,
    image: UploadFile = File(...),
    request_body: UploadFile = File(...),
) -> dict:
    """Run all variations of an experiment with the same input.

    Creates one job per variation, all tagged with the experiment ID.
    Accepts the same multipart form as POST /api/v1/jobs.
    """
    if not db.is_enabled():
        raise HTTPException(status_code=400, detail="Database required for experiments")

    exp = await db.get_experiment(experiment_id)
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found.")
    if exp["status"] == "running":
        raise HTTPException(status_code=409, detail="Experiment is already running.")

    # Validate image (same checks as create_job)
    image_data, _ = await _validate_image(image)

    body = await request_body.read()
    try:
        request = JobRequest.model_validate_json(body)
    except Exception:
        raise HTTPException(
            status_code=422,
            detail="Invalid request body. Expected JSON with 'recommendations' (array) and 'brand_guidelines' (object).",
        )
    if not request.recommendations:
        raise HTTPException(status_code=400, detail="At least one recommendation is required.")
    _validate_recommendations(request.recommendations)
    _validate_brand_guidelines(request.brand_guidelines)

    image_b64 = base64.b64encode(image_data).decode()
    base_config = exp.get("base_config", {})
    job_ids = []

    for variation in exp["variations"]:
        # Merge base_config with variation-specific overrides
        merged = {**base_config, **variation.get("config", {})}
        overrides = ConfigOverrides(**{k: v for k, v in merged.items() if k in ConfigOverrides.model_fields})
        rc = RuntimeConfig(overrides)
        config_snapshot = rc.snapshot()

        job = Job(
            original_image_b64=image_b64,
            recommendations=request.recommendations,
            brand_guidelines=request.brand_guidelines,
            config_snapshot=config_snapshot,
            experiment_id=experiment_id,
            variation_name=variation["name"],
        )

        _jobs[job.job_id] = job
        await db.create_job(
            job_id=job.job_id,
            status=job.status.value,
            created_at=job.created_at,
            original_image_b64=image_b64,
            recommendations=[r.model_dump() for r in request.recommendations],
            brand_guidelines=request.brand_guidelines.model_dump(),
            config_snapshot=config_snapshot,
            experiment_id=experiment_id,
            variation_name=variation["name"],
        )

        asyncio.create_task(_process_job(job))
        metrics.jobs_submitted.inc()
        job_ids.append({"job_id": job.job_id, "variation": variation["name"]})

    await db.update_experiment_status(experiment_id, "running")
    logger.info("experiment_started", experiment_id=experiment_id, num_jobs=len(job_ids))

    return {
        "experiment_id": experiment_id,
        "jobs": job_ids,
    }


@app.get("/api/v1/experiments/{experiment_id}/results")
async def get_experiment_results(experiment_id: str) -> dict:
    """Compare results across all variations of an experiment.

    Returns per-variation aggregates: acceptance rate, average score,
    retry counts, and job status.
    """
    if not db.is_enabled():
        raise HTTPException(status_code=400, detail="Database required for experiments")

    exp = await db.get_experiment(experiment_id)
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")

    jobs = await db.get_experiment_jobs(experiment_id)

    # Group jobs by variation
    from collections import defaultdict
    by_variation: dict[str, list] = defaultdict(list)
    for j in jobs:
        by_variation[j.get("variation_name", "unknown")].append(j)

    variations = []
    all_done = True
    for var_name, var_jobs in by_variation.items():
        total_variants = 0
        accepted = 0
        total_score = 0.0
        total_attempts = 0
        total_duration = 0.0
        total_tokens = 0
        total_cost = 0.0
        jobs_completed = 0
        jobs_failed = 0
        jobs_running = 0
        prompt_versions = {}

        for j in var_jobs:
            if j["status"] in ("pending", "running"):
                jobs_running += 1
                all_done = False
            elif j["status"] == "completed":
                jobs_completed += 1
            elif j["status"] == "failed":
                jobs_failed += 1

            for rec in (j.get("results") or []):
                for v in rec.get("variants", []):
                    total_variants += 1
                    if v.get("status") == "accepted":
                        accepted += 1
                    total_score += v.get("evaluation_score") or 0.0
                    total_attempts += v.get("attempts", 1)
                    total_duration += v.get("duration_s") or 0.0
                    total_tokens += v.get("total_tokens") or 0
                    total_cost += v.get("total_cost_usd") or 0.0
                    if v.get("prompt_versions") and not prompt_versions:
                        prompt_versions = v["prompt_versions"]

        variations.append({
            "name": var_name,
            "jobs": len(var_jobs),
            "jobs_completed": jobs_completed,
            "jobs_failed": jobs_failed,
            "jobs_running": jobs_running,
            "variants_total": total_variants,
            "variants_accepted": accepted,
            "acceptance_rate": round(accepted / total_variants, 3) if total_variants else 0,
            "avg_score": round(total_score / total_variants, 3) if total_variants else 0,
            "avg_attempts": round(total_attempts / total_variants, 1) if total_variants else 0,
            "avg_duration_s": round(total_duration / total_variants, 1) if total_variants else 0,
            "total_tokens": total_tokens,
            "total_cost_usd": round(total_cost, 4),
            "prompt_versions": prompt_versions,
        })

    # Update experiment status if all jobs are done
    if all_done and jobs:
        await db.update_experiment_status(experiment_id, "completed")

    return {
        "experiment_id": experiment_id,
        "name": exp["name"],
        "status": "completed" if (all_done and jobs) else "running" if jobs else "pending",
        "variations": variations,
    }


# ── Rollout config endpoints ────────────────────────────────────────


@app.post("/api/v1/rollout/configs", status_code=201)
async def create_rollout_config_endpoint(body: dict) -> dict:
    """Create or update a rollout config with a traffic weight.

    Body: {"name": "...", "config": {...}, "weight": 0.9, "active": true}
    """
    name = (body.get("name") or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="Config name is required.")
    weight = body.get("weight", 1.0)
    if not isinstance(weight, (int, float)) or weight < 0 or weight > 1.0:
        raise HTTPException(status_code=400, detail="Weight must be between 0.0 and 1.0.")
    config = body.get("config", {})
    if config:
        try:
            ov = ConfigOverrides(**{k: v for k, v in config.items() if k in ConfigOverrides.model_fields})
            errors = ov.validate_values()
            if errors:
                raise HTTPException(status_code=400, detail="; ".join(errors))
        except HTTPException:
            raise
        except Exception:
            pass
    active = body.get("active", True)
    if not db.is_enabled():
        raise HTTPException(status_code=503, detail="Database required for rollout configs.")
    await db.create_rollout_config(name=name, config_json=config, weight=weight, active=active)
    logger.info("rollout_config_created", name=name, weight=weight)
    return {"name": name, "weight": weight, "active": active, "config": config}


@app.get("/api/v1/rollout/configs")
async def list_rollout_configs_endpoint() -> dict:
    """List all rollout configs and their weights."""
    if not db.is_enabled():
        return {"configs": [], "total_weight": 0}
    configs = await db.list_rollout_configs()
    total_weight = sum(c["weight"] for c in configs if c.get("active"))
    return {"configs": configs, "total_weight": round(total_weight, 3)}


@app.put("/api/v1/rollout/configs/{name}")
async def update_rollout_config_endpoint(name: str, body: dict) -> dict:
    """Update a rollout config's weight and/or active status."""
    if not db.is_enabled():
        raise HTTPException(status_code=503, detail="Database required for rollout configs.")
    weight = body.get("weight")
    active = body.get("active")
    if weight is not None and (not isinstance(weight, (int, float)) or weight < 0 or weight > 1.0):
        raise HTTPException(status_code=400, detail="Weight must be between 0.0 and 1.0.")
    if weight is None and active is None:
        raise HTTPException(status_code=400, detail="Provide 'weight' or 'active' to update.")
    found = await db.update_rollout_config(name, weight=weight, active=active)
    if not found:
        raise HTTPException(status_code=404, detail=f"Rollout config '{name}' not found.")
    logger.info("rollout_config_updated", name=name, weight=weight, active=active)
    return {"name": name, "updated": True}


@app.delete("/api/v1/rollout/configs/{name}")
async def delete_rollout_config_endpoint(name: str) -> dict:
    """Remove a rollout config."""
    if not db.is_enabled():
        raise HTTPException(status_code=503, detail="Database required for rollout configs.")
    found = await db.delete_rollout_config(name)
    if not found:
        raise HTTPException(status_code=404, detail=f"Rollout config '{name}' not found.")
    logger.info("rollout_config_deleted", name=name)
    return {"name": name, "deleted": True}


@app.get("/api/v1/rollout/performance")
async def rollout_performance() -> dict:
    """Per-rollout-config feedback and performance stats."""
    if not db.is_enabled():
        raise HTTPException(status_code=503, detail="Database required for rollout performance.")
    stats = await db.get_rollout_performance()
    return {"configs": stats}


# ── Static frontend ────────────────────────────────────────────────
_static_dir = pathlib.Path(__file__).resolve().parent.parent / "static"
if _static_dir.is_dir():
    @app.get("/")
    async def root():
        return FileResponse(str(_static_dir / "index.html"))

    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")
