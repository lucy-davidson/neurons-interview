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
import json
import logging
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator

from app.config import settings
from app import db, metrics
from app.models import (
    Job,
    JobListResponse,
    JobRequest,
    JobResponse,
    JobStatus,
    JobSummary,
    RecommendationResult,
)
from app.services.image import validate_image
from app.services.llm import check_openai_health, check_gemini_health, check_claude_health
from app.workflow.graph import run_all_recommendations

# ── Structured logging ─────────────────────────────────────────────

# Write JSON logs to a persistent file (if /app/logs is mounted)
_log_handlers: list[logging.Handler] = [logging.StreamHandler()]  # stdout
_log_dir = "/app/logs"
import os as _os
if _os.path.isdir(_log_dir):
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

        await run_all_recommendations(
            image_b64=job.original_image_b64,
            recommendations=job.recommendations,
            brand_guidelines=job.brand_guidelines,
            max_attempts=settings.max_retries,
            job_results=job.results,
            on_variant_complete=lambda: _persist_results(job.job_id, job.results),
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
    # -- Validate image type ------------------------------------------------
    if image.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image type '{image.content_type}'. "
            f"Allowed: {ALLOWED_CONTENT_TYPES}",
        )

    image_data = await image.read()

    # -- Validate file size ------------------------------------------------
    if len(image_data) > settings.max_image_size_bytes:
        limit_mb = settings.max_image_size_bytes // (1024 * 1024)
        actual_mb = round(len(image_data) / (1024 * 1024), 1)
        raise HTTPException(
            status_code=400,
            detail=f"Image file size ({actual_mb} MB) exceeds {limit_mb} MB limit.",
        )

    # -- Validate image is decodable ---------------------------------------
    if not validate_image(image_data):
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")

    # -- Validate image dimensions -----------------------------------------
    import io as _io
    from PIL import Image as _PILImage
    try:
        _img = _PILImage.open(_io.BytesIO(image_data))
        w, h = _img.size
        if w > settings.max_image_dimension or h > settings.max_image_dimension:
            raise HTTPException(
                status_code=400,
                detail=f"Image dimensions ({w}x{h}) exceed {settings.max_image_dimension}px limit.",
            )
        # Warn about RGBA in response headers (non-blocking)
        _warnings: list[str] = []
        if _img.mode == "RGBA":
            _warnings.append("Image has an alpha channel (RGBA). Some image providers handle transparency inconsistently. Consider converting to RGB.")
    except HTTPException:
        raise
    except Exception:
        pass  # PIL validation already handled above

    # -- Parse and validate request body -----------------------------------
    body = await request_body.read()
    try:
        request = JobRequest.model_validate_json(body)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid request body: {exc}")

    if not request.recommendations:
        raise HTTPException(status_code=400, detail="At least one recommendation is required.")

    # -- Validate recommendation count ------------------------------------
    if len(request.recommendations) > settings.max_recommendations:
        raise HTTPException(
            status_code=400,
            detail=f"Too many recommendations ({len(request.recommendations)}). Maximum is {settings.max_recommendations}.",
        )

    # -- Validate recommendation text length ------------------------------
    for rec in request.recommendations:
        if len(rec.description) > settings.max_recommendation_text_length:
            raise HTTPException(
                status_code=400,
                detail=f"Recommendation '{rec.id}' description ({len(rec.description)} chars) "
                f"exceeds {settings.max_recommendation_text_length} character limit.",
            )

    image_b64 = base64.b64encode(image_data).decode()

    job = Job(
        original_image_b64=image_b64,
        recommendations=request.recommendations,
        brand_guidelines=request.brand_guidelines,
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
    """
    job = _jobs.get(job_id)
    if job:
        return job.to_response()

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
    if job and job.status in (JobStatus.PENDING, JobStatus.RUNNING):
        job.status = JobStatus.FAILED
        job.error = "Cancelled by user"
        if db.is_enabled():
            await db.update_job_status(job_id, "failed", error="Cancelled by user")
        logger.info("job_cancelled", job_id=job_id)
    return {"status": "cancelled"}


@app.post("/api/v1/jobs/{job_id}/feedback")
async def submit_feedback(job_id: str, body: dict) -> dict:
    """Record user feedback on a specific variant.

    Body: {"variant_id": "...", "feedback_type": "thumbs_up"|"thumbs_down"|"selected"|"refinement", "comment": "..."}
    """
    variant_id = body.get("variant_id", "")
    feedback_type = body.get("feedback_type", "")
    comment = body.get("comment")

    if feedback_type not in ("thumbs_up", "thumbs_down", "selected", "refinement"):
        raise HTTPException(status_code=400, detail=f"Invalid feedback_type: {feedback_type}")
    if not variant_id:
        raise HTTPException(status_code=400, detail="variant_id is required")

    # Look up variant context for the feedback record
    text_provider = image_provider = text_model = image_model = recommendation_type = None
    evaluation_score = None

    job = _jobs.get(job_id)
    if job:
        for rec in job.results:
            for v in rec.variants:
                if v.variant_id == variant_id:
                    text_provider = v.text_provider
                    image_provider = v.image_provider
                    text_model = v.text_model
                    image_model = v.image_model
                    evaluation_score = v.evaluation_score
                    recommendation_type = rec.recommendation_id
                    break

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
    import io as _io
    from PIL import Image as _PILImage
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
                img = _PILImage.open(_io.BytesIO(base64.b64decode(v.edited_image_b64)))
                img.thumbnail((300, 300))
                buf = _io.BytesIO()
                img.save(buf, format="JPEG", quality=50)
                return Response(content=buf.getvalue(), media_type="image/jpeg")

    raise HTTPException(status_code=404, detail="Variant not found")


# ── Static frontend ────────────────────────────────────────────────
import pathlib as _pathlib
_static_dir = _pathlib.Path(__file__).resolve().parent.parent / "static"
if _static_dir.is_dir():
    @app.get("/")
    async def root():
        return FileResponse(str(_static_dir / "index.html"))

    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")
