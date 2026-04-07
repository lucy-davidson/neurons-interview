"""PostgreSQL persistence layer for job state.

Uses asyncpg for non-blocking database access.  Jobs are stored with
their full state as JSONB, enabling incremental updates as variants
complete while keeping the schema simple.

All operations are wrapped in try/except so database failures never
crash the application -- they log the error and degrade gracefully.
"""

from __future__ import annotations

import json
from typing import Any

import asyncpg
import structlog

from app.config import settings

logger = structlog.get_logger()

_pool: asyncpg.Pool | None = None

# ── Schema ─────────────────────────────────────────────────────────

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id          TEXT PRIMARY KEY,
    status          TEXT NOT NULL DEFAULT 'pending',
    created_at      TEXT NOT NULL,
    original_image_b64 TEXT NOT NULL DEFAULT '',
    recommendations JSONB NOT NULL DEFAULT '[]',
    brand_guidelines JSONB NOT NULL DEFAULT '{}',
    results         JSONB NOT NULL DEFAULT '[]',
    error           TEXT,
    last_heartbeat  TIMESTAMPTZ
);
"""

_MIGRATION_SQL = """
ALTER TABLE jobs ADD COLUMN IF NOT EXISTS last_heartbeat TIMESTAMPTZ;
ALTER TABLE jobs ADD COLUMN IF NOT EXISTS config_snapshot JSONB;
ALTER TABLE jobs ADD COLUMN IF NOT EXISTS experiment_id TEXT;
ALTER TABLE jobs ADD COLUMN IF NOT EXISTS variation_name TEXT;
ALTER TABLE feedback ADD COLUMN IF NOT EXISTS text_model TEXT;
ALTER TABLE feedback ADD COLUMN IF NOT EXISTS image_model TEXT;
"""

_EXPERIMENTS_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS experiments (
    experiment_id   TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    description     TEXT,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    status          TEXT NOT NULL DEFAULT 'pending',
    base_config     JSONB NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS experiment_configs (
    id              SERIAL PRIMARY KEY,
    experiment_id   TEXT NOT NULL REFERENCES experiments(experiment_id),
    variation_name  TEXT NOT NULL,
    config_json     JSONB NOT NULL DEFAULT '{}'
);
"""

_FEEDBACK_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS feedback (
    id                  SERIAL PRIMARY KEY,
    job_id              TEXT NOT NULL,
    variant_id          TEXT NOT NULL,
    feedback_type       TEXT NOT NULL,
    comment             TEXT,
    created_at          TIMESTAMPTZ DEFAULT NOW(),
    text_provider       TEXT,
    image_provider      TEXT,
    text_model          TEXT,
    image_model         TEXT,
    evaluation_score    FLOAT,
    recommendation_type TEXT
);
"""


# ── Connection lifecycle ───────────────────────────────────────────


async def init_db() -> None:
    """Create the connection pool and ensure the schema exists."""
    global _pool
    if not settings.database_url:
        logger.info("db_disabled", reason="DATABASE_URL not set")
        return
    dsn = settings.database_url
    try:
        logger.info("db_connecting", host=dsn.split("@")[-1])
        _pool = await asyncpg.create_pool(dsn, min_size=2, max_size=10)
        async with _pool.acquire() as conn:
            await conn.execute(_SCHEMA_SQL)
            await conn.execute(_FEEDBACK_SCHEMA_SQL)
            await conn.execute(_EXPERIMENTS_SCHEMA_SQL)
            await conn.execute(_MIGRATION_SQL)
        logger.info("db_initialised")
    except Exception as exc:
        logger.error("db_connection_failed", error=str(exc), exc_info=True)
        _pool = None  # Fall back to in-memory mode


async def close_db() -> None:
    """Gracefully shut down the connection pool."""
    global _pool
    if _pool:
        await _pool.close()
        _pool = None


def is_enabled() -> bool:
    """Return True if the database is configured and connected."""
    return _pool is not None


# ── Health check ───────────────────────────────────────────────────


async def check_health() -> dict:
    """Test database connectivity and return status with latency."""
    import time
    if not _pool:
        return {"status": "not_configured", "latency_ms": 0}
    start = time.monotonic()
    try:
        async with _pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        latency = round((time.monotonic() - start) * 1000)
        return {"status": "healthy", "latency_ms": latency}
    except Exception as exc:
        latency = round((time.monotonic() - start) * 1000)
        return {"status": "error", "latency_ms": latency, "error": str(exc)[:200]}


# ── Heartbeat ──────────────────────────────────────────────────────


async def touch_heartbeat(job_id: str) -> None:
    """Update the heartbeat timestamp to signal the job is still alive."""
    if not _pool:
        return
    try:
        await _pool.execute(
            "UPDATE jobs SET last_heartbeat = NOW() WHERE job_id = $1",
            job_id,
        )
    except Exception as exc:
        logger.error("db_heartbeat_failed", job_id=job_id, error=str(exc))


async def fail_stale_jobs(timeout_seconds: int = 300) -> int:
    """Mark running/pending jobs as failed if their heartbeat is stale.

    Returns the number of jobs marked as failed.
    """
    if not _pool:
        return 0
    try:
        result = await _pool.execute(
            """
            UPDATE jobs
            SET status = 'failed',
                error = 'Job timed out (no heartbeat for ' || $1 || ' seconds)'
            WHERE status IN ('running', 'pending')
              AND last_heartbeat IS NOT NULL
              AND last_heartbeat < NOW() - INTERVAL '1 second' * $1
            """,
            timeout_seconds,
        )
        # result is like "UPDATE N"
        count = int(result.split()[-1])
        if count:
            logger.warning("stale_jobs_recovered", count=count, timeout_seconds=timeout_seconds)
        return count
    except Exception as exc:
        logger.error("db_fail_stale_jobs_error", error=str(exc))
        return 0


# ── CRUD operations ────────────────────────────────────────────────


async def create_job(
    job_id: str,
    status: str,
    created_at: str,
    original_image_b64: str,
    recommendations: list[dict[str, Any]],
    brand_guidelines: dict[str, Any],
    config_snapshot: dict[str, Any] | None = None,
    experiment_id: str | None = None,
    variation_name: str | None = None,
) -> None:
    """Insert a new job row."""
    if not _pool:
        return
    try:
        await _pool.execute(
            """
            INSERT INTO jobs (job_id, status, created_at, original_image_b64,
                              recommendations, brand_guidelines, results,
                              config_snapshot, experiment_id, variation_name)
            VALUES ($1, $2, $3, $4, $5::jsonb, $6::jsonb, '[]'::jsonb,
                    $7::jsonb, $8, $9)
            """,
            job_id,
            status,
            created_at,
            original_image_b64,
            json.dumps(recommendations),
            json.dumps(brand_guidelines),
            json.dumps(config_snapshot) if config_snapshot else None,
            experiment_id,
            variation_name,
        )
    except Exception as exc:
        logger.error("db_create_job_failed", job_id=job_id, error=str(exc))


async def update_job_status(job_id: str, status: str, error: str | None = None) -> None:
    """Update a job's status (and optional error message)."""
    if not _pool:
        return
    try:
        await _pool.execute(
            "UPDATE jobs SET status = $1, error = $2 WHERE job_id = $3",
            status,
            error,
            job_id,
        )
    except Exception as exc:
        logger.error("db_update_status_failed", job_id=job_id, status=status, error=str(exc))


async def update_job_results(job_id: str, results: list[dict[str, Any]]) -> None:
    """Overwrite the results JSONB column with the current state."""
    if not _pool:
        return
    try:
        await _pool.execute(
            "UPDATE jobs SET results = $1::jsonb WHERE job_id = $2",
            json.dumps(results),
            job_id,
        )
    except Exception as exc:
        logger.error("db_update_results_failed", job_id=job_id, error=str(exc))


async def get_job(job_id: str) -> dict[str, Any] | None:
    """Fetch a single job by ID. Returns None if not found or on error."""
    if not _pool:
        return None
    try:
        row = await _pool.fetchrow("SELECT * FROM jobs WHERE job_id = $1", job_id)
        if not row:
            return None
        return _row_to_dict(row)
    except Exception as exc:
        logger.error("db_get_job_failed", job_id=job_id, error=str(exc))
        return None


async def list_jobs() -> list[dict[str, Any]]:
    """Return lightweight summaries for all jobs (no images or results)."""
    if not _pool:
        return []
    try:
        rows = await _pool.fetch(
            """
            SELECT job_id, status, created_at,
                   jsonb_array_length(recommendations) AS num_recommendations,
                   error, last_heartbeat
            FROM jobs
            ORDER BY created_at DESC
            """
        )
        return [
            {
                "job_id": r["job_id"],
                "status": r["status"],
                "created_at": r["created_at"],
                "num_recommendations": r["num_recommendations"],
                "error": r["error"],
                "last_heartbeat": r["last_heartbeat"].isoformat() if r["last_heartbeat"] else None,
            }
            for r in rows
        ]
    except Exception as exc:
        logger.error("db_list_jobs_failed", error=str(exc))
        return []


# ── Feedback operations ────────────────────────────────────────────


async def save_feedback(
    job_id: str,
    variant_id: str,
    feedback_type: str,
    comment: str | None = None,
    text_provider: str | None = None,
    image_provider: str | None = None,
    text_model: str | None = None,
    image_model: str | None = None,
    evaluation_score: float | None = None,
    recommendation_type: str | None = None,
) -> None:
    """Insert a feedback record."""
    if not _pool:
        return
    try:
        await _pool.execute(
            """
            INSERT INTO feedback (job_id, variant_id, feedback_type, comment,
                                  text_provider, image_provider, text_model, image_model,
                                  evaluation_score, recommendation_type)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            """,
            job_id, variant_id, feedback_type, comment,
            text_provider, image_provider, text_model, image_model,
            evaluation_score, recommendation_type,
        )
    except Exception as exc:
        logger.error("db_save_feedback_failed", job_id=job_id, variant_id=variant_id, error=str(exc))


async def get_feedback_for_job(job_id: str) -> list[dict[str, Any]]:
    """Return all feedback entries for a job."""
    if not _pool:
        return []
    try:
        rows = await _pool.fetch(
            "SELECT * FROM feedback WHERE job_id = $1 ORDER BY created_at",
            job_id,
        )
        return [
            {
                "id": r["id"],
                "job_id": r["job_id"],
                "variant_id": r["variant_id"],
                "feedback_type": r["feedback_type"],
                "comment": r["comment"],
                "created_at": r["created_at"].isoformat() if r["created_at"] else None,
                "text_provider": r["text_provider"],
                "image_provider": r["image_provider"],
                "evaluation_score": r["evaluation_score"],
                "recommendation_type": r["recommendation_type"],
            }
            for r in rows
        ]
    except Exception as exc:
        logger.error("db_get_feedback_failed", job_id=job_id, error=str(exc))
        return []


async def get_feedback_stats() -> dict[str, Any]:
    """Return aggregate feedback statistics across all jobs."""
    if not _pool:
        return {}
    try:
        row = await _pool.fetchrow("""
            SELECT
                COUNT(*) AS total,
                COUNT(*) FILTER (WHERE feedback_type = 'thumbs_up') AS thumbs_up,
                COUNT(*) FILTER (WHERE feedback_type = 'thumbs_down') AS thumbs_down,
                COUNT(*) FILTER (WHERE feedback_type = 'selected') AS selected,
                COUNT(*) FILTER (WHERE feedback_type = 'refinement') AS refinements,
                COUNT(DISTINCT job_id) AS jobs_with_feedback
            FROM feedback
        """)
        return {
            "total": row["total"],
            "thumbs_up": row["thumbs_up"],
            "thumbs_down": row["thumbs_down"],
            "selected": row["selected"],
            "refinements": row["refinements"],
            "jobs_with_feedback": row["jobs_with_feedback"],
        }
    except Exception as exc:
        logger.error("db_get_feedback_stats_failed", error=str(exc))
        return {}


# ── Experiment operations ──────────────────────────────────────────


async def create_experiment(
    experiment_id: str,
    name: str,
    description: str | None,
    base_config: dict[str, Any],
    variations: list[dict[str, Any]],
) -> None:
    """Insert an experiment and its config variations."""
    if not _pool:
        return
    try:
        async with _pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO experiments (experiment_id, name, description, base_config)
                VALUES ($1, $2, $3, $4::jsonb)
                """,
                experiment_id, name, description, json.dumps(base_config),
            )
            for v in variations:
                await conn.execute(
                    """
                    INSERT INTO experiment_configs (experiment_id, variation_name, config_json)
                    VALUES ($1, $2, $3::jsonb)
                    """,
                    experiment_id, v["name"], json.dumps(v.get("config", {})),
                )
    except Exception as exc:
        logger.error("db_create_experiment_failed", experiment_id=experiment_id, error=str(exc))


async def get_experiment(experiment_id: str) -> dict[str, Any] | None:
    """Fetch an experiment with its config variations."""
    if not _pool:
        return None
    try:
        row = await _pool.fetchrow(
            "SELECT * FROM experiments WHERE experiment_id = $1", experiment_id,
        )
        if not row:
            return None
        configs = await _pool.fetch(
            "SELECT variation_name, config_json FROM experiment_configs WHERE experiment_id = $1",
            experiment_id,
        )
        base_config = row["base_config"]
        if isinstance(base_config, str):
            base_config = json.loads(base_config)
        return {
            "experiment_id": row["experiment_id"],
            "name": row["name"],
            "description": row["description"],
            "created_at": row["created_at"].isoformat() if row["created_at"] else None,
            "status": row["status"],
            "base_config": base_config,
            "variations": [
                {
                    "name": c["variation_name"],
                    "config": json.loads(c["config_json"]) if isinstance(c["config_json"], str) else c["config_json"],
                }
                for c in configs
            ],
        }
    except Exception as exc:
        logger.error("db_get_experiment_failed", experiment_id=experiment_id, error=str(exc))
        return None


async def list_experiments() -> list[dict[str, Any]]:
    """Return all experiments with lightweight summaries."""
    if not _pool:
        return []
    try:
        rows = await _pool.fetch(
            "SELECT experiment_id, name, description, created_at, status FROM experiments ORDER BY created_at DESC"
        )
        return [
            {
                "experiment_id": r["experiment_id"],
                "name": r["name"],
                "description": r["description"],
                "created_at": r["created_at"].isoformat() if r["created_at"] else None,
                "status": r["status"],
            }
            for r in rows
        ]
    except Exception as exc:
        logger.error("db_list_experiments_failed", error=str(exc))
        return []


async def update_experiment_status(experiment_id: str, status: str) -> None:
    """Update an experiment's status."""
    if not _pool:
        return
    try:
        await _pool.execute(
            "UPDATE experiments SET status = $1 WHERE experiment_id = $2",
            status, experiment_id,
        )
    except Exception as exc:
        logger.error("db_update_experiment_status_failed", error=str(exc))


async def get_experiment_jobs(experiment_id: str) -> list[dict[str, Any]]:
    """Return all jobs belonging to an experiment."""
    if not _pool:
        return []
    try:
        rows = await _pool.fetch(
            """
            SELECT job_id, status, created_at, variation_name, config_snapshot,
                   results, error
            FROM jobs
            WHERE experiment_id = $1
            ORDER BY variation_name, created_at
            """,
            experiment_id,
        )
        out = []
        for r in rows:
            results = r["results"]
            if isinstance(results, str):
                results = json.loads(results)
            config_snapshot = r.get("config_snapshot")
            if isinstance(config_snapshot, str):
                config_snapshot = json.loads(config_snapshot)
            out.append({
                "job_id": r["job_id"],
                "status": r["status"],
                "created_at": r["created_at"],
                "variation_name": r["variation_name"],
                "config_snapshot": config_snapshot,
                "results": results,
                "error": r["error"],
            })
        return out
    except Exception as exc:
        logger.error("db_get_experiment_jobs_failed", error=str(exc))
        return []


def _row_to_dict(row: asyncpg.Record) -> dict[str, Any]:
    """Convert a database row into the dict shape expected by JobResponse."""
    results = row["results"]
    if isinstance(results, str):
        results = json.loads(results)
    recommendations = row["recommendations"]
    if isinstance(recommendations, str):
        recommendations = json.loads(recommendations)
    brand_guidelines = row["brand_guidelines"]
    if isinstance(brand_guidelines, str):
        brand_guidelines = json.loads(brand_guidelines)

    config_snapshot = row.get("config_snapshot")
    if isinstance(config_snapshot, str):
        config_snapshot = json.loads(config_snapshot)

    return {
        "job_id": row["job_id"],
        "status": row["status"],
        "created_at": row["created_at"],
        "original_image_b64": row["original_image_b64"],
        "recommendations": recommendations,
        "brand_guidelines": brand_guidelines,
        "results": results,
        "error": row["error"],
        "config_snapshot": config_snapshot,
        "experiment_id": row.get("experiment_id"),
        "variation_name": row.get("variation_name"),
    }
