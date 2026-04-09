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

_ROLLOUT_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS rollout_configs (
    name            TEXT PRIMARY KEY,
    config_json     JSONB NOT NULL DEFAULT '{}',
    weight          FLOAT NOT NULL DEFAULT 1.0,
    active          BOOLEAN NOT NULL DEFAULT true,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);
"""

_MIGRATION_SQL = """
ALTER TABLE jobs ADD COLUMN IF NOT EXISTS last_heartbeat TIMESTAMPTZ;
ALTER TABLE jobs ADD COLUMN IF NOT EXISTS config_snapshot JSONB;
ALTER TABLE jobs ADD COLUMN IF NOT EXISTS experiment_id TEXT;
ALTER TABLE jobs ADD COLUMN IF NOT EXISTS variation_name TEXT;
ALTER TABLE jobs ADD COLUMN IF NOT EXISTS rollout_config_name TEXT;
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
            await conn.execute(_ROLLOUT_SCHEMA_SQL)
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
    rollout_config_name: str | None = None,
) -> None:
    """Insert a new job row."""
    if not _pool:
        return
    try:
        await _pool.execute(
            """
            INSERT INTO jobs (job_id, status, created_at, original_image_b64,
                              recommendations, brand_guidelines, results,
                              config_snapshot, experiment_id, variation_name,
                              rollout_config_name)
            VALUES ($1, $2, $3, $4, $5::jsonb, $6::jsonb, '[]'::jsonb,
                    $7::jsonb, $8, $9, $10)
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
            rollout_config_name,
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


# ── Critic calibration ─────────────────────────────────────────────


async def get_feedback_calibration() -> dict[str, Any]:
    """Analyse correlation between critic scores and user feedback.

    Returns score distributions by sentiment, per-provider satisfaction,
    threshold analysis, and per-recommendation-type breakdown.
    """
    if not _pool:
        return {}
    try:
        async with _pool.acquire() as conn:
            # Score distribution by sentiment
            score_rows = await conn.fetch("""
                SELECT
                    CASE WHEN feedback_type IN ('thumbs_up', 'selected') THEN 'positive'
                         WHEN feedback_type = 'thumbs_down' THEN 'negative'
                         ELSE 'other' END AS sentiment,
                    COUNT(*) AS count,
                    AVG(evaluation_score) AS avg_score,
                    MIN(evaluation_score) AS min_score,
                    MAX(evaluation_score) AS max_score,
                    percentile_cont(0.5) WITHIN GROUP (ORDER BY evaluation_score) AS median_score
                FROM feedback
                WHERE evaluation_score IS NOT NULL
                  AND feedback_type IN ('thumbs_up', 'thumbs_down', 'selected')
                GROUP BY sentiment
            """)
            score_dist = {}
            for r in score_rows:
                score_dist[r["sentiment"]] = {
                    "count": r["count"],
                    "avg_score": round(float(r["avg_score"]), 3) if r["avg_score"] else 0,
                    "median_score": round(float(r["median_score"]), 3) if r["median_score"] else 0,
                    "min_score": round(float(r["min_score"]), 3) if r["min_score"] else 0,
                    "max_score": round(float(r["max_score"]), 3) if r["max_score"] else 0,
                }

            # Per-provider satisfaction
            provider_rows = await conn.fetch("""
                SELECT
                    text_provider, image_provider,
                    COUNT(*) AS total,
                    COUNT(*) FILTER (WHERE feedback_type IN ('thumbs_up', 'selected')) AS positive,
                    COUNT(*) FILTER (WHERE feedback_type = 'thumbs_down') AS negative
                FROM feedback
                WHERE feedback_type IN ('thumbs_up', 'thumbs_down', 'selected')
                GROUP BY text_provider, image_provider
            """)
            provider_satisfaction = [
                {
                    "text_provider": r["text_provider"],
                    "image_provider": r["image_provider"],
                    "total": r["total"],
                    "positive": r["positive"],
                    "negative": r["negative"],
                    "positive_rate": round(r["positive"] / r["total"], 3) if r["total"] else 0,
                }
                for r in provider_rows
            ]

            # Threshold analysis
            thresholds = [0.3, 0.5, 0.6, 0.7, 0.8, 0.9]
            threshold_analysis = []
            for t in thresholds:
                row = await conn.fetchrow("""
                    SELECT
                        COUNT(*) AS total_above,
                        COUNT(*) FILTER (WHERE feedback_type IN ('thumbs_up', 'selected')) AS positive_above
                    FROM feedback
                    WHERE evaluation_score IS NOT NULL
                      AND evaluation_score >= $1
                      AND feedback_type IN ('thumbs_up', 'thumbs_down', 'selected')
                """, t)
                threshold_analysis.append({
                    "threshold": t,
                    "total_above": row["total_above"],
                    "positive_above": row["positive_above"],
                    "positive_rate": round(row["positive_above"] / row["total_above"], 3) if row["total_above"] else 0,
                })

            # Per-recommendation-type breakdown
            rec_rows = await conn.fetch("""
                SELECT
                    recommendation_type,
                    COUNT(*) AS total,
                    COUNT(*) FILTER (WHERE feedback_type IN ('thumbs_up', 'selected')) AS positive,
                    COUNT(*) FILTER (WHERE feedback_type = 'thumbs_down') AS negative,
                    AVG(evaluation_score) AS avg_score
                FROM feedback
                WHERE feedback_type IN ('thumbs_up', 'thumbs_down', 'selected')
                  AND recommendation_type IS NOT NULL
                GROUP BY recommendation_type
            """)
            by_rec_type = [
                {
                    "type": r["recommendation_type"],
                    "total": r["total"],
                    "positive": r["positive"],
                    "negative": r["negative"],
                    "positive_rate": round(r["positive"] / r["total"], 3) if r["total"] else 0,
                    "avg_score": round(float(r["avg_score"]), 3) if r["avg_score"] else 0,
                }
                for r in rec_rows
            ]

            # Total count
            total_row = await conn.fetchrow("""
                SELECT COUNT(*) AS cnt FROM feedback
                WHERE evaluation_score IS NOT NULL
                  AND feedback_type IN ('thumbs_up', 'thumbs_down', 'selected')
            """)

            return {
                "score_distribution": score_dist,
                "provider_satisfaction": provider_satisfaction,
                "threshold_analysis": threshold_analysis,
                "by_recommendation_type": by_rec_type,
                "total_feedback_with_scores": total_row["cnt"],
            }
    except Exception as exc:
        logger.error("db_get_calibration_failed", error=str(exc))
        return {}


# ── Rollout config operations ─────────────────────────────────────


async def create_rollout_config(
    name: str, config_json: dict[str, Any], weight: float, active: bool = True,
) -> None:
    """Insert or update a rollout config (upsert)."""
    if not _pool:
        return
    try:
        await _pool.execute(
            """
            INSERT INTO rollout_configs (name, config_json, weight, active, updated_at)
            VALUES ($1, $2::jsonb, $3, $4, NOW())
            ON CONFLICT (name) DO UPDATE SET
                config_json = EXCLUDED.config_json,
                weight = EXCLUDED.weight,
                active = EXCLUDED.active,
                updated_at = NOW()
            """,
            name, json.dumps(config_json), weight, active,
        )
    except Exception as exc:
        logger.error("db_create_rollout_config_failed", name=name, error=str(exc))


async def list_rollout_configs() -> list[dict[str, Any]]:
    """Return all rollout configs."""
    if not _pool:
        return []
    try:
        rows = await _pool.fetch(
            "SELECT name, config_json, weight, active, created_at, updated_at "
            "FROM rollout_configs ORDER BY name"
        )
        return [
            {
                "name": r["name"],
                "config": json.loads(r["config_json"]) if isinstance(r["config_json"], str) else r["config_json"],
                "weight": float(r["weight"]),
                "active": r["active"],
                "created_at": r["created_at"].isoformat() if r["created_at"] else None,
                "updated_at": r["updated_at"].isoformat() if r["updated_at"] else None,
            }
            for r in rows
        ]
    except Exception as exc:
        logger.error("db_list_rollout_configs_failed", error=str(exc))
        return []


async def get_active_rollout_configs() -> list[dict[str, Any]]:
    """Return only active rollout configs (for traffic routing)."""
    if not _pool:
        return []
    try:
        rows = await _pool.fetch(
            "SELECT name, config_json, weight FROM rollout_configs "
            "WHERE active = true ORDER BY name"
        )
        return [
            {
                "name": r["name"],
                "config": json.loads(r["config_json"]) if isinstance(r["config_json"], str) else r["config_json"],
                "weight": float(r["weight"]),
            }
            for r in rows
        ]
    except Exception as exc:
        logger.error("db_get_active_rollout_configs_failed", error=str(exc))
        return []


async def update_rollout_config(
    name: str, weight: float | None = None, active: bool | None = None,
) -> bool:
    """Update a rollout config's weight and/or active flag. Returns True if found."""
    if not _pool:
        return False
    try:
        sets = ["updated_at = NOW()"]
        params: list[Any] = []
        idx = 1
        if weight is not None:
            sets.append(f"weight = ${idx}")
            params.append(weight)
            idx += 1
        if active is not None:
            sets.append(f"active = ${idx}")
            params.append(active)
            idx += 1
        params.append(name)
        result = await _pool.execute(
            f"UPDATE rollout_configs SET {', '.join(sets)} WHERE name = ${idx}",
            *params,
        )
        return result.endswith("1")
    except Exception as exc:
        logger.error("db_update_rollout_config_failed", name=name, error=str(exc))
        return False


async def delete_rollout_config(name: str) -> bool:
    """Delete a rollout config. Returns True if found."""
    if not _pool:
        return False
    try:
        result = await _pool.execute(
            "DELETE FROM rollout_configs WHERE name = $1", name,
        )
        return result.endswith("1")
    except Exception as exc:
        logger.error("db_delete_rollout_config_failed", name=name, error=str(exc))
        return False


async def get_rollout_performance() -> list[dict[str, Any]]:
    """Return per-rollout-config feedback and performance stats."""
    if not _pool:
        return []
    try:
        rows = await _pool.fetch("""
            SELECT
                j.rollout_config_name,
                COUNT(DISTINCT j.job_id) AS job_count,
                COUNT(f.id) AS feedback_count,
                COUNT(f.id) FILTER (WHERE f.feedback_type IN ('thumbs_up', 'selected')) AS positive,
                COUNT(f.id) FILTER (WHERE f.feedback_type = 'thumbs_down') AS negative,
                AVG(f.evaluation_score) AS avg_score
            FROM jobs j
            LEFT JOIN feedback f ON j.job_id = f.job_id
            WHERE j.rollout_config_name IS NOT NULL
            GROUP BY j.rollout_config_name
            ORDER BY j.rollout_config_name
        """)
        return [
            {
                "config_name": r["rollout_config_name"],
                "job_count": r["job_count"],
                "feedback_count": r["feedback_count"],
                "positive": r["positive"],
                "negative": r["negative"],
                "positive_rate": round(r["positive"] / r["feedback_count"], 3) if r["feedback_count"] else 0,
                "avg_score": round(float(r["avg_score"]), 3) if r["avg_score"] else None,
            }
            for r in rows
        ]
    except Exception as exc:
        logger.error("db_get_rollout_performance_failed", error=str(exc))
        return []


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
