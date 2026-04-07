"""Tests for the FastAPI endpoints.

Unit tests mock the LLM calls and database. Integration tests require
Docker services (PostgreSQL) to be running.
"""

from __future__ import annotations

import base64
import io
import json
import time
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app, _jobs
from app.models import RecommendationResult, VariantResult

client = TestClient(app)


# ── Fixtures ──────────────────────────────────────────────────────────


def _make_test_image() -> bytes:
    """Create a minimal valid PNG image."""
    img = Image.new("RGB", (100, 100), color="red")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_request_body(num_recs: int = 1) -> str:
    """Build a valid JobRequest JSON string with *num_recs* recommendations."""
    recs = [
        {
            "id": f"rec_{i}",
            "title": f"Recommendation {i}",
            "description": f"Improve aspect {i}",
            "type": "contrast_salience",
        }
        for i in range(1, num_recs + 1)
    ]
    return json.dumps(
        {
            "recommendations": recs,
            "brand_guidelines": {
                "protected_regions": ["Do not modify logo"],
                "typography": "Keep fonts",
                "aspect_ratio": "1:1",
                "brand_elements": "Logo must stay",
            },
        }
    )


def _submit_job(body: str | None = None) -> dict:
    """Submit a job via the API and return the response JSON."""
    resp = client.post(
        "/api/v1/jobs",
        files={
            "image": ("test.png", _make_test_image(), "image/png"),
            "request_body": ("body.json", body or _make_request_body(), "application/json"),
        },
    )
    assert resp.status_code == 202
    return resp.json()


# ── Health ────────────────────────────────────────────────────────────


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"


def test_cors_headers_present():
    """CORS preflight requests should return appropriate headers."""
    resp = client.options(
        "/api/v1/jobs",
        headers={
            "Origin": "http://localhost:7860",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert "access-control-allow-origin" in resp.headers


# ── Input Validation ──────────────────────────────────────────────────


def test_create_job_invalid_image_type():
    resp = client.post(
        "/api/v1/jobs",
        files={
            "image": ("test.txt", b"not an image", "text/plain"),
            "request_body": ("body.json", _make_request_body(), "application/json"),
        },
    )
    assert resp.status_code == 400
    assert "Unsupported image type" in resp.json()["detail"]


def test_create_job_invalid_body():
    resp = client.post(
        "/api/v1/jobs",
        files={
            "image": ("test.png", _make_test_image(), "image/png"),
            "request_body": ("body.json", "not json", "application/json"),
        },
    )
    assert resp.status_code == 422


def test_create_job_empty_recommendations():
    body = json.dumps({"recommendations": [], "brand_guidelines": {}})
    resp = client.post(
        "/api/v1/jobs",
        files={
            "image": ("test.png", _make_test_image(), "image/png"),
            "request_body": ("body.json", body, "application/json"),
        },
    )
    assert resp.status_code == 400


def test_create_job_corrupt_image():
    """A file with image MIME type but invalid contents should be rejected."""
    resp = client.post(
        "/api/v1/jobs",
        files={
            "image": ("test.png", b"not a real png", "image/png"),
            "request_body": ("body.json", _make_request_body(), "application/json"),
        },
    )
    assert resp.status_code == 400
    assert "not a valid image" in resp.json()["detail"]


def test_create_job_image_too_large():
    """An image exceeding the file size limit should be rejected."""
    # Create an image larger than 10MB
    large_img = Image.new("RGB", (100, 100), color="red")
    buf = io.BytesIO()
    large_img.save(buf, format="PNG")
    # Pad to exceed 50MB limit
    buf.write(b"\x00" * (51 * 1024 * 1024))
    resp = client.post(
        "/api/v1/jobs",
        files={
            "image": ("big.png", buf.getvalue(), "image/png"),
            "request_body": ("body.json", _make_request_body(), "application/json"),
        },
    )
    assert resp.status_code == 400
    assert "file size" in resp.json()["detail"].lower()


def test_create_job_image_too_large_dimensions():
    """An image exceeding the dimension limit should be rejected."""
    huge_img = Image.new("RGB", (9000, 9000), color="blue")
    buf = io.BytesIO()
    huge_img.save(buf, format="PNG")
    resp = client.post(
        "/api/v1/jobs",
        files={
            "image": ("huge.png", buf.getvalue(), "image/png"),
            "request_body": ("body.json", _make_request_body(), "application/json"),
        },
    )
    assert resp.status_code == 400
    assert "dimensions" in resp.json()["detail"].lower()


def test_create_job_too_many_recommendations():
    """Exceeding the max recommendations limit should be rejected."""
    resp = client.post(
        "/api/v1/jobs",
        files={
            "image": ("test.png", _make_test_image(), "image/png"),
            "request_body": ("body.json", _make_request_body(num_recs=8), "application/json"),
        },
    )
    assert resp.status_code == 400
    assert "too many" in resp.json()["detail"].lower()


def test_create_job_recommendation_text_too_long():
    """A recommendation with an excessively long description should be rejected."""
    body = json.dumps({
        "recommendations": [{
            "id": "rec_1",
            "title": "Test",
            "description": "x" * 3000,
            "type": "contrast",
        }],
        "brand_guidelines": {},
    })
    resp = client.post(
        "/api/v1/jobs",
        files={
            "image": ("test.png", _make_test_image(), "image/png"),
            "request_body": ("body.json", body, "application/json"),
        },
    )
    assert resp.status_code == 400
    assert "character limit" in resp.json()["detail"].lower()


# ── Job Not Found ─────────────────────────────────────────────────────


def test_get_job_not_found():
    resp = client.get("/api/v1/jobs/nonexistent")
    assert resp.status_code == 404


def test_get_image_job_not_found():
    resp = client.get("/api/v1/jobs/nonexistent/image/v1")
    assert resp.status_code == 404


# ── Job Lifecycle (mocked workflow) ───────────────────────────────────


@patch("app.main.run_all_recommendations", new_callable=AsyncMock)
def test_create_and_poll_job(mock_run):
    """Submit a job, let the mock workflow complete, and poll for results."""
    mock_variant = VariantResult(
        variant_id="rec_1_v1",
        variant_title="Bold Headline Contrast",
        variant_description="Increase headline contrast with dark backdrop",
        status="accepted",
        attempts=1,
        edited_image_b64="abc123base64",
        evaluation_score=0.92,
        evaluation_feedback="Good application of contrast",
    )
    mock_result = RecommendationResult(
        recommendation_id="rec_1",
        recommendation_title="Recommendation 1",
        variants=[mock_variant],
    )

    async def fake_run(**kwargs):
        job_results = kwargs.get("job_results", [])
        job_results.append(mock_result)
        if kwargs.get("on_variant_complete"):
            await kwargs["on_variant_complete"]()
        return job_results

    mock_run.side_effect = fake_run

    # Submit
    data = _submit_job()
    job_id = data["job_id"]
    assert data["status"] in ("pending", "running", "completed")

    # Poll until complete (max 5 seconds)
    for _ in range(10):
        resp = client.get(f"/api/v1/jobs/{job_id}")
        assert resp.status_code == 200
        poll_data = resp.json()
        if poll_data["status"] == "completed":
            break
        time.sleep(0.5)

    assert poll_data["status"] == "completed"
    assert len(poll_data["results"]) == 1
    assert len(poll_data["results"][0]["variants"]) == 1

    variant = poll_data["results"][0]["variants"][0]
    assert variant["variant_id"] == "rec_1_v1"
    assert variant["variant_title"] == "Bold Headline Contrast"
    assert variant["status"] == "accepted"
    assert variant["evaluation_score"] == 0.92
    assert variant["edited_image_b64"] == "abc123base64"


@patch("app.main.run_all_recommendations", new_callable=AsyncMock)
def test_job_failure_sets_status(mock_run):
    """A workflow exception should set the job to 'failed' with an error message."""
    mock_run.side_effect = RuntimeError("LLM API unreachable")

    data = _submit_job()
    job_id = data["job_id"]

    for _ in range(10):
        resp = client.get(f"/api/v1/jobs/{job_id}")
        poll_data = resp.json()
        if poll_data["status"] == "failed":
            break
        time.sleep(0.5)

    assert poll_data["status"] == "failed"
    assert "LLM API unreachable" in poll_data["error"]


@patch("app.main.run_all_recommendations", new_callable=AsyncMock)
def test_list_jobs_returns_all(mock_run):
    """The list endpoint should return summaries for all submitted jobs."""
    mock_run.return_value = []

    # Submit two jobs
    _submit_job()
    _submit_job()

    resp = client.get("/api/v1/jobs")
    assert resp.status_code == 200
    data = resp.json()
    # At least the two we just created (others may exist from other tests)
    assert len(data["jobs"]) >= 2

    for job_summary in data["jobs"]:
        assert "job_id" in job_summary
        assert "status" in job_summary
        assert "num_recommendations" in job_summary


@patch("app.main.run_all_recommendations", new_callable=AsyncMock)
def test_get_variant_image(mock_run):
    """The image endpoint should return the base64 image for a specific variant."""
    mock_variant = VariantResult(
        variant_id="rec_1_v1",
        variant_title="Test Variant",
        variant_description="Test",
        status="accepted",
        attempts=1,
        edited_image_b64="dGVzdGltYWdl",  # base64("testimage")
        evaluation_score=1.0,
    )
    mock_result = RecommendationResult(
        recommendation_id="rec_1",
        recommendation_title="Test",
        variants=[mock_variant],
    )

    async def fake_run(**kwargs):
        job_results = kwargs.get("job_results", [])
        job_results.append(mock_result)
        return job_results

    mock_run.side_effect = fake_run

    data = _submit_job()
    job_id = data["job_id"]

    # Wait for completion
    for _ in range(10):
        resp = client.get(f"/api/v1/jobs/{job_id}")
        if resp.json()["status"] == "completed":
            break
        time.sleep(0.5)

    resp = client.get(f"/api/v1/jobs/{job_id}/image/rec_1_v1")
    assert resp.status_code == 200
    assert resp.json()["image_b64"] == "dGVzdGltYWdl"


@patch("app.main.run_all_recommendations", new_callable=AsyncMock)
def test_get_variant_image_not_found(mock_run):
    """Requesting a non-existent variant ID should return 404."""
    mock_result = RecommendationResult(
        recommendation_id="rec_1",
        recommendation_title="Test",
        variants=[],
    )

    async def fake_run(**kwargs):
        job_results = kwargs.get("job_results", [])
        job_results.append(mock_result)
        return job_results

    mock_run.side_effect = fake_run

    data = _submit_job()
    job_id = data["job_id"]

    for _ in range(10):
        resp = client.get(f"/api/v1/jobs/{job_id}")
        if resp.json()["status"] == "completed":
            break
        time.sleep(0.5)

    resp = client.get(f"/api/v1/jobs/{job_id}/image/nonexistent_variant")
    assert resp.status_code == 404


@patch("app.main.run_all_recommendations", new_callable=AsyncMock)
def test_multiple_recommendations(mock_run):
    """A job with multiple recommendations should return results for each."""
    mock_results = [
        RecommendationResult(
            recommendation_id=f"rec_{i}",
            recommendation_title=f"Recommendation {i}",
            variants=[
                VariantResult(
                    variant_id=f"rec_{i}_v1",
                    variant_title=f"Variant for rec {i}",
                    variant_description="Test",
                    status="accepted",
                    attempts=1,
                    evaluation_score=0.9,
                )
            ],
        )
        for i in range(1, 4)
    ]

    async def fake_run(**kwargs):
        job_results = kwargs.get("job_results", [])
        job_results.extend(mock_results)
        return job_results

    mock_run.side_effect = fake_run

    data = _submit_job(_make_request_body(num_recs=3))
    job_id = data["job_id"]

    for _ in range(10):
        resp = client.get(f"/api/v1/jobs/{job_id}")
        if resp.json()["status"] == "completed":
            break
        time.sleep(0.5)

    poll_data = resp.json()
    assert poll_data["status"] == "completed"
    assert len(poll_data["results"]) == 3
    for i, result in enumerate(poll_data["results"], 1):
        assert result["recommendation_id"] == f"rec_{i}"
        assert len(result["variants"]) == 1


# ── Database Persistence (mocked DB) ──────────────────────────────────


@patch("app.main.db")
@patch("app.main.run_all_recommendations", new_callable=AsyncMock)
def test_job_persisted_to_db_on_create(mock_run, mock_db):
    """Creating a job should call db.create_job when DB is enabled."""
    mock_run.return_value = []
    mock_db.is_enabled.return_value = True
    mock_db.create_job = AsyncMock()
    mock_db.update_job_status = AsyncMock()
    mock_db.update_job_results = AsyncMock()
    mock_db.touch_heartbeat = AsyncMock()

    data = _submit_job()

    mock_db.create_job.assert_called_once()
    call_kwargs = mock_db.create_job.call_args
    assert call_kwargs.kwargs["job_id"] == data["job_id"]
    assert call_kwargs.kwargs["status"] == "pending"


@patch("app.main.db")
@patch("app.main.run_all_recommendations", new_callable=AsyncMock)
def test_job_status_updated_in_db(mock_run, mock_db):
    """The background worker should update job status in the database."""
    mock_db.is_enabled.return_value = True
    mock_db.create_job = AsyncMock()
    mock_db.update_job_status = AsyncMock()
    mock_db.update_job_results = AsyncMock()
    mock_db.touch_heartbeat = AsyncMock()

    async def fake_run(**kwargs):
        job_results = kwargs.get("job_results", [])
        return job_results

    mock_run.side_effect = fake_run

    _submit_job()
    time.sleep(1)

    # Should have been called with "running" and then "completed"
    status_calls = [
        c.args[1] for c in mock_db.update_job_status.call_args_list
    ]
    assert "running" in status_calls
    assert "completed" in status_calls


@patch("app.main.db")
def test_fallback_to_memory_without_db(mock_db):
    """When DB is disabled, the API should still work with in-memory store."""
    mock_db.is_enabled.return_value = False

    resp = client.get("/api/v1/jobs")
    assert resp.status_code == 200


@patch("app.main.db")
@patch("app.main.run_all_recommendations", new_callable=AsyncMock)
def test_job_retrieved_from_db_when_not_in_memory(mock_run, mock_db):
    """If a job is not in the in-memory store, it should be fetched from DB."""
    mock_db.is_enabled.return_value = True
    mock_db.get_job = AsyncMock(return_value={
        "job_id": "db-only-job",
        "status": "completed",
        "created_at": "2026-04-05T00:00:00Z",
        "original_image_b64": "",
        "recommendations": [],
        "brand_guidelines": {},
        "results": [
            {
                "recommendation_id": "rec_1",
                "recommendation_title": "Test",
                "variants": [
                    {
                        "variant_id": "rec_1_v1",
                        "variant_title": "DB Variant",
                        "variant_description": "From database",
                        "status": "accepted",
                        "attempts": 1,
                        "edited_image_b64": None,
                        "evaluation_score": 0.8,
                        "evaluation_feedback": "",
                        "audit_trail": [],
                    }
                ],
            }
        ],
        "error": None,
    })

    resp = client.get("/api/v1/jobs/db-only-job")
    assert resp.status_code == 200
    data = resp.json()
    assert data["job_id"] == "db-only-job"
    assert data["status"] == "completed"
    assert data["results"][0]["variants"][0]["variant_title"] == "DB Variant"


# ── Metrics & Observability ───────────────────────────────────────────


# ── Cancel endpoint ───────────────────────────────────────────────────


@patch("app.main.db")
@patch("app.main.run_all_recommendations", new_callable=AsyncMock)
def test_cancel_job(mock_run, mock_db):
    """Cancelling a running job should set its status to failed."""
    mock_db.is_enabled.return_value = False

    async def slow_run(**kwargs):
        import asyncio
        await asyncio.sleep(10)  # simulate long job

    mock_run.side_effect = slow_run

    data = _submit_job()
    job_id = data["job_id"]

    # Cancel immediately
    resp = client.post(f"/api/v1/jobs/{job_id}/cancel")
    assert resp.status_code == 200
    assert resp.json()["status"] == "cancelled"

    # Job should now be failed
    resp = client.get(f"/api/v1/jobs/{job_id}")
    assert resp.json()["status"] == "failed"
    assert "Cancelled" in resp.json()["error"]


def test_cancel_nonexistent_job():
    """Cancelling a job that doesn't exist should still return 200."""
    resp = client.post("/api/v1/jobs/nonexistent/cancel")
    assert resp.status_code == 200


# ── Feedback endpoint ─────────────────────────────────────────────────


@patch("app.main.db")
def test_submit_feedback(mock_db):
    """Submitting feedback should return recorded status."""
    mock_db.is_enabled.return_value = False
    resp = client.post(
        "/api/v1/jobs/test-job/feedback",
        json={"variant_id": "v1", "feedback_type": "thumbs_up"},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "recorded"


@patch("app.main.db")
def test_submit_feedback_with_comment(mock_db):
    """Feedback with a refinement comment should be accepted."""
    mock_db.is_enabled.return_value = False
    resp = client.post(
        "/api/v1/jobs/test-job/feedback",
        json={"variant_id": "v1", "feedback_type": "refinement", "comment": "Make it brighter"},
    )
    assert resp.status_code == 200


def test_submit_feedback_invalid_type():
    """Invalid feedback type should return 400."""
    resp = client.post(
        "/api/v1/jobs/test-job/feedback",
        json={"variant_id": "v1", "feedback_type": "invalid"},
    )
    assert resp.status_code == 400


def test_submit_feedback_missing_variant():
    """Missing variant_id should return 400."""
    resp = client.post(
        "/api/v1/jobs/test-job/feedback",
        json={"variant_id": "", "feedback_type": "thumbs_up"},
    )
    assert resp.status_code == 400


# ── Thumbnail endpoint ────────────────────────────────────────────────


@patch("app.main.run_all_recommendations", new_callable=AsyncMock)
def test_thumbnail_endpoint(mock_run):
    """The thumbnail endpoint should return a JPEG image."""
    # Create a small test image as base64
    test_img = Image.new("RGB", (100, 100), color="green")
    buf = io.BytesIO()
    test_img.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    mock_variant = VariantResult(
        variant_id="rec_1_v1",
        variant_title="Test",
        variant_description="Test",
        status="accepted",
        attempts=1,
        edited_image_b64=img_b64,
        evaluation_score=1.0,
    )
    mock_result = RecommendationResult(
        recommendation_id="rec_1",
        recommendation_title="Test",
        variants=[mock_variant],
    )

    async def fake_run(**kwargs):
        job_results = kwargs.get("job_results", [])
        job_results.append(mock_result)
        return job_results

    mock_run.side_effect = fake_run

    data = _submit_job()
    job_id = data["job_id"]
    time.sleep(1)

    resp = client.get(f"/api/v1/thumb/{job_id}/rec_1_v1")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/jpeg"
    assert len(resp.content) > 0


def test_thumbnail_not_found():
    """Requesting a thumbnail for a nonexistent job should return 404."""
    resp = client.get("/api/v1/thumb/nonexistent/v1")
    assert resp.status_code == 404


# ── Input validation: RGBA warning ────────────────────────────────────


@patch("app.main.run_all_recommendations", new_callable=AsyncMock)
def test_rgba_image_returns_warning(mock_run):
    """Uploading an RGBA image should return a warning in the response."""
    mock_run.return_value = []

    # Create RGBA image
    rgba_img = Image.new("RGBA", (100, 100), color=(255, 0, 0, 128))
    buf = io.BytesIO()
    rgba_img.save(buf, format="PNG")

    resp = client.post(
        "/api/v1/jobs",
        files={
            "image": ("test.png", buf.getvalue(), "image/png"),
            "request_body": ("body.json", _make_request_body(), "application/json"),
        },
    )
    assert resp.status_code == 202
    data = resp.json()
    assert "warnings" in data
    assert any("alpha" in w.lower() or "rgba" in w.lower() for w in data["warnings"])


# ── Metrics & Observability ───────────────────────────────────────────


def test_prometheus_metrics_endpoint():
    """The /metrics endpoint should return Prometheus-format data."""
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "visrec_jobs_submitted_total" in resp.text


def test_metrics_summary_endpoint():
    """The /api/v1/metrics/summary endpoint should return structured JSON."""
    resp = client.get("/api/v1/metrics/summary")
    assert resp.status_code == 200
    data = resp.json()
    assert "jobs_submitted" in data
    assert "jobs_completed" in data
    assert "jobs_failed" in data
    assert "jobs_in_progress" in data
    assert "variants_accepted" in data
    assert "parse_failures" in data
    assert "agent_invocations" in data
    assert "ideator" in data["agent_invocations"]
    assert "idea_critic" in data["agent_invocations"]
    assert "editor" in data["agent_invocations"]
    assert "critic" in data["agent_invocations"]
    assert "refiner" in data["agent_invocations"]
