"""Tests for the adaptive per-provider rate limiter."""

from __future__ import annotations

import asyncio

import pytest

from app.services.llm import (
    _AdaptiveRateLimiter,
    _is_rate_limit_error,
    _get_limiter,
)


# ═════════════════════════════════════════════════════════════════════
# Rate limit error detection
# ═════════════════════════════════════════════════════════════════════


def test_detects_429_error():
    exc = Exception("Error code: 429 - Too Many Requests")
    assert _is_rate_limit_error(exc) is True


def test_detects_rate_limit_text():
    exc = Exception("Rate limit exceeded, please retry after 30s")
    assert _is_rate_limit_error(exc) is True


def test_does_not_detect_quota_error():
    """Quota errors are permanent, not transient rate limits."""
    exc = Exception("Quota exhausted for this billing period")
    assert _is_rate_limit_error(exc) is False


def test_detects_resource_exhausted():
    exc = Exception("RESOURCE_EXHAUSTED: too many requests")
    assert _is_rate_limit_error(exc) is True


def test_does_not_detect_normal_error():
    exc = Exception("Invalid API key")
    assert _is_rate_limit_error(exc) is False


def test_does_not_detect_image_other():
    """IMAGE_OTHER was removed from rate limit detection -- it's often content policy, not throttling."""
    exc = Exception("FinishReason.IMAGE_OTHER")
    assert _is_rate_limit_error(exc) is False


# ═════════════════════════════════════════════════════════════════════
# Adaptive rate limiter
# ═════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_limiter_starts_at_default():
    limiter = _AdaptiveRateLimiter("test_default", default_rpm=20.0)
    assert limiter.rpm == 20.0


@pytest.mark.asyncio
async def test_limiter_backoff_halves_rate():
    limiter = _AdaptiveRateLimiter("test_backoff", default_rpm=20.0)
    await limiter.backoff()
    assert limiter.rpm == 10.0
    await limiter.backoff()
    assert limiter.rpm == 5.0


@pytest.mark.asyncio
async def test_limiter_backoff_has_floor():
    limiter = _AdaptiveRateLimiter("test_floor", default_rpm=2.0)
    await limiter.backoff()  # 1.0
    await limiter.backoff()  # stays at 1.0 (minimum)
    assert limiter.rpm == 1.0


@pytest.mark.asyncio
async def test_limiter_auto_configures_from_headers():
    limiter = _AdaptiveRateLimiter("test_headers", default_rpm=10.0)
    await limiter.update_from_headers({"x-ratelimit-limit-requests": "100"})
    # Should use 80% of 100 = 80
    assert limiter.rpm == 80.0


@pytest.mark.asyncio
async def test_limiter_ignores_missing_headers():
    limiter = _AdaptiveRateLimiter("test_no_headers", default_rpm=10.0)
    await limiter.update_from_headers({})
    assert limiter.rpm == 10.0  # unchanged


@pytest.mark.asyncio
async def test_limiter_ignores_invalid_headers():
    limiter = _AdaptiveRateLimiter("test_bad_headers", default_rpm=10.0)
    await limiter.update_from_headers({"x-ratelimit-limit-requests": "not_a_number"})
    assert limiter.rpm == 10.0  # unchanged


@pytest.mark.asyncio
async def test_limiter_acquire_does_not_block_under_limit():
    """Acquire should return immediately when under the rate limit."""
    limiter = _AdaptiveRateLimiter("test_acquire", default_rpm=100.0)
    # Should complete without timeout
    await asyncio.wait_for(limiter.acquire(), timeout=1.0)


# ═════════════════════════════════════════════════════════════════════
# Per-provider isolation
# ═════════════════════════════════════════════════════════════════════


def test_providers_have_separate_limiters():
    openai = _get_limiter("openai")
    gemini = _get_limiter("gemini")
    assert openai is not gemini
    assert openai.name == "openai"
    assert gemini.name == "gemini"


def test_get_limiter_returns_same_instance():
    a = _get_limiter("openai")
    b = _get_limiter("openai")
    assert a is b


@pytest.mark.asyncio
async def test_backoff_is_per_provider():
    """Backing off one provider should not affect another."""
    limiter_a = _AdaptiveRateLimiter("provider_a", default_rpm=20.0)
    limiter_b = _AdaptiveRateLimiter("provider_b", default_rpm=20.0)
    await limiter_a.backoff()
    assert limiter_a.rpm == 10.0
    assert limiter_b.rpm == 20.0  # unaffected
