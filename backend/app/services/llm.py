"""Thin wrappers around LLM APIs for vision chat and image editing.

Two providers are supported; each capability can use a different one:
  - ``settings.text_provider``  -- "openai" or "gemini" (vision chat)
  - ``settings.image_provider`` -- "openai" or "gemini" (image editing)

Provider SDK clients are lazy-initialised on first use so the app
starts quickly and only pays import cost when the provider is needed.
"""

from __future__ import annotations

import asyncio
import base64
import time
from typing import Any

import structlog
from aiolimiter import AsyncLimiter

from app.config import settings
from app import metrics

logger = structlog.get_logger()

# ── Per-provider adaptive rate limiters ───────────────────────────────
#
# Each provider gets its own rate limiter that auto-configures from API
# response headers. OpenAI sends x-ratelimit-limit-requests and
# x-ratelimit-remaining-requests on every response. Gemini has similar
# signals. The limiter starts at a default and adjusts to the actual
# limit once the first response arrives.


class _AdaptiveRateLimiter:
    """Per-provider rate limiter that learns from API response headers."""

    def __init__(self, name: str, default_rpm: float = 30.0):
        self.name = name
        self.rpm = default_rpm
        self._default_rpm = default_rpm
        self.limiter = AsyncLimiter(max_rate=default_rpm, time_period=60)
        self._lock = asyncio.Lock()
        self._configured = False
        self._successes = 0

    async def acquire(self) -> None:
        await self.limiter.acquire()

    async def update_from_headers(self, headers: dict) -> None:
        """Read rate limit info from API response headers and adjust."""
        # OpenAI headers: x-ratelimit-limit-requests, x-ratelimit-remaining-requests
        limit_str = headers.get("x-ratelimit-limit-requests")
        if limit_str:
            try:
                actual_rpm = float(limit_str)
                # Use 80% of the actual limit for safety margin
                target = actual_rpm * 0.8
                if target > 0 and abs(target - self.rpm) > 1:
                    async with self._lock:
                        self.rpm = target
                        self.limiter = AsyncLimiter(max_rate=self.rpm, time_period=60)
                        if not self._configured:
                            logger.info("rate_limit_auto_configured",
                                        provider=self.name, actual_rpm=actual_rpm, using_rpm=round(self.rpm, 1))
                            self._configured = True
                        metrics.current_rate_limit.set(self.rpm)
            except (ValueError, TypeError):
                pass

    async def on_success(self) -> None:
        """After enough consecutive successes, gradually recover the rate."""
        async with self._lock:
            self._successes += 1
            if self._successes >= 5 and self.rpm < self._default_rpm:
                new_rpm = min(self.rpm * 1.5, self._default_rpm)
                self.rpm = new_rpm
                self.limiter = AsyncLimiter(max_rate=self.rpm, time_period=60)
                self._successes = 0
                metrics.current_rate_limit.set(self.rpm)
                logger.info("rate_limit_recovery", provider=self.name, new_rpm=round(self.rpm, 1))

    async def backoff(self) -> None:
        """Halve the rate after an error."""
        async with self._lock:
            new_rpm = max(self.rpm / 2, 1.0)
            if new_rpm != self.rpm:
                self.rpm = new_rpm
                self.limiter = AsyncLimiter(max_rate=self.rpm, time_period=60)
                self._successes = 0
                metrics.current_rate_limit.set(self.rpm)
                logger.warning("rate_limit_backoff", provider=self.name, new_rpm=round(self.rpm, 1))


_limiters: dict[str, _AdaptiveRateLimiter] = {
    "openai": _AdaptiveRateLimiter("openai", default_rpm=settings.llm_rate_limit),
    "gemini": _AdaptiveRateLimiter("gemini", default_rpm=settings.llm_rate_limit),
    "claude": _AdaptiveRateLimiter("claude", default_rpm=settings.llm_rate_limit),
}


def _get_limiter(provider: str) -> _AdaptiveRateLimiter:
    if provider not in _limiters:
        _limiters[provider] = _AdaptiveRateLimiter(provider, default_rpm=settings.llm_rate_limit)
    return _limiters[provider]


def _is_rate_limit_error(exc: Exception) -> bool:
    """Check if an exception indicates a transient rate limit (not quota exhaustion)."""
    msg = str(exc).lower()
    # insufficient_quota is permanent -- don't treat it as a rate limit
    if "insufficient_quota" in msg:
        return False
    return any(s in msg for s in [
        "rate limit", "429", "too many requests",
        "resource_exhausted",
    ])

# ── Provider clients (lazy-initialised) ───────────────────────────────

_openai_client: Any | None = None
_gemini_client: Any | None = None
_anthropic_client: Any | None = None


def _get_openai() -> Any:
    """Return (and cache) the async OpenAI client."""
    global _openai_client
    if _openai_client is None:
        from openai import AsyncOpenAI

        _openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
    return _openai_client


def _get_gemini() -> Any:
    """Return (and cache) the Google GenAI client."""
    global _gemini_client
    if _gemini_client is None:
        from google import genai

        _gemini_client = genai.Client(api_key=settings.google_api_key)
    return _gemini_client


def _get_anthropic() -> Any:
    """Return (and cache) the async Anthropic client."""
    global _anthropic_client
    if _anthropic_client is None:
        from anthropic import AsyncAnthropic

        _anthropic_client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    return _anthropic_client


# ── Dependency health checks ──────────────────────────────────────────


async def check_openai_health() -> dict[str, Any]:
    """Test OpenAI API connectivity with a minimal call."""
    if not settings.openai_api_key:
        return {"status": "not_configured", "latency_ms": 0}
    start = time.monotonic()
    try:
        client = _get_openai()
        await client.models.list()
        latency = round((time.monotonic() - start) * 1000)
        return {"status": "healthy", "latency_ms": latency}
    except Exception as exc:
        latency = round((time.monotonic() - start) * 1000)
        return {"status": "error", "latency_ms": latency, "error": str(exc)[:200]}


async def check_gemini_health() -> dict[str, Any]:
    """Test Gemini API connectivity with a minimal call."""
    if not settings.google_api_key:
        return {"status": "not_configured", "latency_ms": 0}
    start = time.monotonic()
    try:
        client = _get_gemini()
        # Use sync client in executor to avoid async client issues
        import asyncio
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: list(client.models.list()))
        latency = round((time.monotonic() - start) * 1000)
        return {"status": "healthy", "latency_ms": latency}
    except Exception as exc:
        latency = round((time.monotonic() - start) * 1000)
        return {"status": "error", "latency_ms": latency, "error": str(exc)[:200]}


async def check_claude_health() -> dict[str, Any]:
    """Test Anthropic Claude API connectivity with a minimal call."""
    if not settings.anthropic_api_key:
        return {"status": "not_configured", "latency_ms": 0}
    start = time.monotonic()
    try:
        client = _get_anthropic()
        await client.messages.create(
            model=settings.claude_vision_model,
            max_tokens=1,
            messages=[{"role": "user", "content": "hi"}],
        )
        latency = round((time.monotonic() - start) * 1000)
        return {"status": "healthy", "latency_ms": latency}
    except Exception as exc:
        latency = round((time.monotonic() - start) * 1000)
        return {"status": "error", "latency_ms": latency, "error": str(exc)[:200]}


# ── Vision chat ───────────────────────────────────────────────────────


def _fallback_providers(primary: str) -> list[str]:
    """Return a prioritised list of alternate providers that have API keys."""
    candidates = []
    if primary != "openai" and settings.openai_api_key:
        candidates.append("openai")
    if primary != "gemini" and settings.google_api_key:
        candidates.append("gemini")
    if primary != "claude" and settings.anthropic_api_key:
        candidates.append("claude")
    return candidates


async def _call_vision_chat(
    provider: str,
    system_prompt: str,
    user_text: str,
    image_b64: str | None,
    second_image_b64: str | None,
) -> str:
    """Dispatch a vision chat call to the named provider."""
    if provider == "gemini":
        return await _gemini_vision_chat(system_prompt, user_text, image_b64, second_image_b64)
    if provider == "claude":
        return await _claude_vision_chat(system_prompt, user_text, image_b64, second_image_b64)
    return await _openai_vision_chat(system_prompt, user_text, image_b64, second_image_b64)


async def vision_chat(
    system_prompt: str,
    user_text: str,
    image_b64: str | None = None,
    second_image_b64: str | None = None,
) -> str:
    """Send a vision-capable chat completion and return the assistant text.

    Supports up to two images (e.g. original + edited for comparison).
    Falls back to the alternate provider if the primary fails and the
    alternate has an API key configured.
    """
    provider = settings.text_provider
    limiter = _get_limiter(provider)
    await limiter.acquire()
    metrics.llm_calls_total.labels(provider=provider, call_type="vision_chat").inc()
    start = time.monotonic()

    try:
        result = await _call_vision_chat(provider, system_prompt, user_text, image_b64, second_image_b64)
        duration = time.monotonic() - start
        metrics.llm_call_duration.labels(provider=provider, call_type="vision_chat").observe(duration)
        logger.debug("vision_chat_completed", provider=provider, duration_s=round(duration, 2))
        await limiter.on_success()
        return result

    except Exception as exc:
        duration = time.monotonic() - start
        metrics.llm_errors_total.labels(provider=provider, call_type="vision_chat", error_type=type(exc).__name__).inc()
        logger.error("vision_chat_failed", provider=provider, error=str(exc), duration_s=round(duration, 2))

        if _is_rate_limit_error(exc):
            await limiter.backoff()

        # Attempt fallback to alternate providers in priority order
        for fallback in _fallback_providers(provider):
            fb_limiter = _get_limiter(fallback)
            await fb_limiter.acquire()
            logger.warning("vision_chat_fallback", primary=provider, fallback=fallback)
            metrics.llm_calls_total.labels(provider=fallback, call_type="vision_chat").inc()
            start2 = time.monotonic()
            try:
                result = await _call_vision_chat(fallback, system_prompt, user_text, image_b64, second_image_b64)
                duration2 = time.monotonic() - start2
                metrics.llm_call_duration.labels(provider=fallback, call_type="vision_chat").observe(duration2)
                metrics.provider_fallbacks.labels(primary=provider, fallback=fallback, call_type="vision_chat").inc()
                logger.info("vision_chat_fallback_succeeded", fallback=fallback, duration_s=round(duration2, 2))
                await fb_limiter.on_success()
                return result
            except Exception as fallback_exc:
                metrics.llm_errors_total.labels(provider=fallback, call_type="vision_chat", error_type=type(fallback_exc).__name__).inc()
                logger.error("vision_chat_fallback_failed", fallback=fallback, error=str(fallback_exc))

        raise


async def _gemini_vision_chat(
    system_prompt: str,
    user_text: str,
    image_b64: str | None = None,
    second_image_b64: str | None = None,
) -> str:
    """Gemini implementation of multi-modal vision chat."""
    from google.genai import types

    client = _get_gemini()
    parts = [types.Part.from_text(text=user_text)]

    if image_b64:
        parts.append(types.Part.from_bytes(
            data=base64.b64decode(image_b64),
            mime_type="image/png",
        ))
    if second_image_b64:
        parts.append(types.Part.from_bytes(
            data=base64.b64decode(second_image_b64),
            mime_type="image/png",
        ))

    resp = await client.aio.models.generate_content(
        model=settings.gemini_vision_model,
        contents=parts,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            max_output_tokens=4096,
        ),
    )
    try:
        return resp.text or ""
    except (AttributeError, TypeError, ValueError):
        # Gemini sometimes returns empty/blocked responses
        if resp.candidates:
            reason = getattr(resp.candidates[0], "finish_reason", "unknown")
            logger.warning("gemini_vision_empty_response", reason=str(reason))
        return ""


async def _openai_vision_chat(
    system_prompt: str,
    user_text: str,
    image_b64: str | None = None,
    second_image_b64: str | None = None,
) -> str:
    """OpenAI implementation of multi-modal vision chat."""
    client = _get_openai()

    user_content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
    for b64 in (image_b64, second_image_b64):
        if b64:
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"},
            })

    resp = await client.chat.completions.with_raw_response.create(
        model=settings.openai_vision_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        max_tokens=4096,
    )

    # Read rate limit headers and auto-configure the limiter
    headers = dict(resp.headers)
    await _get_limiter("openai").update_from_headers(headers)

    parsed = resp.parse()
    return parsed.choices[0].message.content or ""


async def _claude_vision_chat(
    system_prompt: str,
    user_text: str,
    image_b64: str | None = None,
    second_image_b64: str | None = None,
) -> str:
    """Anthropic Claude implementation of multi-modal vision chat."""
    client = _get_anthropic()

    content: list[dict[str, Any]] = []
    for b64 in (image_b64, second_image_b64):
        if b64:
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": b64},
            })
    content.append({"type": "text", "text": user_text})

    resp = await client.messages.create(
        model=settings.claude_vision_model,
        max_tokens=4096,
        system=system_prompt,
        messages=[{"role": "user", "content": content}],
    )
    return resp.content[0].text


# ── Image editing ─────────────────────────────────────────────────────


async def _call_edit_image(provider: str, prompt: str, image_b64: str) -> str:
    """Dispatch an image edit call to the named provider.

    Claude doesn't have a native image generation API, so it falls
    through to OpenAI or Gemini for image editing.
    """
    if provider == "gemini":
        return await _gemini_edit_image(prompt, image_b64)
    return await _openai_edit_image(prompt, image_b64)


async def edit_image(prompt: str, image_b64: str) -> str:
    """Edit an image and return the result as a base-64 encoded PNG string.

    Falls back to the alternate provider if the primary fails and the
    alternate has an API key configured.
    """
    provider = settings.image_provider
    limiter = _get_limiter(provider)
    await limiter.acquire()
    metrics.llm_calls_total.labels(provider=provider, call_type="edit_image").inc()
    start = time.monotonic()

    try:
        result = await _call_edit_image(provider, prompt, image_b64)
        duration = time.monotonic() - start
        metrics.llm_call_duration.labels(provider=provider, call_type="edit_image").observe(duration)
        logger.info("edit_image_completed", provider=provider, duration_s=round(duration, 2))
        await limiter.on_success()
        return result

    except Exception as exc:
        duration = time.monotonic() - start
        metrics.llm_errors_total.labels(provider=provider, call_type="edit_image", error_type=type(exc).__name__).inc()
        logger.error("edit_image_failed", provider=provider, error=str(exc), duration_s=round(duration, 2))

        if _is_rate_limit_error(exc):
            await limiter.backoff()

        # Attempt fallback -- skip Claude for image editing (no native API)
        for fallback in _fallback_providers(provider):
            if fallback == "claude":
                continue
            fb_limiter = _get_limiter(fallback)
            await fb_limiter.acquire()
            logger.warning("edit_image_fallback", primary=provider, fallback=fallback)
            metrics.llm_calls_total.labels(provider=fallback, call_type="edit_image").inc()
            start2 = time.monotonic()
            try:
                result = await _call_edit_image(fallback, prompt, image_b64)
                duration2 = time.monotonic() - start2
                metrics.llm_call_duration.labels(provider=fallback, call_type="edit_image").observe(duration2)
                metrics.provider_fallbacks.labels(primary=provider, fallback=fallback, call_type="edit_image").inc()
                logger.info("edit_image_fallback_succeeded", fallback=fallback, duration_s=round(duration2, 2))
                await fb_limiter.on_success()
                return result
            except Exception as fallback_exc:
                metrics.llm_errors_total.labels(provider=fallback, call_type="edit_image", error_type=type(fallback_exc).__name__).inc()
                logger.error("edit_image_fallback_failed", fallback=fallback, error=str(fallback_exc))

        raise


async def _gemini_edit_image(prompt: str, image_b64: str) -> str:
    """Gemini image-editing with automatic retry on empty responses."""
    from google.genai import types

    client = _get_gemini()
    image_bytes = base64.b64decode(image_b64)

    logger.info("gemini_edit_image_starting", prompt_length=len(prompt), prompt_preview=prompt[:200])

    def _sync_generate():
        return client.models.generate_content(
            model=settings.gemini_image_model,
            contents=[
                types.Part.from_text(text=prompt),
                types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
            ],
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"],
            ),
        )

    last_error: str | None = None
    loop = asyncio.get_event_loop()

    for attempt in range(3):
        resp = await loop.run_in_executor(None, _sync_generate)

        # Extract the first image part from the response
        if resp.candidates and resp.candidates[0].content and resp.candidates[0].content.parts:
            for part in resp.candidates[0].content.parts:
                if part.inline_data and part.inline_data.mime_type.startswith("image/"):
                    return base64.b64encode(part.inline_data.data).decode()

        reason = (
            getattr(resp.candidates[0], "finish_reason", "unknown")
            if resp.candidates
            else "no candidates"
        )
        last_error = f"Gemini returned no image (reason: {reason})"
        logger.warning("edit_image_retry", attempt=attempt + 1, reason=str(reason))
        # Signal rate limit backoff for IMAGE_OTHER (likely throttling)
        await _get_limiter("gemini").backoff()
        # Wait before retry -- let the adaptive limiter settle
        await asyncio.sleep(10 * (attempt + 1))

    raise RuntimeError(last_error)


async def _openai_edit_image(prompt: str, image_b64: str) -> str:
    """OpenAI image-editing (supports gpt-image-1 and dall-e models)."""
    client = _get_openai()
    image_bytes = base64.b64decode(image_b64)

    kwargs: dict[str, Any] = {
        "model": settings.openai_image_model,
        "image": image_bytes,
        "prompt": prompt,
        "n": 1,
        "size": settings.image_size,
    }
    # dall-e models return base64 directly; newer models may return a URL
    if settings.openai_image_model.startswith("dall-e"):
        kwargs["response_format"] = "b64_json"

    resp = await client.images.edit(**kwargs)

    if resp.data[0].b64_json:
        return resp.data[0].b64_json
    if resp.data[0].url:
        import httpx

        async with httpx.AsyncClient() as http:
            r = await http.get(resp.data[0].url)
            return base64.b64encode(r.content).decode()

    raise RuntimeError("Image edit returned neither b64_json nor url")
