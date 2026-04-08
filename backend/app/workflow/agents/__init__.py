"""Agent utilities — timing decorator and token tracking helpers."""

from __future__ import annotations

import functools
import time
from typing import Any

import structlog

from app import metrics
from app.services.llm import LLMCallRecord, _llm_call_record

logger = structlog.get_logger()


def timed_agent(agent_name: str, prompt_version: str = ""):
    """Decorator that times a LangGraph agent node and tracks its token usage.

    Sets up a fresh :class:`LLMCallRecord` scope so all LLM calls within
    the agent accumulate into a single record.  After the agent returns,
    emits a structured log event and records Prometheus metrics.
    """

    def decorator(fn):
        @functools.wraps(fn)
        async def wrapper(state: dict, *args: Any, **kwargs: Any) -> dict:
            record = LLMCallRecord()
            token = _llm_call_record.set(record)
            start = time.monotonic()
            try:
                result = await fn(state, *args, **kwargs)
                duration = time.monotonic() - start

                # Prometheus
                metrics.agent_duration.labels(agent=agent_name).observe(duration)
                metrics.agent_tokens.labels(agent=agent_name, token_type="prompt").inc(record.total_prompt_tokens)
                metrics.agent_tokens.labels(agent=agent_name, token_type="completion").inc(record.total_completion_tokens)

                # Structured log
                logger.info(
                    "agent_completed",
                    agent=agent_name,
                    duration_s=round(duration, 3),
                    total_tokens=record.total_tokens,
                    prompt_tokens=record.total_prompt_tokens,
                    completion_tokens=record.total_completion_tokens,
                    cost_usd=round(record.total_cost_usd, 6),
                    prompt_version=prompt_version,
                )

                # Append timing record to state for VariantResult
                timings = list(result.get("agent_timings") or state.get("agent_timings") or [])
                timings.append({
                    "agent": agent_name,
                    "duration_s": round(duration, 3),
                    "tokens": record.total_tokens,
                    "cost_usd": round(record.total_cost_usd, 6),
                    "prompt_version": prompt_version,
                })
                result["agent_timings"] = timings

                # Accumulate totals
                result["total_tokens"] = (result.get("total_tokens") or state.get("total_tokens") or 0) + record.total_tokens
                result["total_cost_usd"] = (result.get("total_cost_usd") or state.get("total_cost_usd") or 0.0) + record.total_cost_usd

                return result
            finally:
                _llm_call_record.reset(token)

        return wrapper

    return decorator
