"""LangGraph workflow for processing visual recommendation variants.

Graph topology (per variant):
    editor -> critic -> should_retry?
                          |-- yes -> refiner -> editor (loop, max 3 attempts)
                          +-- no  -> END

Outer orchestration:
    ideator (produces 5 variant ideas with edit prompts)
    -> idea_critic (reviews all 5, rejects weak ones)
    -> ideator (replaces rejected ones, up to 2 rounds)
    -> run each approved variant through the graph above
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Literal

import structlog

from langgraph.graph import END, StateGraph

from app import metrics
from app.models import (
    AuditEntry,
    BrandGuidelines,
    Recommendation,
    RecommendationResult,
    VariantResult,
)
from app.workflow.agents.critic import run_critic
from app.workflow.agents.editor import run_editor
from app.workflow.agents.idea_critic import run_idea_critic
from app.workflow.agents.ideator import run_ideator, run_ideator_replacements
from app.workflow.agents.refiner import run_refiner
from app.workflow.state import RecommendationState

logger = structlog.get_logger()

MAX_IDEA_REVISIONS = 2  # Max rounds of idea critique before proceeding


def _pick_rollout_config_for_variant(configs: list[dict]) -> dict | None:
    """Weighted random selection of a rollout config for a single variant."""
    import random
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


# ── Routing logic ───────────────────────────────────────────────────


def should_retry(state: RecommendationState) -> Literal["refiner", "end"]:
    """Decide whether to loop back for another edit attempt or finish."""
    if state.get("evaluation_passed"):
        return "end"
    if state.get("attempt", 1) >= state.get("max_attempts", 3):
        return "end"
    return "refiner"


# ── Build the graph ─────────────────────────────────────────────────


def build_recommendation_graph() -> StateGraph:
    """Construct the LangGraph state graph for a single variant.

    The graph is simpler now that the ideator produces the edit_prompt
    directly. The per-variant graph only handles: edit -> evaluate -> retry.
    """
    graph = StateGraph(RecommendationState)

    graph.add_node("editor", run_editor)
    graph.add_node("critic", run_critic)
    graph.add_node("refiner", run_refiner)

    graph.set_entry_point("editor")
    graph.add_edge("editor", "critic")
    graph.add_conditional_edges("critic", should_retry, {"refiner": "refiner", "end": END})
    graph.add_edge("refiner", "editor")

    return graph


# Compile once at module level -- reused across all variant invocations
_compiled_graph = build_recommendation_graph().compile()

# ── Internal helpers ────────────────────────────────────────────────


def _serialize_guidelines(guidelines: BrandGuidelines) -> str:
    """Convert brand guidelines to a JSON string for embedding in prompts."""
    return json.dumps(guidelines.model_dump(), indent=2)


async def _run_variant(
    image_b64: str,
    recommendation: Recommendation,
    variant: dict[str, str],
    variant_id: str,
    brand_guidelines_text: str,
    max_attempts: int,
    runtime_config: Any | None = None,
) -> VariantResult:
    """Execute the edit-evaluate-refine loop for one variant idea."""
    import time as _time

    initial_state: RecommendationState = {
        "original_image_b64": image_b64,
        "recommendation_id": recommendation.id,
        "recommendation_title": variant["title"],
        "recommendation_description": variant["description"],
        "recommendation_type": recommendation.type,
        "brand_guidelines_text": brand_guidelines_text,
        "runtime_config": runtime_config,
        "plan": "",
        "edit_prompt": variant.get("edit_prompt", variant["description"]),
        "plan_approved": True,
        "plan_feedback": "",
        "plan_revision_count": 0,
        "edited_image_b64": "",
        "evaluation_passed": False,
        "evaluation_score": 0.0,
        "evaluation_feedback": "",
        "refiner_feedback": "",
        "attempt": 1,
        "max_attempts": max_attempts,
        "audit_trail": [],
        "status": "",
        "agent_timings": [],
        "critic_evaluations": [],
        "total_tokens": 0,
        "total_cost_usd": 0.0,
    }

    logger.info(
        "variant_starting",
        variant_id=variant_id,
        recommendation_id=recommendation.id,
        variant_title=variant["title"],
    )

    variant_start = _time.monotonic()
    final_state = await _compiled_graph.ainvoke(initial_state)
    variant_duration = _time.monotonic() - variant_start

    status = "accepted" if final_state.get("evaluation_passed") else "max_retries_exceeded"
    total_tokens = final_state.get("total_tokens") or 0
    total_cost = final_state.get("total_cost_usd") or 0.0

    metrics.variant_duration.observe(variant_duration)
    logger.info(
        "variant_duration",
        variant_id=variant_id,
        duration_s=round(variant_duration, 2),
        total_tokens=total_tokens,
        total_cost_usd=round(total_cost, 6),
        attempts=final_state.get("attempt", 1),
        final_status=status,
    )

    # Use runtime_config if available, fall back to global settings
    from app.config import settings as _settings
    rc = runtime_config
    text_provider = rc.text_provider if rc else _settings.text_provider
    image_provider = rc.image_provider if rc else _settings.image_provider
    text_model = rc.text_model if rc else (
        getattr(_settings, f"{_settings.text_provider}_vision_model", _settings.text_provider)
        if _settings.text_provider != "claude" else _settings.claude_vision_model
    )
    image_model = rc.image_model if rc else getattr(_settings, f"{_settings.image_provider}_image_model", _settings.image_provider)

    # Import prompt versions lazily to avoid circular imports
    from app.workflow.prompt_versions import PROMPT_VERSIONS

    return VariantResult(
        variant_id=variant_id,
        variant_title=variant["title"],
        variant_description=variant["description"],
        status=status,
        attempts=final_state.get("attempt", 1),
        edited_image_b64=final_state.get("edited_image_b64"),
        evaluation_score=final_state.get("evaluation_score"),
        evaluation_feedback=final_state.get("evaluation_feedback"),
        audit_trail=final_state.get("audit_trail", []),
        text_provider=text_provider,
        image_provider=image_provider,
        text_model=text_model,
        image_model=image_model,
        duration_s=round(variant_duration, 2),
        total_tokens=total_tokens,
        total_cost_usd=round(total_cost, 6),
        agent_timings=final_state.get("agent_timings", []),
        prompt_versions=PROMPT_VERSIONS,
        critic_evaluations=final_state.get("critic_evaluations", []),
    )


# ── Public orchestration ────────────────────────────────────────────


async def run_recommendation_workflow(
    image_b64: str,
    recommendation: Recommendation,
    brand_guidelines: BrandGuidelines,
    max_attempts: int = 3,
    rec_result: RecommendationResult | None = None,
    on_variant_complete: Any | None = None,
    runtime_config: Any | None = None,
    rollout_configs: list[dict] | None = None,
) -> RecommendationResult:
    """Ideate, critique ideas, then run each approved variant through the graph.

    The idea critique loop:
    1. Ideator generates 5 variant ideas (with edit prompts)
    2. Idea Critic reviews all 5 together, rejects weak ones
    3. Ideator generates replacements for rejected ones
    4. Repeat up to MAX_IDEA_REVISIONS times
    5. Run all approved variants through the edit-evaluate-refine graph
    """
    guidelines_text = _serialize_guidelines(brand_guidelines)

    if rec_result is None:
        rec_result = RecommendationResult(
            recommendation_id=recommendation.id,
            recommendation_title=recommendation.title,
        )

    # Step 1: Ideate -- produce variant ideas with edit prompts
    logger.info("ideating_variants", recommendation_id=recommendation.id)
    from app.config import settings as _settings

    # Generate a pool of ideas upfront (10), then try them in batches
    POOL_SIZE = 10
    rc = runtime_config
    BATCH_SIZE = rc.num_variants if rc else _settings.num_variants  # how many to try at a time (default 2)

    idea_pool = await run_ideator(
        recommendation_title=recommendation.title,
        recommendation_description=recommendation.description,
        recommendation_type=recommendation.type,
        brand_guidelines_text=guidelines_text,
        image_b64=image_b64,
        num_variants=POOL_SIZE,
        runtime_config=runtime_config,
    )
    logger.info("idea_pool_generated", pool_size=len(idea_pool), recommendation_id=recommendation.id)

    # Step 2: Idea critique -- review the pool and remove weak ideas
    approved, rejected, feedback = await run_idea_critic(
        recommendation_title=recommendation.title,
        recommendation_description=recommendation.description,
        recommendation_type=recommendation.type,
        brand_guidelines_text=guidelines_text,
        image_b64=image_b64,
        variants=idea_pool,
        runtime_config=runtime_config,
    )
    # Use approved ideas as the pool; if critic rejected too many, keep originals
    pool = approved if len(approved) >= BATCH_SIZE else idea_pool
    logger.info("idea_pool_curated", pool_size=len(pool), rejected=len(rejected))

    # Step 3: Draw from pool in batches, stop when we get accepted variants
    variant_counter = 0
    pool_index = 0

    async def _process_variant(idx: int, v: dict) -> VariantResult | None:
        try:
            # Per-variant config: if rollout configs exist, each variant
            # independently picks a config. This means a single job can
            # show variants from different models side by side, so users
            # always see a mix during A/B testing.
            variant_rc = runtime_config
            if rollout_configs:
                from app.config import ConfigOverrides, RuntimeConfig as _RC
                selected = _pick_rollout_config_for_variant(rollout_configs)
                if selected:
                    variant_rc = _RC(ConfigOverrides(**{
                        k: val for k, val in selected["config"].items()
                        if k in ConfigOverrides.model_fields
                    }))
                    logger.info("variant_config_assigned",
                                variant_idx=idx, config_name=selected["name"])

            vr = await _run_variant(
                image_b64=image_b64,
                recommendation=recommendation,
                variant=v,
                variant_id=f"{recommendation.id}_v{idx + 1}",
                brand_guidelines_text=guidelines_text,
                max_attempts=max_attempts,
                runtime_config=variant_rc,
            )
            rec_result.variants.append(vr)
            metrics.variants_processed.labels(status=vr.status).inc()
            if vr.evaluation_score is not None:
                metrics.variant_score.observe(vr.evaluation_score)
            metrics.variant_attempts.observe(vr.attempts)
            logger.info("variant_completed", variant_id=vr.variant_id, variant_title=vr.variant_title, status=vr.status)
            if on_variant_complete:
                await on_variant_complete()
            return vr
        except Exception as exc:
            metrics.variants_processed.labels(status="error").inc()
            metrics.llm_errors_total.labels(
                provider="unknown", call_type="variant_pipeline", error_type=type(exc).__name__,
            ).inc()
            logger.error(
                "variant_failed",
                variant_number=idx + 1,
                variant_title=v.get("title", ""),
                error=str(exc),
                exc_info=True,
            )
            return None

    while pool_index < len(pool):
        # Take the next batch from the pool
        batch = pool[pool_index:pool_index + BATCH_SIZE]
        pool_index += len(batch)

        logger.info(
            "trying_batch",
            batch_size=len(batch),
            tried=variant_counter,
            remaining_in_pool=len(pool) - pool_index,
            titles=[v["title"] for v in batch],
        )

        # Run batch concurrently
        await asyncio.gather(*[
            _process_variant(variant_counter + i, v) for i, v in enumerate(batch)
        ])
        variant_counter += len(batch)

        # Check if we have enough accepted variants
        accepted = [v for v in rec_result.variants if v.status == "accepted"]
        if len(accepted) >= BATCH_SIZE:
            logger.info("target_reached", accepted=len(accepted), target=BATCH_SIZE, total_tried=variant_counter)
            break

        logger.info("batch_progress", accepted=len(accepted), target=BATCH_SIZE, total_tried=variant_counter, pool_remaining=len(pool) - pool_index)

    if not any(v.status == "accepted" for v in rec_result.variants):
        logger.warning("all_pool_exhausted", total_tried=variant_counter, recommendation_id=recommendation.id)

    return rec_result


async def run_all_recommendations(
    image_b64: str,
    recommendations: list[Recommendation],
    brand_guidelines: BrandGuidelines,
    max_attempts: int = 3,
    job_results: list[RecommendationResult] | None = None,
    on_variant_complete: Any | None = None,
    runtime_config: Any | None = None,
    rollout_configs: list[dict] | None = None,
) -> list[RecommendationResult]:
    """Process every recommendation concurrently.

    Variant results stream into *job_results* incrementally so the
    polling endpoint can expose partial progress.
    """
    if job_results is None:
        job_results = []

    # Pre-create result objects so they appear in API responses immediately
    rec_results: list[RecommendationResult] = []
    for rec in recommendations:
        rr = RecommendationResult(
            recommendation_id=rec.id,
            recommendation_title=rec.title,
        )
        rec_results.append(rr)
        job_results.append(rr)

    # Process recommendations concurrently -- each internally runs variants sequentially
    tasks = [
        run_recommendation_workflow(
            image_b64, rec, brand_guidelines, max_attempts,
            rec_result=rr, on_variant_complete=on_variant_complete,
            runtime_config=runtime_config,
            rollout_configs=rollout_configs,
        )
        for rec, rr in zip(recommendations, rec_results)
    ]
    await asyncio.gather(*tasks, return_exceptions=False)

    return job_results
