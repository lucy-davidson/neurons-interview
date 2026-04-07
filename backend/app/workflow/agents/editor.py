"""Editor agent -- calls the image-editing model to produce a visual variant.

This is a thin orchestration layer: it takes the ideator's edit prompt,
sends it to the configured image provider, and records the result in the
audit trail.
"""

from __future__ import annotations

import structlog

from app import metrics
from app.models import AuditEntry
from app.services.llm import edit_image
from app.workflow.state import RecommendationState

logger = structlog.get_logger()


async def run_editor(state: RecommendationState) -> RecommendationState:
    """Generate an edited image using the ideator's edit prompt."""
    metrics.agent_invocations.labels(agent="editor").inc()

    attempt = state.get("attempt", 1)
    edit_prompt = state["edit_prompt"]

    logger.info(
        "editor_generating_image",
        recommendation_id=state["recommendation_id"],
        attempt=attempt,
    )

    edited_b64 = await edit_image(
        prompt=edit_prompt,
        image_b64=state["original_image_b64"],
        runtime_config=state.get("runtime_config"),
    )

    trail = list(state.get("audit_trail", []))
    trail.append(
        AuditEntry(
            agent="editor",
            action="image_generated",
            detail=f"Attempt {attempt}. Prompt: {edit_prompt[:200]}",
        )
    )

    return {
        **state,
        "edited_image_b64": edited_b64,
        "audit_trail": trail,
    }
