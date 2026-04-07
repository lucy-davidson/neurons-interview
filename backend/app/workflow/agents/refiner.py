"""Refiner agent -- translates critic feedback into a revised edit prompt.

When the critic rejects an edited image, the refiner analyses what went
wrong and produces a new edit_prompt that addresses the feedback. This
revised prompt goes directly to the editor for the next attempt.
"""

from __future__ import annotations

import structlog

from app import metrics
from app.models import AuditEntry
from app.services.llm import vision_chat
from app.workflow.state import RecommendationState

logger = structlog.get_logger()

SYSTEM_PROMPT = """\
You are a Refinement Strategist for marketing creative editing.

You will receive:
1. The original recommendation and variant description.
2. The edit prompt that was used for the previous attempt.
3. The critic's evaluation feedback on the result.
4. Brand guidelines that must be respected.

Your job is to produce a REVISED EDIT PROMPT that addresses the critic's
concerns. The revised prompt will be sent directly to an AI image editor,
so it must be self-contained and describe the desired final image.

Be specific about what to change compared to the previous attempt.
Keep the revised prompt to 1-3 sentences.
"""


async def run_refiner(state: RecommendationState) -> RecommendationState:
    """Produce a revised edit_prompt based on critic feedback and increment the attempt counter."""
    metrics.agent_invocations.labels(agent="refiner").inc()

    attempt = state.get("attempt", 1)

    user_text = (
        f"**Original recommendation:** {state['recommendation_title']}\n"
        f"{state['recommendation_description']}\n\n"
        f"**Previous edit prompt:** {state.get('edit_prompt', 'N/A')}\n\n"
        f"**Critic feedback:** {state.get('evaluation_feedback', 'N/A')}\n\n"
        f"**Brand guidelines:** {state['brand_guidelines_text']}\n\n"
        "Write a revised edit prompt that addresses the critic's concerns."
    )

    revised_prompt = await vision_chat(
        system_prompt=SYSTEM_PROMPT,
        user_text=user_text,
        image_b64=state.get("edited_image_b64"),
    )

    trail = list(state.get("audit_trail", []))
    trail.append(
        AuditEntry(
            agent="refiner",
            action="revised_edit_prompt",
            detail=f"Attempt {attempt}. Revised prompt: {revised_prompt[:300]}",
        )
    )

    return {
        **state,
        "edit_prompt": revised_prompt.strip(),
        "refiner_feedback": revised_prompt.strip(),
        "attempt": attempt + 1,
        "audit_trail": trail,
    }
