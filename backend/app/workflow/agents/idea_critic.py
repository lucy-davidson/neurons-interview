"""Idea Critic agent -- reviews variant ideas before they enter the pipeline.

Evaluates all variant ideas together and rejects weak ones, providing
feedback so the ideator can generate better replacements. This catches
bad ideas early, before expensive image generation.
"""

from __future__ import annotations

import json

import structlog

from app import metrics
from app.services.llm import vision_chat

logger = structlog.get_logger()

SYSTEM_PROMPT = """\
You are a senior Creative Director reviewing variant ideas before they
are sent to an AI image editor.

You will receive:
1. The original marketing creative image.
2. The recommendation that needs to be applied.
3. Brand guidelines that must be respected.
4. A list of proposed variant ideas, each with a title, description,
   and edit_prompt.

Review ALL variants together and evaluate each one on:
A) **Visual impact** – Will this change be CLEARLY VISIBLE to a human
   viewer? Reject ideas that would produce subtle or imperceptible
   changes, such as "slightly increase letter spacing", "marginally
   bolder font", or "minor color adjustment". The change must be
   obvious and impactful.
B) **Recommendation alignment** – Will this idea achieve what the
   recommendation asks for?
C) **Brand compliance** – Does this idea risk violating brand constraints?
D) **Distinctiveness** – Is this idea meaningfully different from the
   others, or is it a near-duplicate?
E) **Feasibility** – Can an AI image editor realistically execute this?

For each variant, decide: approve or reject.

Respond in valid JSON:
{
  "reviews": [
    {"title": "...", "approved": true},
    {"title": "...", "approved": false, "reason": "Too similar to variant 1..."},
    ...
  ],
  "overall_feedback": "Summary of what's missing or could be improved"
}
No extra text outside the JSON.
"""


def _strip_code_fences(text: str) -> str:
    """Remove markdown code fences that LLMs often wrap around JSON."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]
    if cleaned.endswith("```"):
        cleaned = cleaned.rsplit("```", 1)[0]
    return cleaned.strip()


async def run_idea_critic(
    recommendation_title: str,
    recommendation_description: str,
    recommendation_type: str,
    brand_guidelines_text: str,
    image_b64: str,
    variants: list[dict[str, str]],
    runtime_config: object | None = None,
) -> tuple[list[dict[str, str]], list[dict[str, str]], str]:
    """Review variant ideas and split into approved and rejected lists.

    Returns:
        (approved, rejected, feedback)
        - approved: list of variant dicts that passed review
        - rejected: list of variant dicts that were rejected
        - feedback: overall feedback string for the ideator
    """
    metrics.agent_invocations.labels(agent="idea_critic").inc()

    variants_text = "\n".join(
        f"{i+1}. **{v['title']}**\n"
        f"   Description: {v['description']}\n"
        f"   Edit prompt: {v.get('edit_prompt', 'N/A')}"
        for i, v in enumerate(variants)
    )

    user_text = (
        f"**Recommendation ({recommendation_type}):** "
        f"{recommendation_title}\n{recommendation_description}\n\n"
        f"**Brand guidelines:**\n{brand_guidelines_text}\n\n"
        f"**Proposed variants:**\n{variants_text}\n\n"
        "Review each variant. Approve good ones, reject weak or duplicative ones."
    )

    raw = await vision_chat(
        system_prompt=SYSTEM_PROMPT,
        user_text=user_text,
        image_b64=image_b64,
        runtime_config=runtime_config,
    )

    cleaned = _strip_code_fences(raw)

    try:
        parsed = json.loads(cleaned)
        reviews = parsed.get("reviews", [])
        overall_feedback = parsed.get("overall_feedback", "")

        # Match reviews back to variants by index (same order)
        approved = []
        rejected = []
        for i, v in enumerate(variants):
            if i < len(reviews) and not reviews[i].get("approved", True):
                # Add rejection reason to the variant for context
                v_copy = dict(v)
                v_copy["rejection_reason"] = reviews[i].get("reason", "")
                rejected.append(v_copy)
            else:
                approved.append(v)

        logger.info(
            "idea_critic_completed",
            total=len(variants),
            approved=len(approved),
            rejected=len(rejected),
        )

        return approved, rejected, overall_feedback

    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        metrics.llm_parse_failures.labels(agent="idea_critic").inc()
        logger.warning("idea_critic_parse_failure", error=str(exc), raw_prefix=raw[:300])
        # On failure, approve all variants (don't block the pipeline)
        return variants, [], ""
