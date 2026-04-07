"""Ideator agent -- turns a vague recommendation into concrete variant ideas.

Given a marketing creative and a high-level recommendation, the ideator
brainstorms specific, actionable variant ideas that each interpret the
recommendation differently while respecting brand guidelines.

Each variant includes a title, description, AND a concrete edit prompt
ready to send to the image editor.
"""

from __future__ import annotations

import json

import structlog

from app import metrics
from app.services.llm import vision_chat

logger = structlog.get_logger()

SYSTEM_PROMPT = """\
You are an expert Creative Director specialising in marketing visuals.

You will receive:
1. A marketing creative image.
2. A textual recommendation describing a desired visual improvement.
3. Brand guidelines that MUST be respected.

Your job is to brainstorm specific, concrete variant ideas that fulfil
the recommendation. Each variant should be a distinct, actionable
interpretation of the recommendation.

For each variant, provide:
- A short title (5-10 words)
- A detailed description of exactly what visual changes to make (2-3 sentences)
- An edit_prompt: a single, self-contained instruction (1-3 sentences)
  that will be sent directly to an AI image editor. The edit_prompt must
  describe the desired final image, incorporating the changes while
  preserving brand-protected elements.

Be specific: name colours, positions, sizes, effects. Each variant should
be meaningfully different from the others while all staying true to the
recommendation intent and brand guidelines.

Respond in valid JSON:
{
  "variants": [
    {"title": "...", "description": "...", "edit_prompt": "..."},
    {"title": "...", "description": "...", "edit_prompt": "..."},
    ...
  ]
}
No extra text outside the JSON.
"""

REPLACEMENT_PROMPT = """\
You are an expert Creative Director specialising in marketing visuals.

You previously brainstormed variant ideas for a recommendation, but some
were rejected by a reviewer. You need to generate replacement ideas.

You will receive:
1. A marketing creative image.
2. The original recommendation.
3. Brand guidelines.
4. The variants that were APPROVED (keep these in mind to avoid duplicates).
5. The variants that were REJECTED with the reviewer's feedback.

Generate replacement variant ideas for each rejected one. Each replacement
must be meaningfully different from both the approved variants AND the
rejected ones. Include title, description, and edit_prompt for each.

Respond in valid JSON:
{
  "variants": [
    {"title": "...", "description": "...", "edit_prompt": "..."},
    ...
  ]
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


def _parse_variants(raw: str) -> list[dict[str, str]]:
    """Parse LLM response into a list of variant dicts."""
    cleaned = _strip_code_fences(raw)
    parsed = json.loads(cleaned)
    variants = parsed["variants"]
    if not isinstance(variants, list) or len(variants) == 0:
        raise ValueError("Empty variants list")
    # Ensure each variant has an edit_prompt (fallback to description)
    for v in variants:
        if "edit_prompt" not in v or not v["edit_prompt"]:
            v["edit_prompt"] = v.get("description", "")
    return variants


async def run_ideator(
    recommendation_title: str,
    recommendation_description: str,
    recommendation_type: str,
    brand_guidelines_text: str,
    image_b64: str,
    num_variants: int = 5,
) -> list[dict[str, str]]:
    """Generate variant ideas for a recommendation.

    Returns a list of dicts, each with "title", "description", and "edit_prompt".
    Falls back to a single pass-through variant if the LLM output is
    unparseable, so the downstream workflow always has something to run.
    """
    metrics.agent_invocations.labels(agent="ideator").inc()

    MAX_RETRIES = 3
    last_error = None

    for attempt in range(MAX_RETRIES):
        user_text = (
            f"**Recommendation ({recommendation_type}):** "
            f"{recommendation_title}\n{recommendation_description}\n\n"
            f"**Brand guidelines:**\n{brand_guidelines_text}\n\n"
            f"Generate {num_variants} specific, concrete variant ideas that fulfil this recommendation."
        )
        # On retry, add a hint about the format
        if attempt > 0:
            user_text += (
                "\n\nIMPORTANT: Your previous response could not be parsed. "
                "Make sure to respond with ONLY valid JSON, no extra text. "
                "Keep descriptions shorter to avoid truncation."
            )

        raw = await vision_chat(
            system_prompt=SYSTEM_PROMPT,
            user_text=user_text,
            image_b64=image_b64,
        )

        try:
            return _parse_variants(raw)[:num_variants]
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            last_error = exc
            metrics.llm_parse_failures.labels(agent="ideator").inc()
            logger.warning(
                "ideator_parse_failure",
                attempt=attempt + 1,
                max_retries=MAX_RETRIES,
                error=str(exc),
                raw_prefix=raw[:300],
            )

    # All retries exhausted -- fall back to original recommendation
    logger.error("ideator_all_retries_failed", error=str(last_error))
    return [
        {
            "title": recommendation_title,
            "description": recommendation_description,
            "edit_prompt": f"Apply this recommendation to the image: {recommendation_description}",
        }
    ]


async def run_ideator_replacements(
    recommendation_title: str,
    recommendation_description: str,
    recommendation_type: str,
    brand_guidelines_text: str,
    image_b64: str,
    approved: list[dict[str, str]],
    rejected: list[dict[str, str]],
    rejection_feedback: str,
) -> list[dict[str, str]]:
    """Generate replacement variants for rejected ideas.

    Takes the approved and rejected lists so replacements don't
    duplicate existing ideas.
    """
    metrics.agent_invocations.labels(agent="ideator").inc()

    approved_text = "\n".join(
        f"- {v['title']}: {v['description']}" for v in approved
    ) or "None yet"

    rejected_text = "\n".join(
        f"- {v['title']}: {v['description']}" for v in rejected
    )

    user_text = (
        f"**Recommendation ({recommendation_type}):** "
        f"{recommendation_title}\n{recommendation_description}\n\n"
        f"**Brand guidelines:**\n{brand_guidelines_text}\n\n"
        f"**Approved variants (do NOT duplicate these):**\n{approved_text}\n\n"
        f"**Rejected variants:**\n{rejected_text}\n\n"
        f"**Reviewer feedback:** {rejection_feedback}\n\n"
        f"Generate {len(rejected)} replacement variant ideas that address the feedback "
        f"and are different from both the approved and rejected variants."
    )

    num_needed = len(rejected)
    for attempt in range(3):
        if attempt > 0:
            user_text += "\n\nIMPORTANT: Respond with ONLY valid JSON. Keep descriptions short."

        raw = await vision_chat(
            system_prompt=REPLACEMENT_PROMPT,
            user_text=user_text,
            image_b64=image_b64,
        )

        try:
            return _parse_variants(raw)[:num_needed]
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            metrics.llm_parse_failures.labels(agent="ideator").inc()
            logger.warning("ideator_replacement_parse_failure", attempt=attempt + 1, error=str(exc), raw_prefix=raw[:300])

    # All retries exhausted
    return rejected
