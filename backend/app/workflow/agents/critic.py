"""Critic agent -- evaluates an edited image against the original
recommendation and brand guidelines using a vision-language model.

Includes a fast SSIM pre-check to catch identical images before
wasting an LLM call. Provides the pass/fail signal and numeric score
that drive the retry-or-accept routing decision in the workflow graph.
"""

from __future__ import annotations

import base64
import io
import json

import numpy as np
import structlog
from PIL import Image
from skimage.metrics import structural_similarity as ssim

from app import metrics
from app.models import AuditEntry
from app.services.llm import vision_chat
from app.workflow.state import RecommendationState

logger = structlog.get_logger()

SYSTEM_PROMPT = """\
You are an expert visual QA critic for marketing creatives.

You will receive:
1. The ORIGINAL marketing image.
2. The EDITED marketing image (a variant).
3. The recommendation that was supposed to be applied.
4. The brand guidelines that must be respected.

IMPORTANT: Your FIRST task before anything else is to determine whether
the edited image is actually different from the original. Many AI image
editors return the original image unchanged or with imperceptible
modifications. You must catch this.

Step 1: **Visible difference check (MANDATORY)**
Look at both images carefully. Ask yourself: "If I showed these two
images to 10 random people, would at least 8 of them notice a
difference?" If the answer is NO, the variant MUST FAIL with
passed=false and score=0.0, regardless of everything else.

Common failures to watch for:
- Image looks identical (editor returned the original unchanged)
- Only tiny color shifts that aren't noticeable at normal viewing size
- Text changes that are too subtle (slightly bolder, minor spacing)
- Minor saturation or contrast tweaks that don't change the feel

Step 2: **Recommendation compliance** – If the image IS visibly
different, was the recommendation correctly applied? The change should
be obvious and match the intent.

Step 3: **Brand guideline compliance** – Were all brand constraints
respected? Check protected regions, typography, aspect ratio, and brand
elements.

Respond in valid JSON:
{
  "passed": true/false,
  "score": 0.0-1.0,
  "visually_different": true/false,
  "recommendation_compliance": "...",
  "brand_compliance": "...",
  "feedback": "Specific actionable feedback for improvement (if not passed)"
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


SSIM_THRESHOLD = 0.95  # above this, images are considered too similar


def _compute_ssim(original_b64: str, edited_b64: str) -> float:
    """Compute SSIM between two base64-encoded images. Returns 0.0-1.0."""
    try:
        orig = Image.open(io.BytesIO(base64.b64decode(original_b64))).convert("RGB")
        edit = Image.open(io.BytesIO(base64.b64decode(edited_b64))).convert("RGB")
        # Resize to same dimensions for comparison
        edit = edit.resize(orig.size)
        # Downscale for speed (SSIM on full-res is slow)
        max_dim = 512
        if orig.width > max_dim or orig.height > max_dim:
            ratio = max_dim / max(orig.width, orig.height)
            new_size = (int(orig.width * ratio), int(orig.height * ratio))
            orig = orig.resize(new_size)
            edit = edit.resize(new_size)
        orig_arr = np.array(orig)
        edit_arr = np.array(edit)
        score = ssim(orig_arr, edit_arr, channel_axis=2)
        return float(score)
    except Exception:
        return 0.0  # on error, assume images are different


async def run_critic(state: RecommendationState) -> RecommendationState:
    """Compare the edited image to the original and score compliance.

    Runs a fast SSIM check first -- if images are nearly identical,
    auto-rejects without calling the LLM.
    """
    metrics.agent_invocations.labels(agent="critic").inc()

    attempt = state.get("attempt", 1)

    # Fast SSIM pre-check -- catch identical images without an LLM call
    ssim_score = _compute_ssim(state["original_image_b64"], state["edited_image_b64"])
    logger.info("critic_ssim_check", ssim=round(ssim_score, 4), threshold=SSIM_THRESHOLD, attempt=attempt)

    if ssim_score > SSIM_THRESHOLD:
        trail = list(state.get("audit_trail", []))
        trail.append(AuditEntry(
            agent="critic",
            action="auto_rejected_identical",
            detail=f"Attempt {attempt}. SSIM={ssim_score:.4f} > {SSIM_THRESHOLD} — images are visually identical. Skipped LLM evaluation.",
        ))
        return {
            **state,
            "evaluation_passed": False,
            "evaluation_score": 0.0,
            "evaluation_feedback": f"No visible difference from original (SSIM={ssim_score:.3f}). The image editor did not make a meaningful change.",
            "audit_trail": trail,
        }

    # Step 1: BLIND comparison -- ask what's different WITHOUT revealing
    # what the edit was supposed to be. This prevents anchoring bias.
    blind_prompt = (
        "You are a visual comparison expert. You will see two marketing images.\n"
        "The first is the ORIGINAL. The second is a MODIFIED version.\n\n"
        "Describe ONLY the visible differences between the two images.\n"
        "Be specific: what changed in terms of colors, layout, text, elements, backgrounds?\n"
        "If you cannot see any meaningful difference, say 'NO VISIBLE DIFFERENCE'.\n"
        "Do NOT guess or assume -- only describe what you can actually see."
    )

    blind_result = await vision_chat(
        system_prompt=blind_prompt,
        user_text="What are the visible differences between these two images?",
        image_b64=state["original_image_b64"],
        second_image_b64=state["edited_image_b64"],
    )

    logger.info("critic_blind_comparison", attempt=attempt, result=blind_result[:200])

    # If the blind comparison finds no difference, reject immediately
    no_diff_phrases = ["no visible difference", "no meaningful difference", "identical", "no difference", "same image", "no changes"]
    if any(phrase in blind_result.lower() for phrase in no_diff_phrases):
        trail = list(state.get("audit_trail", []))
        trail.append(AuditEntry(
            agent="critic",
            action="blind_comparison_rejected",
            detail=f"Attempt {attempt}. Blind comparison found no visible difference: {blind_result[:300]}",
        ))
        return {
            **state,
            "evaluation_passed": False,
            "evaluation_score": 0.0,
            "evaluation_feedback": f"Blind comparison found no visible difference: {blind_result[:200]}",
            "audit_trail": trail,
        }

    # Step 2: Full evaluation -- now tell it what the edit was supposed to be
    user_text = (
        f"**Recommendation:** {state['recommendation_title']}\n"
        f"{state['recommendation_description']}\n\n"
        f"**Brand guidelines:**\n{state['brand_guidelines_text']}\n\n"
        f"**Blind comparison found these differences:** {blind_result[:500]}\n\n"
        "The first image is the ORIGINAL. The second image is the EDITED variant.\n"
        "Given the differences identified above, evaluate whether the recommendation "
        "was effectively applied and brand guidelines were respected."
    )

    raw = await vision_chat(
        system_prompt=SYSTEM_PROMPT,
        user_text=user_text,
        image_b64=state["original_image_b64"],
        second_image_b64=state["edited_image_b64"],
    )

    cleaned = _strip_code_fences(raw)

    try:
        parsed = json.loads(cleaned)
        passed = bool(parsed.get("passed", False))
        score = float(parsed.get("score", 0.0))
        visually_different = parsed.get("visually_different", True)
        feedback: str = parsed.get("feedback", "")

        # Hard override: if the critic says the images aren't visually different,
        # force a failure regardless of the score
        if not visually_different:
            passed = False
            score = 0.0
            if not feedback:
                feedback = "No visible difference between original and edited image."
            logger.info("critic_no_visible_difference", attempt=attempt)

        metrics.variant_score.observe(score)
    except (json.JSONDecodeError, KeyError, ValueError):
        # Conservative default: treat unparseable responses as failures
        metrics.llm_parse_failures.labels(agent="critic").inc()
        logger.warning("critic_parse_failure", raw_prefix=raw[:200])
        passed = False
        score = 0.0
        feedback = raw

    trail = list(state.get("audit_trail", []))
    trail.append(
        AuditEntry(
            agent="critic",
            action="evaluation_completed",
            detail=(
                f"Attempt {attempt}. Passed={passed}, Score={score:.2f}. "
                f"Feedback: {feedback[:300]}"
            ),
        )
    )

    return {
        **state,
        "evaluation_passed": passed,
        "evaluation_score": score,
        "evaluation_feedback": feedback,
        "audit_trail": trail,
    }
