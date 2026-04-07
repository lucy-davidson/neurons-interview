"""Unit tests for individual agents and workflow routing logic.

Each agent is tested by mocking only the LLM call (vision_chat or
edit_image) and verifying input processing, output parsing, state
updates, and graceful fallback behaviour.
"""

from __future__ import annotations

import base64
import io
import json
from unittest.mock import AsyncMock, patch

import pytest
from PIL import Image

from app.workflow.agents.ideator import run_ideator, run_ideator_replacements
from app.workflow.agents.idea_critic import run_idea_critic
from app.workflow.agents.critic import run_critic, _compute_ssim
from app.workflow.agents.editor import run_editor
from app.workflow.agents.refiner import run_refiner
from app.workflow.graph import should_retry


# ── Helpers ───────────────────────────────────────────────────────────


def _make_state(**overrides) -> dict:
    """Build a minimal valid RecommendationState dict."""
    state = {
        "original_image_b64": "dGVzdA==",  # base64("test")
        "recommendation_id": "rec_1",
        "recommendation_title": "Strengthen Headline",
        "recommendation_description": "Increase contrast on the headline text.",
        "recommendation_type": "contrast_salience",
        "brand_guidelines_text": '{"protected_regions": ["logo"]}',
        "plan": "",
        "edit_prompt": "Make the headline bolder with more contrast.",
        "plan_approved": True,
        "plan_feedback": "",
        "plan_revision_count": 0,
        "edited_image_b64": "",
        "evaluation_passed": False,
        "evaluation_score": 0.0,
        "evaluation_feedback": "",
        "refiner_feedback": "",
        "attempt": 1,
        "max_attempts": 3,
        "audit_trail": [],
        "status": "",
    }
    state.update(overrides)
    return state


def _make_variant(title="Test Variant", desc="Test description"):
    return {
        "title": title,
        "description": desc,
        "edit_prompt": f"Apply: {desc}",
    }


# ═════════════════════════════════════════════════════════════════════
# Ideator
# ═════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
@patch("app.workflow.agents.ideator.vision_chat", new_callable=AsyncMock)
async def test_ideator_parses_five_variants(mock_chat):
    """Ideator should return 5 variant dicts with edit_prompt."""
    variants = [
        {"title": f"Variant {i}", "description": f"Desc {i}", "edit_prompt": f"Edit {i}"}
        for i in range(5)
    ]
    mock_chat.return_value = json.dumps({"variants": variants})

    result = await run_ideator(
        recommendation_title="Test", recommendation_description="Test desc",
        recommendation_type="contrast", brand_guidelines_text="{}",
        image_b64="dGVzdA==",
    )
    assert len(result) == 5
    assert result[0]["title"] == "Variant 0"
    assert "edit_prompt" in result[0]


@pytest.mark.asyncio
@patch("app.workflow.agents.ideator.vision_chat", new_callable=AsyncMock)
async def test_ideator_strips_code_fences(mock_chat):
    """Ideator should handle LLM responses wrapped in markdown code fences."""
    variants = [
        {"title": f"V{i}", "description": f"D{i}", "edit_prompt": f"E{i}"}
        for i in range(5)
    ]
    mock_chat.return_value = f"```json\n{json.dumps({'variants': variants})}\n```"

    result = await run_ideator(
        recommendation_title="T", recommendation_description="D",
        recommendation_type="t", brand_guidelines_text="{}", image_b64="dGVzdA==",
    )
    assert len(result) == 5


@pytest.mark.asyncio
@patch("app.workflow.agents.ideator.vision_chat", new_callable=AsyncMock)
async def test_ideator_fallback_on_bad_json(mock_chat):
    """Ideator should fall back to a single variant when JSON is unparseable."""
    mock_chat.return_value = "I cannot generate variants for this image."

    result = await run_ideator(
        recommendation_title="Original Title",
        recommendation_description="Original Desc",
        recommendation_type="t", brand_guidelines_text="{}", image_b64="dGVzdA==",
    )
    assert len(result) == 1
    assert result[0]["title"] == "Original Title"
    assert "edit_prompt" in result[0]


@pytest.mark.asyncio
@patch("app.workflow.agents.ideator.vision_chat", new_callable=AsyncMock)
async def test_ideator_adds_edit_prompt_fallback(mock_chat):
    """Ideator should add edit_prompt from description if LLM omits it."""
    variants = [{"title": "V1", "description": "Make it bold"}]
    mock_chat.return_value = json.dumps({"variants": variants})

    result = await run_ideator(
        recommendation_title="T", recommendation_description="D",
        recommendation_type="t", brand_guidelines_text="{}", image_b64="dGVzdA==",
    )
    assert result[0]["edit_prompt"] == "Make it bold"


@pytest.mark.asyncio
@patch("app.workflow.agents.ideator.vision_chat", new_callable=AsyncMock)
async def test_ideator_caps_at_five(mock_chat):
    """Ideator should return at most 5 variants even if LLM returns more."""
    variants = [
        {"title": f"V{i}", "description": f"D{i}", "edit_prompt": f"E{i}"}
        for i in range(10)
    ]
    mock_chat.return_value = json.dumps({"variants": variants})

    result = await run_ideator(
        recommendation_title="T", recommendation_description="D",
        recommendation_type="t", brand_guidelines_text="{}", image_b64="dGVzdA==",
    )
    assert len(result) == 5


@pytest.mark.asyncio
@patch("app.workflow.agents.ideator.vision_chat", new_callable=AsyncMock)
async def test_ideator_replacements(mock_chat):
    """Ideator replacements should generate the requested number of variants."""
    replacements = [
        {"title": "New V1", "description": "Better idea", "edit_prompt": "Do better"}
    ]
    mock_chat.return_value = json.dumps({"variants": replacements})

    result = await run_ideator_replacements(
        recommendation_title="T", recommendation_description="D",
        recommendation_type="t", brand_guidelines_text="{}",
        image_b64="dGVzdA==",
        approved=[_make_variant("Good one")],
        rejected=[_make_variant("Bad one")],
        rejection_feedback="Too vague",
    )
    assert len(result) == 1
    assert result[0]["title"] == "New V1"


# ═════════════════════════════════════════════════════════════════════
# Idea Critic
# ═════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
@patch("app.workflow.agents.idea_critic.vision_chat", new_callable=AsyncMock)
async def test_idea_critic_approves_all(mock_chat):
    """Idea critic should approve all variants when all are good."""
    reviews = [
        {"title": "V1", "approved": True},
        {"title": "V2", "approved": True},
    ]
    mock_chat.return_value = json.dumps({"reviews": reviews, "overall_feedback": ""})

    variants = [_make_variant("V1"), _make_variant("V2")]
    approved, rejected, feedback = await run_idea_critic(
        recommendation_title="T", recommendation_description="D",
        recommendation_type="t", brand_guidelines_text="{}",
        image_b64="dGVzdA==", variants=variants,
    )
    assert len(approved) == 2
    assert len(rejected) == 0


@pytest.mark.asyncio
@patch("app.workflow.agents.idea_critic.vision_chat", new_callable=AsyncMock)
async def test_idea_critic_rejects_some(mock_chat):
    """Idea critic should split variants into approved and rejected."""
    reviews = [
        {"title": "V1", "approved": True},
        {"title": "V2", "approved": False, "reason": "Too similar to V1"},
        {"title": "V3", "approved": True},
    ]
    mock_chat.return_value = json.dumps({
        "reviews": reviews,
        "overall_feedback": "Need more diversity",
    })

    variants = [_make_variant("V1"), _make_variant("V2"), _make_variant("V3")]
    approved, rejected, feedback = await run_idea_critic(
        recommendation_title="T", recommendation_description="D",
        recommendation_type="t", brand_guidelines_text="{}",
        image_b64="dGVzdA==", variants=variants,
    )
    assert len(approved) == 2
    assert len(rejected) == 1
    assert rejected[0]["title"] == "V2"
    assert "Too similar" in rejected[0]["rejection_reason"]
    assert "diversity" in feedback


@pytest.mark.asyncio
@patch("app.workflow.agents.idea_critic.vision_chat", new_callable=AsyncMock)
async def test_idea_critic_approves_on_parse_failure(mock_chat):
    """Idea critic should approve all on unparseable response."""
    mock_chat.return_value = "These all look fine to me."

    variants = [_make_variant("V1"), _make_variant("V2")]
    approved, rejected, feedback = await run_idea_critic(
        recommendation_title="T", recommendation_description="D",
        recommendation_type="t", brand_guidelines_text="{}",
        image_b64="dGVzdA==", variants=variants,
    )
    assert len(approved) == 2
    assert len(rejected) == 0


# ═════════════════════════════════════════════════════════════════════
# Editor
# ═════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
@patch("app.workflow.agents.editor.edit_image", new_callable=AsyncMock)
async def test_editor_stores_image(mock_edit):
    """Editor should store the generated image in state."""
    mock_edit.return_value = "base64encodedimage"

    state = await run_editor(_make_state(edit_prompt="Make headline bolder"))
    assert state["edited_image_b64"] == "base64encodedimage"
    assert len(state["audit_trail"]) == 1
    assert state["audit_trail"][0].agent == "editor"


# ═════════════════════════════════════════════════════════════════════
# Critic
# ═════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
@patch("app.workflow.agents.critic.vision_chat", new_callable=AsyncMock)
async def test_critic_passes_high_score(mock_chat):
    """Critic should pass a variant with high compliance score."""
    mock_chat.return_value = json.dumps({
        "passed": True, "score": 0.95,
        "recommendation_compliance": "Excellent",
        "brand_compliance": "All guidelines respected",
        "feedback": "",
    })

    state = await run_critic(_make_state(edited_image_b64="abc"))
    assert state["evaluation_passed"] is True
    assert state["evaluation_score"] == 0.95


@pytest.mark.asyncio
@patch("app.workflow.agents.critic.vision_chat", new_callable=AsyncMock)
async def test_critic_fails_low_score(mock_chat):
    """Critic should fail a variant with low compliance score."""
    mock_chat.return_value = json.dumps({
        "passed": False, "score": 0.3,
        "feedback": "The headline change is barely visible.",
    })

    state = await run_critic(_make_state(edited_image_b64="abc"))
    assert state["evaluation_passed"] is False
    assert state["evaluation_score"] == 0.3
    assert "barely visible" in state["evaluation_feedback"]


@pytest.mark.asyncio
@patch("app.workflow.agents.critic.vision_chat", new_callable=AsyncMock)
async def test_critic_fails_on_unparseable(mock_chat):
    """Critic should treat unparseable responses as failures."""
    mock_chat.return_value = "I cannot evaluate this image."

    state = await run_critic(_make_state(edited_image_b64="abc"))
    assert state["evaluation_passed"] is False
    assert state["evaluation_score"] == 0.0


@pytest.mark.asyncio
@patch("app.workflow.agents.critic.vision_chat", new_callable=AsyncMock)
async def test_critic_handles_code_fenced_json(mock_chat):
    """Critic should parse JSON wrapped in markdown code fences."""
    mock_chat.return_value = f"```json\n{json.dumps({'passed': True, 'score': 0.88, 'feedback': ''})}\n```"

    state = await run_critic(_make_state(edited_image_b64="abc"))
    assert state["evaluation_passed"] is True
    assert state["evaluation_score"] == 0.88


# ═════════════════════════════════════════════════════════════════════
# Refiner
# ═════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
@patch("app.workflow.agents.refiner.vision_chat", new_callable=AsyncMock)
async def test_refiner_produces_revised_prompt(mock_chat):
    """Refiner should produce a revised edit_prompt and increment the attempt counter."""
    mock_chat.return_value = "Make the headline text much darker and add a contrasting backdrop."

    state = await run_refiner(_make_state(
        evaluation_feedback="The change was too subtle.",
        attempt=1,
    ))
    assert "darker" in state["edit_prompt"]
    assert state["attempt"] == 2
    assert len(state["audit_trail"]) == 1
    assert state["audit_trail"][0].agent == "refiner"


# ═════════════════════════════════════════════════════════════════════
# Routing Logic
# ═════════════════════════════════════════════════════════════════════


def test_should_retry_when_failed_and_under_limit():
    state = {"evaluation_passed": False, "attempt": 1, "max_attempts": 3}
    assert should_retry(state) == "refiner"


def test_should_retry_when_failed_at_second_attempt():
    state = {"evaluation_passed": False, "attempt": 2, "max_attempts": 3}
    assert should_retry(state) == "refiner"


def test_should_not_retry_when_passed():
    state = {"evaluation_passed": True, "attempt": 1, "max_attempts": 3}
    assert should_retry(state) == "end"


def test_should_not_retry_at_max_attempts():
    state = {"evaluation_passed": False, "attempt": 3, "max_attempts": 3}
    assert should_retry(state) == "end"


def test_should_not_retry_when_passed_at_max():
    state = {"evaluation_passed": True, "attempt": 3, "max_attempts": 3}
    assert should_retry(state) == "end"


def test_should_retry_defaults_to_refiner():
    """With no evaluation_passed set, should default to retry."""
    state = {"attempt": 1, "max_attempts": 3}
    assert should_retry(state) == "refiner"


# ═════════════════════════════════════════════════════════════════════
# SSIM Check
# ═════════════════════════════════════════════════════════════════════


def _img_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def test_ssim_identical_images():
    """SSIM of identical images should be ~1.0."""
    img = Image.new("RGB", (100, 100), color="red")
    b64 = _img_to_b64(img)
    score = _compute_ssim(b64, b64)
    assert score > 0.99


def test_ssim_very_different_images():
    """SSIM of completely different images should be low."""
    img1 = Image.new("RGB", (100, 100), color="red")
    img2 = Image.new("RGB", (100, 100), color="blue")
    score = _compute_ssim(_img_to_b64(img1), _img_to_b64(img2))
    assert score < 0.5


def test_ssim_slightly_different():
    """SSIM of slightly modified image should be high but not 1.0."""
    import numpy as np
    arr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    img1 = Image.fromarray(arr)
    # Add small noise
    arr2 = arr.copy()
    arr2[:50, :50] = 0  # black out a quarter of the image
    img2 = Image.fromarray(arr2)
    score = _compute_ssim(_img_to_b64(img1), _img_to_b64(img2))
    assert 0.5 < score < 0.99


@pytest.mark.asyncio
@patch("app.workflow.agents.critic.vision_chat", new_callable=AsyncMock)
async def test_critic_auto_rejects_identical_images(mock_chat):
    """Critic should auto-reject when SSIM indicates identical images."""
    # Create identical images
    img = Image.new("RGB", (100, 100), color="red")
    b64 = _img_to_b64(img)

    state = _make_state(
        original_image_b64=b64,
        edited_image_b64=b64,
    )
    result = await run_critic(state)

    assert result["evaluation_passed"] is False
    assert result["evaluation_score"] == 0.0
    assert "SSIM" in result["evaluation_feedback"] or "No visible difference" in result["evaluation_feedback"]
    # LLM should NOT have been called
    mock_chat.assert_not_called()


# ═════════════════════════════════════════════════════════════════════
# Blind Comparison
# ═════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
@patch("app.workflow.agents.critic.vision_chat", new_callable=AsyncMock)
async def test_critic_blind_rejects_no_difference(mock_chat):
    """Critic should reject when blind comparison finds no visible difference."""
    img1 = Image.new("RGB", (100, 100), color="red")
    img2 = Image.new("RGB", (100, 100), color="blue")
    b64_1 = _img_to_b64(img1)
    b64_2 = _img_to_b64(img2)

    # First call is blind comparison -- returns no difference
    # (even though images are different at pixel level, simulate LLM saying no diff)
    mock_chat.return_value = "NO VISIBLE DIFFERENCE. The images appear identical."

    state = _make_state(
        original_image_b64=b64_1,
        edited_image_b64=b64_2,
    )
    result = await run_critic(state)

    assert result["evaluation_passed"] is False
    assert result["evaluation_score"] == 0.0
    assert "no visible difference" in result["evaluation_feedback"].lower()
    # Only one LLM call (blind comparison) -- should NOT proceed to full evaluation
    assert mock_chat.call_count == 1


@pytest.mark.asyncio
@patch("app.workflow.agents.critic.vision_chat", new_callable=AsyncMock)
async def test_critic_blind_passes_then_evaluates(mock_chat):
    """Critic should proceed to full evaluation when blind comparison finds differences."""
    img1 = Image.new("RGB", (100, 100), color="red")
    img2 = Image.new("RGB", (100, 100), color="blue")
    b64_1 = _img_to_b64(img1)
    b64_2 = _img_to_b64(img2)

    # First call: blind comparison finds difference
    # Second call: full evaluation passes
    mock_chat.side_effect = [
        "The background color changed from red to blue.",
        json.dumps({"passed": True, "score": 0.9, "visually_different": True, "feedback": ""}),
    ]

    state = _make_state(
        original_image_b64=b64_1,
        edited_image_b64=b64_2,
    )
    result = await run_critic(state)

    assert result["evaluation_passed"] is True
    assert result["evaluation_score"] == 0.9
    # Two LLM calls: blind + full evaluation
    assert mock_chat.call_count == 2
