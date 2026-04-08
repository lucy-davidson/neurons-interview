"""Per-model cost estimation using published pricing.

Costs are approximate and based on publicly available pricing as of
April 2026.  Update the tables when pricing changes.
"""

from __future__ import annotations

# Costs in USD per 1M tokens (input, output)
TOKEN_COST: dict[str, dict[str, tuple[float, float]]] = {
    "openai": {
        "gpt-4o": (2.50, 10.00),
    },
    "gemini": {
        "gemini-2.5-flash": (0.15, 0.60),
    },
    "claude": {
        "claude-sonnet-4-20250514": (3.00, 15.00),
    },
}

# Image generation costs (USD per image)
IMAGE_COST: dict[str, dict[str, float]] = {
    "openai": {"gpt-image-1": 0.04},
    "gemini": {"gemini-2.5-flash-image": 0.0315},
}


def estimate_token_cost(
    provider: str, model: str, prompt_tokens: int, completion_tokens: int,
) -> float:
    """Estimate cost in USD for a text/vision LLM call."""
    costs = TOKEN_COST.get(provider, {}).get(model)
    if not costs:
        return 0.0
    input_rate, output_rate = costs
    return (prompt_tokens * input_rate + completion_tokens * output_rate) / 1_000_000


def estimate_image_cost(provider: str, model: str) -> float:
    """Estimate cost in USD for a single image generation call."""
    return IMAGE_COST.get(provider, {}).get(model, 0.0)
