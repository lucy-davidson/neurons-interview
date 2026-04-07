"""Application settings loaded from environment variables and ``.env``.

All provider keys and model identifiers can be overridden via env vars
(e.g. ``TEXT_PROVIDER=gemini``, ``OPENAI_API_KEY=sk-...``).

Per-job overrides are supported via :class:`RuntimeConfig`, which merges
global defaults with optional overrides submitted at job creation time.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Central configuration for LLM providers, retry policy, and logging."""

    # -- LLM provider selection (text and image can differ) ---------------
    text_provider: str = "openai"
    image_provider: str = "gemini"

    # -- OpenAI -----------------------------------------------------------
    openai_api_key: str = ""
    openai_vision_model: str = "gpt-4o"
    openai_image_model: str = "gpt-image-1"

    # -- Google Gemini ----------------------------------------------------
    google_api_key: str = ""
    gemini_vision_model: str = "gemini-2.5-flash"
    gemini_image_model: str = "gemini-2.5-flash-image"

    # -- Anthropic Claude -------------------------------------------------
    anthropic_api_key: str = ""
    claude_vision_model: str = "claude-sonnet-4-20250514"

    # -- Security ------------------------------------------------------------
    cors_origins: str = "*"  # Comma-separated, e.g. "https://app.neurons.com,http://localhost:7860"

    # -- Database ------------------------------------------------------------
    database_url: str = ""

    # -- Input validation limits -------------------------------------------
    max_image_size_bytes: int = 50 * 1024 * 1024  # 50 MB
    max_image_dimension: int = 8192  # pixels
    max_recommendations: int = 5
    max_recommendation_text_length: int = 2000  # characters

    # -- Rate limiting -------------------------------------------------------
    llm_rate_limit: float = 30.0  # starting max LLM API calls per minute (auto-adjusts down)
    llm_burst: int = 5  # max burst (concurrent calls before throttling)

    # -- Workflow parameters ----------------------------------------------
    num_variants: int = 2  # variants per recommendation (reduce for faster results)
    max_retries: int = 3
    image_size: str = "1024x1024"
    log_level: str = "INFO"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


settings = Settings()


# ── Per-job config overrides ──────────────────────────────────────────


VALID_TEXT_PROVIDERS = {"openai", "gemini", "claude"}
VALID_IMAGE_PROVIDERS = {"openai", "gemini"}  # Claude has no image gen API


class ConfigOverrides(BaseModel):
    """Optional per-job overrides submitted at job creation time.

    Any field left as ``None`` inherits the global default from *settings*.
    """

    text_provider: str | None = None
    image_provider: str | None = None
    text_model: str | None = None
    image_model: str | None = None
    num_variants: int | None = None
    max_retries: int | None = None

    def validate_values(self) -> list[str]:
        """Return a list of validation error messages (empty if valid)."""
        errors: list[str] = []
        if self.text_provider and self.text_provider not in VALID_TEXT_PROVIDERS:
            errors.append(
                f"Invalid text_provider '{self.text_provider}'. "
                f"Must be one of: {', '.join(sorted(VALID_TEXT_PROVIDERS))}"
            )
        if self.image_provider and self.image_provider not in VALID_IMAGE_PROVIDERS:
            errors.append(
                f"Invalid image_provider '{self.image_provider}'. "
                f"Must be one of: {', '.join(sorted(VALID_IMAGE_PROVIDERS))}"
            )
        if self.num_variants is not None and self.num_variants < 1:
            errors.append("num_variants must be at least 1.")
        if self.max_retries is not None and self.max_retries < 1:
            errors.append("max_retries must be at least 1.")
        return errors


class RuntimeConfig:
    """Effective configuration for a single job.

    Merges global :data:`settings` with optional per-job overrides.
    This object is passed through the workflow so agents never import
    ``settings`` directly.
    """

    def __init__(self, overrides: ConfigOverrides | None = None):
        ov = overrides or ConfigOverrides()
        self.text_provider: str = ov.text_provider or settings.text_provider
        self.image_provider: str = ov.image_provider or settings.image_provider
        self.num_variants: int = ov.num_variants or settings.num_variants
        self.max_retries: int = ov.max_retries or settings.max_retries
        self.image_size: str = settings.image_size

        # Resolve model names — override takes precedence, then provider default
        self.text_model: str = ov.text_model or self._default_text_model()
        self.image_model: str = ov.image_model or self._default_image_model()

    def _default_text_model(self) -> str:
        if self.text_provider == "gemini":
            return settings.gemini_vision_model
        if self.text_provider == "claude":
            return settings.claude_vision_model
        return settings.openai_vision_model

    def _default_image_model(self) -> str:
        if self.image_provider == "gemini":
            return settings.gemini_image_model
        return settings.openai_image_model

    def snapshot(self) -> dict[str, Any]:
        """Return a plain dict of the effective config for persistence."""
        return {
            "text_provider": self.text_provider,
            "image_provider": self.image_provider,
            "text_model": self.text_model,
            "image_model": self.image_model,
            "num_variants": self.num_variants,
            "max_retries": self.max_retries,
            "image_size": self.image_size,
        }
