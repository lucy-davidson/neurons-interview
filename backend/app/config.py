"""Application settings loaded from environment variables and ``.env``.

All provider keys and model identifiers can be overridden via env vars
(e.g. ``TEXT_PROVIDER=gemini``, ``OPENAI_API_KEY=sk-...``).
"""

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
