"""Prometheus metrics for the visual-recommendations backend.

Tracks job lifecycle, agent performance, LLM API usage, and error rates.
Scraped via the ``/metrics`` endpoint exposed by the FastAPI instrumentator.
"""

from prometheus_client import Counter, Histogram, Gauge

# ── Job lifecycle ─────────────────────────────────────────────────────

jobs_submitted = Counter(
    "visrec_jobs_submitted_total",
    "Total jobs submitted to the API",
)

jobs_completed = Counter(
    "visrec_jobs_completed_total",
    "Total jobs that finished processing",
    ["status"],  # "completed" or "failed"
)

jobs_in_progress = Gauge(
    "visrec_jobs_in_progress",
    "Number of jobs currently being processed",
)

# ── Variant outcomes ──────────────────────────────────────────────────

variants_processed = Counter(
    "visrec_variants_processed_total",
    "Total variants that completed the pipeline",
    ["status"],  # "accepted" or "max_retries_exceeded"
)

variant_score = Histogram(
    "visrec_variant_score",
    "Critic evaluation scores for completed variants",
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

variant_attempts = Histogram(
    "visrec_variant_attempts",
    "Number of retry attempts per variant",
    buckets=[1, 2, 3, 4, 5],
)

# ── LLM API calls ────────────────────────────────────────────────────

llm_call_duration = Histogram(
    "visrec_llm_call_duration_seconds",
    "Duration of LLM API calls",
    ["provider", "call_type"],  # e.g. ("openai", "vision_chat"), ("gemini", "edit_image")
    buckets=[0.5, 1, 2, 5, 10, 20, 30, 60, 120],
)

llm_calls_total = Counter(
    "visrec_llm_calls_total",
    "Total LLM API calls made",
    ["provider", "call_type"],
)

llm_errors_total = Counter(
    "visrec_llm_errors_total",
    "Total LLM API call failures",
    ["provider", "call_type", "error_type"],
)

# ── Agent-specific ────────────────────────────────────────────────────

agent_invocations = Counter(
    "visrec_agent_invocations_total",
    "Total invocations per agent",
    ["agent"],  # "ideator", "idea_critic", "editor", "critic", "refiner"
)

llm_parse_failures = Counter(
    "visrec_llm_parse_failures_total",
    "LLM responses that failed JSON parsing (graceful fallback triggered)",
    ["agent"],
)

# ── Dependency health ─────────────────────────────────────────────

dependency_up = Gauge(
    "visrec_dependency_up",
    "Whether a dependency is reachable (1=healthy, 0=down)",
    ["dependency"],  # "openai", "gemini", "claude", "postgresql"
)

dependency_latency = Gauge(
    "visrec_dependency_latency_ms",
    "Last measured latency to a dependency in milliseconds",
    ["dependency"],
)

current_rate_limit = Gauge(
    "visrec_current_rate_limit",
    "Current adaptive rate limit (calls per minute)",
)

provider_fallbacks = Counter(
    "visrec_provider_fallbacks_total",
    "Times the system fell back from primary to alternate LLM provider",
    ["primary", "fallback", "call_type"],  # e.g. ("openai", "gemini", "vision_chat")
)

# ── Agent-level timing ───────────────────────────────────────────────

agent_duration = Histogram(
    "visrec_agent_duration_seconds",
    "Duration of individual agent invocations",
    ["agent"],
    buckets=[0.5, 1, 2, 5, 10, 20, 30, 60, 120, 180],
)

# ── Token usage ──────────────────────────────────────────────────────

llm_tokens_total = Counter(
    "visrec_llm_tokens_total",
    "Total tokens consumed by LLM calls",
    ["provider", "call_type", "token_type"],  # token_type: "prompt" | "completion"
)

agent_tokens = Counter(
    "visrec_agent_tokens_total",
    "Total tokens consumed per agent invocation",
    ["agent", "token_type"],
)

# ── Variant-level timing ─────────────────────────────────────────────

variant_duration = Histogram(
    "visrec_variant_duration_seconds",
    "Wall clock time per variant through the edit-evaluate-refine loop",
    buckets=[5, 10, 30, 60, 120, 300, 600],
)

# ── Cost estimation ──────────────────────────────────────────────────

llm_estimated_cost = Counter(
    "visrec_llm_estimated_cost_usd",
    "Estimated cost in USD of LLM API usage",
    ["provider", "call_type"],
)
