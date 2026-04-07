# Codebase Walkthrough

This document walks through every file in the backend, explaining what it does, why it exists, and how it connects to the rest of the system.

---

## 1. Entry Point: `app/main.py`

This is the FastAPI application. It's where HTTP requests arrive and where the async job lifecycle is managed.

**What happens when a request comes in:**

1. `POST /api/v1/jobs` receives a multipart form upload with two parts:
   - `image` — the marketing creative (PNG, JPEG, or WebP)
   - `request_body` — a JSON file containing the recommendations and brand guidelines

2. The endpoint validates the image through several checks:
   - Content type (must be PNG, JPEG, or WebP)
   - File size (max 50 MB)
   - Image decodability (via Pillow)
   - Dimensions (max 8192px per side)
   - RGBA detection (warns about alpha channel inconsistencies across providers)
   - Recommendation count (max 5) and text length (max 2000 chars per description)

3. The job is stored in an in-memory dictionary (`_jobs`) and persisted to PostgreSQL (if configured). An `asyncio.create_task` fires off `_process_job()` in the background. The endpoint immediately returns `202 Accepted` with the job ID.

4. `_process_job()` starts a heartbeat loop (updates every 30s), binds a structlog context with the job ID, then calls `run_all_recommendations()` from the workflow module. A callback (`on_variant_complete`) persists results to the database after each variant finishes, so the polling endpoint shows partial progress.

5. `GET /api/v1/jobs/{job_id}` lets the client poll for status. It checks in-memory first (active jobs), then falls back to the database (jobs from previous sessions). Results stream in incrementally as variants complete.

**Startup lifecycle:**

On boot, the `lifespan` context manager:
1. Initialises the database connection pool and schema
2. Recovers orphaned jobs from previous sessions (marks them as failed)
3. Starts a background cleanup loop that sweeps for stale heartbeats every 60s
4. Starts a dependency health check loop that pings all 3 LLM providers + PostgreSQL every 30s

**Other endpoints:**

- `POST /api/v1/jobs/{job_id}/cancel` — marks a running job as failed with "Cancelled by user"
- `POST /api/v1/jobs/{job_id}/feedback` — records user feedback (thumbs up/down, selected, refinement) with full variant context (provider, model, score)
- `GET /api/v1/feedback/stats` — aggregate feedback counts across all jobs
- `GET /api/v1/experiments/comparison` — groups all variant outcomes by text and image provider
- `GET /api/v1/health/dependencies` — live health check for all providers and the database
- `GET /api/v1/thumb/{job_id}/{variant_id}` — returns a 300px JPEG thumbnail
- `GET /api/v1/jobs/{job_id}/image/{variant_id}` — returns the full base64 image
- `GET /api/v1/metrics/summary` — JSON summary of Prometheus counters for the frontend
- `GET /metrics` — Prometheus scrape endpoint (auto-instrumented by `prometheus-fastapi-instrumentator`)
- `GET /health` — liveness probe with database status
- `GET /` — serves the static HTML frontend

**Why this pattern:** The agentic workflow involves multiple LLM calls and image generation steps that take 30-120+ seconds per recommendation. An async job queue pattern (submit → poll) prevents HTTP timeouts and lets the frontend show incremental progress.

---

## 2. Configuration: `app/config.py`

A single `Settings` class using `pydantic-settings`. All configuration is driven by environment variables (or a `.env` file). Key groups:

- **Provider selection** — `text_provider` and `image_provider` are independently configurable (`openai`, `gemini`, or `claude`)
- **API keys** — `openai_api_key`, `google_api_key`, `anthropic_api_key` (only the active provider's key is required)
- **Model names** — separately configurable for each provider's vision and image models
- **Input validation limits** — max file size, dimensions, recommendation count, text length
- **Rate limiting** — starting RPM and burst size (auto-adjusts from API response headers)
- **Workflow parameters** — `num_variants` (target accepted variants per recommendation), `max_retries`, `image_size`

`model_config = {"extra": "ignore"}` allows old or unused env vars to exist without causing validation errors.

**Why a separate config file:** Centralises all tuneable parameters so they can be changed via environment variables without touching code. The Docker Compose file passes these through from the `.env` file.

---

## 3. Data Models: `app/models.py`

All Pydantic models live here. This file defines the contract between every layer of the system.

**Request models:**
- `Recommendation` — a single recommendation with `id`, `title`, `description`, and `type`
- `BrandGuidelines` — protected regions, typography rules, aspect ratio, brand elements
- `JobRequest` — the JSON body sent by the client (list of recommendations + guidelines)

**Response models:**
- `JobResponse` — returned by the API: job ID, status, results, error, and warnings (e.g. RGBA)
- `RecommendationResult` — per-recommendation container holding a list of `VariantResult`s
- `VariantResult` — per-variant outcome: status (`accepted` or `max_retries_exceeded`), attempt count, edited image, evaluation score/feedback, audit trail, and provider tracking (text_provider, image_provider, text_model, image_model)
- `AuditEntry` — a single timestamped log entry from one of the agents
- `JobSummary` / `JobListResponse` — lightweight projections for the job-list endpoint

**Internal models:**
- `Job` — the full server-side record including the original image. The `to_response()` method strips the original image (which is large) when returning data to the client.
- `JobStatus` — enum: `pending`, `running`, `completed`, `failed`

**Why one models file:** The models are the shared language between the API layer, the workflow layer, and the agent layer. Keeping them in one place makes the data contracts easy to review and prevents circular imports.

---

## 4. Database Persistence: `app/db.py`

PostgreSQL persistence using asyncpg for non-blocking access. Two tables:

**`jobs` table:**
- `job_id` (PK), `status`, `created_at`, `original_image_b64`
- `recommendations` and `brand_guidelines` as JSONB
- `results` as JSONB — overwritten incrementally as variants complete
- `last_heartbeat` — TIMESTAMPTZ, updated every 30s by the background heartbeat loop

**`feedback` table:**
- Links to a job/variant with `feedback_type`, `comment`, timestamps
- Stores the provider context: `text_provider`, `image_provider`, `text_model`, `image_model`
- Stores the critic's `evaluation_score` and `recommendation_type` for later analysis

**Key operations:**
- `touch_heartbeat()` / `fail_stale_jobs()` — the heartbeat system. If a job's heartbeat goes stale (5 min timeout), the cleanup sweep marks it as failed.
- `create_job()` / `update_job_status()` / `update_job_results()` — dual-write with in-memory store
- `save_feedback()` / `get_feedback_stats()` — feedback persistence and aggregation

All operations are wrapped in try/except so database failures never crash the application — they log the error and degrade gracefully to in-memory mode.

---

## 5. Services Layer

### `app/services/image.py`

Image utility functions:
- `validate_image(data)` — uses Pillow to verify that uploaded bytes are actually a valid image
- `b64_to_bytes()` / `bytes_to_b64()` — conversions between base64 strings and raw bytes

### `app/services/llm.py`

The LLM abstraction layer. Three providers are supported (OpenAI, Gemini, Claude), each with their own SDK client.

**Two public functions:**

**`vision_chat(system_prompt, user_text, image_b64, second_image_b64)`**
- Sends a chat completion request with vision capabilities
- Supports up to two images (the critic needs to compare original vs edited)
- Used by the ideator, idea critic, critic, and refiner agents
- Rate-limited per-provider, with automatic fallback to alternate providers on failure

**`edit_image(prompt, image_b64)`**
- Sends an image edit request to the configured image provider
- Used by the editor agent
- Falls back to alternate providers on failure (skips Claude — no native image generation API)
- Gemini implementation retries up to 3 times on empty responses with exponential backoff

**Adaptive rate limiting:**

Each provider gets its own `_AdaptiveRateLimiter` instance. The limiter:
- Starts at a configurable default RPM (30 by default)
- Auto-configures from OpenAI's `x-ratelimit-limit-requests` response header (uses 80% of the actual limit)
- Halves the rate on errors (`backoff()`)
- Gradually recovers after 5 consecutive successes (`on_success()`)
- Uses `aiolimiter.AsyncLimiter` for the actual token-bucket implementation

**`_is_rate_limit_error()`** distinguishes transient rate limits (429, "too many requests", "resource_exhausted") from permanent failures ("insufficient_quota"). Only transient errors trigger backoff.

**Provider fallback:** If the primary provider fails, the system tries alternate providers in order. Only providers with API keys configured are included in the fallback chain. The fallback is per-call, not global — a single failure doesn't switch the default provider.

**Why wrap the API calls:** Isolates provider-specific logic so agents don't need to know about API details. Adding a new provider means implementing two functions (vision chat + image edit) and adding it to the dispatcher — nothing else changes.

---

## 6. Workflow Layer

This is the heart of the system — the LangGraph-based agentic workflow.

### `app/workflow/state.py`

Defines `RecommendationState` as a `TypedDict`. This is the state object that flows through the LangGraph graph. Every agent reads from and writes to this state.

Key fields:
- **Inputs** (set once): `original_image_b64`, recommendation details, `brand_guidelines_text`
- **Mutable** (updated by agents): `edit_prompt`, `edited_image_b64`, evaluation results, `refiner_feedback`, `attempt` counter
- **Accumulator**: `audit_trail` — every agent appends entries here, building the full decision log

### `app/workflow/graph.py`

This file constructs, compiles, and runs the LangGraph StateGraph. It handles two levels of orchestration:

**Inner graph (per variant):**
```
editor → critic → should_retry?
                    ├── "refiner" → refiner → editor (loops back)
                    └── "end"     → END
```

**Outer orchestration (per recommendation):**
```
ideator (generates 10 variant ideas with edit prompts)
    └─► idea critic (reviews all 10, rejects weak/duplicate ones)
          └─► pool of approved ideas
                └─► try in batches of 2:
                      editor → critic → pass?
                                          │ no → refiner → editor (retry up to 3x)
                      keep drawing from pool until 2 accepted or pool exhausted
```

**`build_recommendation_graph()`** creates the per-variant graph:
1. Three nodes: `editor`, `critic`, `refiner`
2. Entry point is `editor` (the edit prompt is already in state from the ideator)
3. After `critic`, a conditional edge (`should_retry()`) checks if the evaluation passed or max attempts reached

The graph is compiled once at module import time for efficiency.

**`run_recommendation_workflow()`** orchestrates a single recommendation:
1. Calls `run_ideator()` to generate a pool of 10 variant ideas
2. Calls `run_idea_critic()` to filter the pool (removes subtle/duplicate/infeasible ideas)
3. Draws from the pool in batches of 2. Each batch runs concurrently via `asyncio.gather`
4. After each batch, checks if we have enough accepted variants (target: 2)
5. Stops when the target is met or the pool is exhausted

**`run_all_recommendations()`** processes all recommendations concurrently:
- Pre-creates empty `RecommendationResult` objects so they appear in the API response immediately
- Runs one `run_recommendation_workflow` coroutine per recommendation via `asyncio.gather`
- Results stream in incrementally through the shared `RecommendationResult` objects

**Why pool-and-batch:** Generating 10 ideas upfront and trying them in batches avoids the latency cost of generating replacement ideas mid-flight. If a batch fails, the next ideas are already waiting in the pool — no extra ideation LLM calls needed.

---

## 7. The Five Agents

Each agent is a single async function. The ideator and idea critic operate on lists of variant ideas. The editor, critic, and refiner are LangGraph nodes that take `RecommendationState`, do their work, and return an updated copy.

### `app/workflow/agents/ideator.py`

**Role:** Turn a vague recommendation into 10 concrete variant ideas, each with a title, description, and edit prompt ready for the image editor.

**How it works:**
1. Sends the recommendation, brand guidelines, and the original image to the vision LLM
2. The system prompt instructs the model to return JSON with a `variants` array
3. Each variant includes an `edit_prompt` — a self-contained instruction for the image editor
4. Retries up to 3 times on parse failure, with a "keep descriptions shorter" hint on retry
5. Falls back to a single pass-through variant if all retries fail (so the pipeline always has something to run)

**`run_ideator_replacements()`** generates replacements for rejected ideas, given the approved and rejected lists plus the reviewer's feedback. Used during the idea critique loop (though the current pool-and-batch approach means this is rarely needed).

### `app/workflow/agents/idea_critic.py`

**Role:** Review all variant ideas together and reject weak ones before they reach the expensive image generation step.

**How it works:**
1. Receives all variant ideas in a single call (not individually — this is important for catching duplicates)
2. Evaluates each on: visual impact, recommendation alignment, brand compliance, distinctiveness, feasibility
3. Returns `(approved, rejected, feedback)` tuple
4. On parse failure: approves all ideas (permissive — better to try than to block)

**Key design detail:** The prompt specifically calls out subtle changes ("slightly increase letter spacing", "marginally bolder font") as automatic rejections. This was added because the biggest real-world problem was the image editor returning unchanged images — catching weak ideas early saves expensive generation attempts.

### `app/workflow/agents/editor.py`

**Role:** Generate the edited image.

**How it works:**
1. Reads the `edit_prompt` from state (set by the ideator, or revised by the refiner on retries)
2. Calls `edit_image()` with the original image and the edit prompt
3. Stores the resulting base64 image in `edited_image_b64`
4. Appends to the audit trail

This is the simplest agent — a bridge between the ideator's intent and the image generation model.

### `app/workflow/agents/critic.py`

**Role:** Evaluate the edited image against the original recommendation and brand guidelines. This is the quality gate.

**Three-layer evaluation:**

1. **SSIM pre-check (deterministic):** Computes structural similarity between original and edited images. If SSIM > 0.95, the images are considered identical — auto-reject without an LLM call. Images are downscaled to 512px for speed.

2. **Blind visual comparison (unbiased LLM):** Asks "what's different between these two images?" without revealing what the edit was supposed to be. This prevents anchoring bias — the model can't convince itself it sees a change that isn't there just because it knows what to look for. If the blind comparison reports no visible difference, auto-reject.

3. **Full LLM evaluation (informed LLM):** With the blind comparison's findings in context, evaluates recommendation compliance and brand guideline compliance. Returns `passed`, `score` (0.0-1.0), `visually_different`, and actionable `feedback`. If `visually_different` is false, the result is overridden to a failure regardless of the score.

**Why three layers:** The single biggest problem in practice was the image editor returning the original image unchanged, and the critic LLM accepting it because it "saw" the expected changes (anchoring bias). SSIM catches pixel-identical images deterministically. The blind comparison catches near-identical images that fool pixel metrics. The informed evaluation handles everything else.

### `app/workflow/agents/refiner.py`

**Role:** Translate critic feedback into a revised edit prompt for the next attempt.

**How it works:**
1. Takes the original recommendation, the previous edit prompt, the critic's feedback, and brand guidelines
2. Produces a revised `edit_prompt` (1-3 sentences) that addresses what went wrong
3. Increments the attempt counter
4. The graph loops back to the editor with the new prompt

**Why a separate refiner:** The critic's output is evaluative ("this failed because X"). The refiner translates that into prescriptive guidance ("make the headline 3x larger in bold red"). This separation keeps each agent focused and produces better revision prompts than asking the editor to interpret raw evaluation feedback.

---

## 8. Observability: `app/metrics.py`

Prometheus metrics for everything, scraped via the `/metrics` endpoint:

- **Job lifecycle:** `jobs_submitted_total`, `jobs_completed_total` (by status), `jobs_in_progress`
- **Variant outcomes:** `variants_processed_total` (by status), `variant_score` (histogram), `variant_attempts` (histogram)
- **LLM API calls:** `llm_call_duration_seconds` (by provider/type), `llm_calls_total`, `llm_errors_total` (by provider/type/error)
- **Agent-specific:** `agent_invocations_total` (by agent), `llm_parse_failures_total` (by agent)
- **Dependency health:** `dependency_up` (gauge by dependency), `dependency_latency_ms`
- **Rate limiting:** `current_rate_limit` (gauge), `provider_fallbacks_total` (by primary/fallback/type)

These metrics feed a Grafana dashboard with 14 panels, auto-provisioned via the `monitoring/grafana/` config.

---

## 9. Infrastructure

### `Dockerfile`

A Python 3.12 slim image:
1. Copies and installs `requirements.txt` first (layer caching)
2. Creates a non-root `appuser` for security
3. Copies the application code and sets ownership
4. Exposes port 8000
5. Runs uvicorn as the non-root user

### `docker-compose.yml` (project root)

Five services:
- **db** — PostgreSQL 16 with a named volume for data persistence
- **backend** — builds from `backend/Dockerfile`, mounts `app/` and `static/` for hot-reload, depends on `db`
- **pgadmin** — database browser, auto-configured to connect to `db`
- **prometheus** — scrapes the backend's `/metrics` endpoint every 5s
- **grafana** — auto-provisions the datasource and dashboard on first boot

### `requirements.txt`

Key dependencies:
- `fastapi` + `uvicorn` — web framework and ASGI server
- `python-multipart` — required for file upload parsing
- `pydantic` + `pydantic-settings` — data validation and configuration
- `openai` — OpenAI Python SDK (async support)
- `google-genai` — Google GenAI SDK
- `anthropic` — Anthropic Python SDK (async support)
- `langgraph` — the agentic workflow framework
- `Pillow` — image validation and thumbnail generation
- `scikit-image` + `numpy` — SSIM computation
- `asyncpg` — non-blocking PostgreSQL driver
- `structlog` — structured JSON logging
- `prometheus-client` + `prometheus-fastapi-instrumentator` — metrics
- `aiolimiter` — async rate limiting
- `httpx` — async HTTP client (fallback for URL-based image responses)

### `tests/`

76 tests across 3 files, all run in ~3 seconds with no API keys:

- `test_agents.py` (27) — agent logic, SSIM computation, blind comparison routing, parse failures, fallbacks, conditional routing edge cases
- `test_api.py` (33) — input validation (file size, dimensions, content type, recommendation count, text length), job lifecycle, cancel, feedback, thumbnails, DB persistence, CORS, metrics
- `test_rate_limiter.py` (16) — error detection (`_is_rate_limit_error` distinguishing transient vs permanent), adaptive backoff, recovery after consecutive successes, per-provider isolation

All tests use mocked LLM calls via `unittest.mock.patch`.

---

## 10. Data Flow Summary

Here's the complete journey of a request through the system:

```
Client                            Backend
  │                                 │
  ├─ POST /api/v1/jobs ────────────►│
  │   (image + JSON)                │
  │                                 ├─ Validate (type, size, dims, recs)
  │                                 ├─ Persist to memory + PostgreSQL
  │                                 ├─ Spawn background task + heartbeat
  │◄── 202 {job_id, status} ────────┤
  │                                 │
  │                                 │  Background: run_all_recommendations()
  │                                 │  ┌─ asyncio.gather (1 per recommendation) ─┐
  │                                 │  │                                          │
  │                                 │  │  Ideator (vision LLM)                    │
  │                                 │  │  → 10 variant ideas with edit prompts    │
  │                                 │  │                                          │
  │                                 │  │  Idea Critic (vision LLM)                │
  │                                 │  │  → filter to approved pool               │
  │                                 │  │                                          │
  │                                 │  │  Pool-and-batch loop:                    │
  │                                 │  │  ┌─ Take next 2 from pool ────────────┐ │
  │                                 │  │  │                                    │ │
  │                                 │  │  │ Editor (image gen API)             │ │
  │                                 │  │  │    ↓                               │ │
  │                                 │  │  │ Critic (SSIM → blind → informed)   │ │
  │                                 │  │  │    ↓                               │ │
  │                                 │  │  │ Pass? ── yes ── save & continue    │ │
  │                                 │  │  │    │                               │ │
  │                                 │  │  │    no (< max retries)              │ │
  │                                 │  │  │    ↓                               │ │
  │                                 │  │  │ Refiner → revised edit prompt      │ │
  │                                 │  │  │    ↓                               │ │
  │                                 │  │  │ (back to Editor)                   │ │
  │                                 │  │  └────────────────────────────────────┘ │
  │                                 │  │  2 accepted? Stop. Else: next batch.    │
  │                                 │  └──────────────────────────────────────────┘
  │                                 │
  │                                 ├─ Results persisted after each variant
  │                                 ├─ Job status → COMPLETED
  │                                 │
  ├─ GET /api/v1/jobs/{id} ────────►│  (partial results while running)
  │◄── {status, results, audits} ───┤
```

Each `VariantResult` in the response contains:
- The edited image (base64)
- An evaluation score (0.0-1.0)
- Evaluation feedback from the critic
- A full audit trail showing every decision made by every agent at every attempt
- The text and image provider/model that produced it
