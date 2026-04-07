# Codebase Walkthrough

A file-by-file tour of the backend. If you've read the README and want to understand *how* things work rather than *what* they do, this is the place.

---

## `app/main.py` — the FastAPI app

Everything starts here. The main flow is straightforward: client uploads an image + JSON, we validate it, kick off a background task, return 202 immediately. The client polls `GET /api/v1/jobs/{job_id}` until results show up.

The validation is fairly aggressive — content type, file size (50 MB), image dimensions (8192px), recommendation count (5), text length (2000 chars). There's also an RGBA check that doesn't reject but warns, because some image providers handle alpha channels inconsistently.

The background task (`_process_job`) does two things in parallel: runs the actual agentic workflow, and sends heartbeat pings to the database every 30 seconds. The heartbeat is how we detect orphaned jobs — if the backend crashes mid-flight, a cleanup sweep picks up the stale heartbeat and marks the job as failed. On startup, we also sweep for any "running" or "pending" jobs left over from a previous process and fail them immediately, since their in-memory state is gone.

Results stream in incrementally. Every time a variant finishes (pass or fail), the `on_variant_complete` callback writes the current results to PostgreSQL. So the polling endpoint doesn't just return "running" — it returns "running, and here are the 3 variants that are done so far."

There are a bunch of other endpoints in here too — cancel, feedback, thumbnails, health checks, experiment comparison, metrics. The static HTML frontend is also served from here via `StaticFiles`.

---

## `app/config.py` — settings

One `Settings` class, everything from env vars. The interesting part is that text and image providers are independently configurable — you can use Gemini for vision/text and OpenAI for image generation, or any combination. Only the active provider's API key is required.

`model_config = {"extra": "ignore"}` is there so you can have old env vars lying around without pydantic complaining.

---

## `app/models.py` — data contracts

All the Pydantic models in one file. The key thing to understand is the nesting: a `Job` has a list of `RecommendationResult`s, each of which has a list of `VariantResult`s. Each variant tracks which provider and model produced it (for the experiment comparison endpoint later).

`Job.to_response()` strips the original image when returning data to the client, since it's large and the client already has it.

---

## `app/db.py` — PostgreSQL persistence

Two tables: `jobs` (with JSONB for results, recommendations, and brand guidelines) and `feedback` (with provider/model context for later analysis).

Everything is wrapped in try/except. If the database goes down, the app keeps running on in-memory state only. This was a deliberate choice — I didn't want a database failure to take down the whole service. The tradeoff is that you lose durability if PostgreSQL is unreachable, but the app stays up.

The heartbeat system lives here too. `touch_heartbeat()` updates a timestamp, and `fail_stale_jobs()` sweeps for anything that hasn't checked in within 5 minutes.

---

## `app/services/image.py` — image utilities

Small file. `validate_image()` runs bytes through Pillow to make sure they're actually a valid image. A few base64 conversion helpers.

---

## `app/services/llm.py` — the LLM abstraction

This is where all the provider-specific code lives. Two public functions: `vision_chat()` (text/vision LLM) and `edit_image()` (image generation). Agents call these without knowing or caring which provider is behind them.

Three providers are implemented:
- **OpenAI** — uses `with_raw_response` for vision chat so we can read rate limit headers from the response
- **Gemini** — the image editor uses the sync SDK in a thread executor because the async client had issues. Retries up to 3 times on empty responses (Gemini sometimes returns no image, usually a throttling signal)
- **Claude** — vision/text only, no image generation (Anthropic doesn't have one), so it's skipped during image editing fallback

The rate limiter is the most interesting part of this file. Each provider gets its own `_AdaptiveRateLimiter`. It starts at a default RPM (30), but the first time we get an OpenAI response, we read `x-ratelimit-limit-requests` from the headers and adjust to 80% of the actual limit. On 429 errors it halves the rate. After 5 consecutive successes it starts recovering. This means you don't need to manually configure rate limits — the system figures it out.

`_is_rate_limit_error()` is careful to distinguish transient rate limits from permanent "insufficient_quota" errors. If your API key is out of credits, backing off won't help, so we don't treat it as a rate limit.

The fallback chain is per-call: if the primary provider fails, we try alternates in order. A single failed call doesn't permanently switch the default — next time we still try the primary first.

---

## `app/workflow/state.py` — graph state

A `TypedDict` that flows through the LangGraph graph. Immutable inputs on one side (original image, recommendation, brand guidelines), mutable state on the other (edit prompt, edited image, evaluation results, attempt counter). The `audit_trail` list is an accumulator that every agent appends to.

---

## `app/workflow/graph.py` — orchestration

Two levels of orchestration happen here.

**The inner graph** handles a single variant through the edit-evaluate-refine loop:

```
editor → critic → should_retry?
                    ├── yes → refiner → editor (loop back)
                    └── no  → END
```

This graph is compiled once at import time and reused for every variant. The entry point is `editor` because the edit prompt is already in state — the ideator put it there.

**The outer orchestration** is plain Python, not a graph. For each recommendation:

1. The ideator generates a pool of 10 variant ideas (each with an edit prompt)
2. The idea critic reviews the whole pool and filters out weak ones
3. We draw from the pool in batches of 2, running each batch through the inner graph concurrently
4. After each batch, we check: do we have 2 accepted variants? If yes, stop. If no, grab the next 2 from the pool.

This pool-and-batch approach was a deliberate choice over generating ideas one at a time. If a batch fails, the next ideas are already sitting in the pool — no extra LLM call to brainstorm replacements. The ideator runs once, and then it's all execution.

`run_all_recommendations()` runs all recommendations concurrently via `asyncio.gather`. It pre-creates empty `RecommendationResult` objects before starting, so they show up in the API response immediately (with zero variants) and fill in as results arrive.

---

## The five agents

### `app/workflow/agents/ideator.py`

Takes a vague recommendation like "strengthen headline impact" and brainstorms 10 specific variant ideas, each with a concrete edit prompt that can go straight to the image editor. The system prompt asks for specific colours, positions, sizes, effects — not abstract concepts.

Retries up to 3 times if the JSON doesn't parse. On the second attempt it adds a "keep descriptions shorter" hint, because the most common failure mode is the LLM's output getting truncated mid-JSON. If all retries fail, it falls back to a single pass-through variant using the original recommendation text, so the rest of the pipeline always has something to work with.

### `app/workflow/agents/idea_critic.py`

Reviews all 10 ideas together in a single call. This matters — reviewing them individually wouldn't catch duplicates. It evaluates visual impact, recommendation alignment, brand compliance, distinctiveness, and feasibility.

The prompt explicitly calls out subtle changes as automatic rejections: "slightly increase letter spacing", "marginally bolder font", that kind of thing. This was added after I kept seeing the image editor return the original image unchanged when given weak prompts. Catching those ideas early saves multiple expensive image generation calls.

On parse failure, it approves everything. Better to waste some image generation calls on mediocre ideas than to block the pipeline.

### `app/workflow/agents/editor.py`

The simplest agent. Reads the edit prompt from state, calls `edit_image()`, stores the result. That's it. It's a bridge between the ideator's intent and whatever image generation API is configured.

### `app/workflow/agents/critic.py`

This one went through several iterations, because the original version had a fundamental problem: the image editor would return the original image unchanged, and the critic would *accept it* because it knew what changes to look for and convinced itself it could see them. Classic anchoring bias.

The fix was three layers of evaluation:

**Layer 1 — SSIM pre-check.** Computes structural similarity between the original and edited images. If SSIM > 0.95, the images are pixel-identical (or close enough). Auto-reject, no LLM call needed. Images are downscaled to 512px first so this is fast.

**Layer 2 — Blind comparison.** Sends both images to the LLM and asks "what's different between these two images?" — crucially, without telling it what the edit was supposed to be. If the model says "no visible difference", we reject. This catches cases where the images are technically different (different compression, minor colour shifts) but a human wouldn't notice.

**Layer 3 — Full evaluation.** Only now do we tell the LLM what the recommendation was and ask it to evaluate compliance. The blind comparison's findings are included as context, so the model can't ignore its own earlier assessment. There's also a `visually_different` field in the response — if the model says false, we force a failure regardless of the score.

### `app/workflow/agents/refiner.py`

When the critic rejects a variant, the refiner reads the feedback and writes a revised edit prompt. The critic says "the text is still too small", the refiner says "make the headline text 3x larger and set it in bold white against the dark background." This revised prompt goes straight back to the editor.

The separation between critic and refiner exists because they're doing different jobs. The critic evaluates ("what's wrong"), the refiner prescribes ("what to do differently"). Asking the editor to interpret raw evaluation feedback directly doesn't work as well.

---

## `app/metrics.py` — Prometheus metrics

Counters, histograms, and gauges for everything: job lifecycle, variant outcomes, LLM call duration (by provider and type), agent invocations, parse failures, dependency health, rate limit state, provider fallbacks. These get scraped by Prometheus and displayed in the Grafana dashboard.

---

## Infrastructure

**`Dockerfile`** — Python 3.12 slim, non-root user, requirements installed first for layer caching.

**`docker-compose.yml`** (at the project root) — five services: PostgreSQL, the backend (with volume mounts for hot-reload), pgAdmin, Prometheus, and Grafana. The Grafana datasource and dashboard are auto-provisioned on first boot.

**`tests/`** — 76 tests across 3 files, all mocked, run in about 3 seconds. `test_agents.py` covers agent logic and the three-layer critic. `test_api.py` covers input validation and the job lifecycle. `test_rate_limiter.py` covers the adaptive rate limiter's backoff, recovery, and error classification.

---

## Data flow

```
Client                            Backend
  │                                 │
  ├─ POST /api/v1/jobs ────────────►│
  │   (image + JSON)                │
  │                                 ├─ Validate, persist, spawn task
  │◄── 202 {job_id} ───────────────┤
  │                                 │
  │                                 │  For each recommendation (concurrent):
  │                                 │
  │                                 │    Ideator → 10 ideas with edit prompts
  │                                 │    Idea Critic → filter pool
  │                                 │
  │                                 │    Batch loop (2 at a time from pool):
  │                                 │      Editor → Critic (SSIM/blind/full) → pass?
  │                                 │        no → Refiner → Editor (retry, up to 3x)
  │                                 │      Stop when 2 accepted or pool empty
  │                                 │
  │                                 │    Results written to DB after each variant
  │                                 │
  ├─ GET /api/v1/jobs/{id} ────────►│
  │◄── {status, results, audits} ───┤  (partial results while still running)
```

Each variant in the response includes the edited image, an evaluation score, the critic's feedback, a full audit trail, and which provider/model produced it.
