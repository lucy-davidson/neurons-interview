# Codebase Walkthrough

> **NOTE:** This walkthrough was written for an earlier version of the architecture. Key changes since then:
> - **Planner and Plan Critic agents removed** -- the Ideator now produces edit prompts directly
> - **Idea Critic added** -- reviews all variant ideas before image generation
> - **Critic enhanced** -- now includes SSIM pre-check and blind visual comparison
> - **Pool-and-batch processing** -- 10 ideas generated, tried 2 at a time
> - **Adaptive rate limiting** -- per-provider, auto-configures from API headers
> - **Static HTML frontend** -- replaced Gradio, served by FastAPI
> - **Grafana + pgAdmin** -- replaced custom Gradio monitor
> - **76 tests** across 3 files (was 18 in 1 file)
> - **3 LLM providers** -- OpenAI, Gemini, Claude with automatic fallback
>
> See the README.md for the current architecture and the Notion docs for detailed explanations.

This document walks through every file in the backend, explaining what it does, why it exists, and how it connects to the rest of the system.

---

## 1. Entry Point: `app/main.py`

This is the FastAPI application. It's where HTTP requests arrive and where the async job lifecycle is managed.

**What happens when a request comes in:**

1. `POST /api/v1/jobs` receives a multipart form upload with two parts:
   - `image` — the marketing creative (PNG, JPEG, or WebP)
   - `request_body` — a JSON file containing the recommendations and brand guidelines

2. The endpoint validates the image type and structure (using Pillow via `services/image.py`), parses the JSON body into a `JobRequest` model, and creates a `Job` object with a unique ID.

3. The job is stored in an in-memory dictionary (`_jobs`) and an `asyncio.create_task` fires off `_process_job()` in the background. The endpoint immediately returns `202 Accepted` with the job ID — the client doesn't wait for processing to finish.

4. `_process_job()` calls `run_all_recommendations()` from the workflow module. When it completes, it writes the results back to the job object and flips the status to `COMPLETED` (or `FAILED` on error).

5. `GET /api/v1/jobs/{job_id}` lets the client poll for status. Once `COMPLETED`, the response includes all recommendation results with their edited images, scores, and audit trails.

6. `GET /api/v1/jobs/{job_id}/image/{recommendation_id}` is a convenience endpoint to fetch a single edited image.

**Why this pattern:** The agentic workflow involves multiple LLM calls and image generation steps that can take 30-60+ seconds. An async job queue pattern (submit → poll) prevents HTTP timeouts and lets the frontend show progress states.

---

## 2. Configuration: `app/config.py`

A single `Settings` class using `pydantic-settings`. All configuration is driven by environment variables (or a `.env` file). This includes:

- `openai_api_key` — the only required value
- `openai_vision_model` / `openai_image_model` — model names, easily swappable
- `max_retries` — how many plan→edit→evaluate cycles to attempt before giving up
- `image_size` — output dimensions for generated images
- `log_level` — controls logging verbosity

**Why a separate config file:** Centralises all tuneable parameters so they can be changed via environment variables without touching code. The Docker Compose file passes these through from the `.env` file.

---

## 3. Data Models: `app/models.py`

All Pydantic models live here. This file defines the contract between every layer of the system.

**Request models:**
- `Recommendation` — a single recommendation with `id`, `title`, `description`, and `type` (e.g. "contrast_salience", "composition")
- `BrandGuidelines` — protected regions, typography rules, aspect ratio, brand elements
- `JobRequest` — the JSON body sent by the client (list of recommendations + guidelines)

**Response models:**
- `JobResponse` — returned by the API: job ID, status, creation time, list of results
- `RecommendationResult` — per-recommendation outcome: status ("accepted" or "max_retries_exceeded"), attempt count, edited image, evaluation score/feedback, and the full audit trail
- `AuditEntry` — a single timestamped log entry from one of the agents (agent name, action, detail text)

**Internal models:**
- `Job` — the full internal job record that includes the original image and all inputs. The `to_response()` method strips the original image (which is large) when returning data to the client.
- `JobStatus` — enum: `pending`, `running`, `completed`, `failed`

**Why one models file:** The models are the shared language between the API layer, the workflow layer, and the agent layer. Keeping them in one place makes the data contracts easy to review and prevents circular imports.

---

## 4. Services Layer

### `app/services/image.py`

Image utility functions:
- `load_image_b64(path)` — reads a file from disk and returns base64 (useful for testing with the sample creatives)
- `b64_to_bytes()` / `bytes_to_b64()` — conversions between base64 strings and raw bytes
- `validate_image(data)` — uses Pillow to verify that uploaded bytes are actually a valid image

### `app/services/llm.py`

Thin async wrappers around the OpenAI API. There are two functions:

**`vision_chat(system_prompt, user_text, image_b64, second_image_b64)`**
- Sends a chat completion request to GPT-4o with vision capabilities
- Supports sending one or two images (the critic needs to compare original vs edited)
- Used by the planner, critic, and refiner agents
- Returns the raw text response

**`edit_image(prompt, image_b64)`**
- Sends an image edit request to gpt-image-1
- Takes the original image and an edit prompt, returns the edited image as base64
- Used by the editor agent
- Handles both `b64_json` and `url` response formats from the API

**Why wrap the API calls:** Isolates the OpenAI-specific logic so agents don't need to know about API details. If you wanted to swap to a different provider (Stability AI, Anthropic, etc.), you'd only change this file.

---

## 5. Workflow Layer

This is the heart of the system — the LangGraph-based agentic workflow.

### `app/workflow/state.py`

Defines `RecommendationState` as a `TypedDict`. This is the state object that flows through the LangGraph graph. Every agent reads from and writes to this state.

Key fields:
- **Inputs** (set once): `original_image_b64`, recommendation details, brand guidelines text
- **Mutable** (updated by agents): `plan`, `edit_prompt`, `edited_image_b64`, evaluation results, `refiner_feedback`, `attempt` counter
- **Accumulator**: `audit_trail` — every agent appends entries here, building the full decision log

**Why TypedDict:** LangGraph uses TypedDict for state definitions. It gives type safety while remaining serialisable.

### `app/workflow/graph.py`

This file constructs, compiles, and runs the LangGraph StateGraph. It's the orchestration layer.

**Graph structure:**
```
planner → editor → critic → should_retry?
                              ├── "refiner" → refiner → planner (loops back)
                              └── "end"     → END
```

**`build_recommendation_graph()`** creates the graph:
1. Adds four nodes (one per agent function)
2. Sets the entry point to `planner`
3. Wires the linear path: planner → editor → critic
4. Adds a conditional edge after `critic`: the `should_retry()` function checks if the evaluation passed or if the max attempt count was reached. If neither, it routes to `refiner`, which feeds back into `planner` for another loop.

The graph is compiled once at module import time (`_compiled_graph`) for efficiency.

**`run_recommendation_workflow()`** executes the graph for a single recommendation:
1. Builds the initial state dict from the inputs
2. Calls `_compiled_graph.ainvoke(initial_state)` — this runs the full graph asynchronously until it reaches END
3. Reads the final state to build a `RecommendationResult`

**`run_all_recommendations()`** processes all recommendations concurrently:
- Creates one `run_recommendation_workflow` coroutine per recommendation
- Runs them all with `asyncio.gather` — this is the parallelism the assignment requires
- Returns a list of results

**Why LangGraph:** The assignment calls for an agentic workflow with conditional iteration. LangGraph's StateGraph gives us declarative graph construction with conditional edges, which maps cleanly to the plan→edit→evaluate→retry loop. It also provides built-in support for async execution.

---

## 6. The Four Agents

Each agent is a single async function that takes `RecommendationState`, does its work, and returns an updated copy of the state. They all follow the same pattern: read from state → call an LLM → parse the response → append to audit trail → return updated state.

### `app/workflow/agents/planner.py`

**Role:** Interpret the recommendation and brand guidelines, produce an actionable plan and a concrete edit prompt.

**How it works:**
1. Builds a user message containing the recommendation details, brand guidelines, and (on retries) the refiner's feedback from the previous attempt
2. Sends this along with the original image to GPT-4o via `vision_chat()`
3. The system prompt instructs the model to return JSON with two keys: `plan` (step-by-step reasoning) and `edit_prompt` (the actual prompt for the image editor)
4. Parses the JSON response (with a graceful fallback if the model doesn't return valid JSON)
5. Appends to the audit trail and returns the updated state

**On retries:** The planner receives the refiner's feedback via the `refiner_feedback` state field. Its system prompt tells it to adjust the plan to address that feedback. This is how the system learns from failed attempts within a single job.

### `app/workflow/agents/editor.py`

**Role:** Generate the edited image.

**How it works:**
1. Reads the `edit_prompt` from state (set by the planner)
2. Calls `edit_image()` with the original image and the edit prompt
3. Stores the resulting base64 image in `edited_image_b64`
4. Appends to the audit trail

This is the simplest agent — it's essentially a bridge between the planner's intent and the image generation model.

### `app/workflow/agents/critic.py`

**Role:** Evaluate the edited image against the recommendation and brand guidelines.

**How it works:**
1. Sends both the original and edited images to GPT-4o vision, along with the recommendation and brand guidelines
2. The system prompt asks for evaluation on two axes:
   - **Recommendation compliance** — was the recommendation effectively applied?
   - **Brand guideline compliance** — were all brand constraints respected?
3. Expects a JSON response with: `passed` (bool), `score` (0.0–1.0), compliance assessments, and `feedback` (actionable improvement notes)
4. Stores evaluation results in state and appends to audit trail

**The conditional edge:** After the critic runs, LangGraph calls `should_retry()` which checks `evaluation_passed`. If `False` and `attempt < max_attempts`, the graph routes to the refiner. If `True` or retries exhausted, it routes to END.

### `app/workflow/agents/refiner.py`

**Role:** Bridge between the critic's feedback and the planner's next attempt.

**How it works:**
1. Takes the critic's evaluation feedback, the plan that was used, and the recommendation
2. Sends these to GPT-4o with a system prompt asking for 2-4 sentences of specific, actionable guidance
3. Stores the refinement guidance in `refiner_feedback` and increments the `attempt` counter
4. The graph then loops back to `planner`, which will use this feedback

**Why a separate refiner:** The critic's output is evaluative ("this failed because X"). The refiner translates that into prescriptive guidance ("next time, do Y instead"). This separation keeps each agent focused on one responsibility and produces better results than asking the planner to self-correct from raw evaluation feedback.

---

## 7. Infrastructure

### `Dockerfile`

A straightforward Python 3.12 slim image:
1. Copies and installs `requirements.txt` first (layer caching — dependencies change less often than code)
2. Copies the application code
3. Exposes port 8000
4. Runs uvicorn

### `docker-compose.yml`

Defines the backend service:
- Builds from the Dockerfile
- Maps port 8000
- Loads environment from `.env`
- Mounts `./app` as a volume for hot-reload during development
- Overrides CMD to use `--reload` for development
- Includes a health check hitting `/health`

### `requirements.txt`

Key dependencies:
- `fastapi` + `uvicorn` — web framework and ASGI server
- `python-multipart` — required for file upload parsing
- `pydantic` + `pydantic-settings` — data validation and configuration
- `openai` — the OpenAI Python SDK (async support)
- `langgraph` — the agentic workflow framework
- `Pillow` — image validation
- `httpx` — async HTTP client (fallback for URL-based image responses)

### `tests/test_api.py`

Six tests covering the API layer with mocked LLM calls:
1. **Health check** — basic smoke test
2. **Invalid image type** — rejects non-image uploads with 400
3. **Invalid JSON body** — rejects malformed request body with 422
4. **Job not found** — returns 404 for unknown job IDs
5. **Create and poll** — full happy path with mocked workflow (verifies 202 response)
6. **Empty recommendations** — rejects jobs with no recommendations

All tests use FastAPI's `TestClient` (synchronous) and `unittest.mock.patch` to avoid real API calls.

---

## 8. Data Flow Summary

Here's the complete journey of a request through the system:

```
Client                          Backend
  │                               │
  ├─ POST /api/v1/jobs ──────────►│
  │   (image + JSON)              │
  │                               ├─ Validate image (Pillow)
  │                               ├─ Parse JobRequest (Pydantic)
  │                               ├─ Create Job (in-memory store)
  │                               ├─ Spawn background task
  │◄── 202 {job_id, status} ──────┤
  │                               │
  │                               │  Background: run_all_recommendations()
  │                               │  ┌─ asyncio.gather ─────────────────┐
  │                               │  │                                  │
  │                               │  │  For each recommendation:        │
  │                               │  │  ┌─ LangGraph ainvoke ────────┐ │
  │                               │  │  │ Planner (GPT-4o vision)    │ │
  │                               │  │  │    ↓                       │ │
  │                               │  │  │ Editor (gpt-image-1)       │ │
  │                               │  │  │    ↓                       │ │
  │                               │  │  │ Critic (GPT-4o vision)     │ │
  │                               │  │  │    ↓                       │ │
  │                               │  │  │ Pass? ── yes ── END        │ │
  │                               │  │  │    │                       │ │
  │                               │  │  │    no                      │ │
  │                               │  │  │    ↓                       │ │
  │                               │  │  │ Refiner (GPT-4o)           │ │
  │                               │  │  │    ↓                       │ │
  │                               │  │  │ (back to Planner)          │ │
  │                               │  │  └────────────────────────────┘ │
  │                               │  └─────────────────────────────────┘
  │                               │
  │                               ├─ Job status → COMPLETED
  │                               │
  ├─ GET /api/v1/jobs/{id} ──────►│
  │◄── {status, results, audits} ─┤
```

Each `RecommendationResult` in the response contains:
- The edited image (base64)
- An evaluation score (0.0–1.0)
- Evaluation feedback from the critic
- A full audit trail showing every decision made by every agent at every attempt
