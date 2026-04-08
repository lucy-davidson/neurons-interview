# Visual Recommendations Backend

An asynchronous, containerised backend service that implements an agentic workflow for generating visual recommendations on marketing creatives. Built for the Neurons AI platform context, where textual recommendations are applied as visual edits to advertising images while respecting brand guidelines.

## Quick Start

**Prerequisites:** Docker and Docker Compose

```bash
git clone <repo-url>

# 1. Add your API keys
cp backend/.env.example backend/.env
# Edit backend/.env — fill in GOOGLE_API_KEY (and optionally OPENAI_API_KEY, ANTHROPIC_API_KEY)

# 2. Symlink .env so docker-compose can read variables
ln -sf backend/.env .env

# 3. Start everything (from project root)
docker-compose up --build
```

Five services come up:

| Service | URL | Purpose |
|---------|-----|---------|
| **Backend + Frontend** | http://localhost:8000 | REST API + static HTML frontend |
| **Grafana** | http://localhost:3000 | Metrics dashboards (no login required) |
| **pgAdmin** | http://localhost:5050 | Database browser |
| **Prometheus** | http://localhost:9090 | Metrics collection |
| **PostgreSQL** | internal only | Job persistence |

The frontend has a **"Load Sample Data"** dropdown for quick demos -- select a sample creative, and the image, recommendations, and brand guidelines are pre-filled.

## Architecture

The service uses a **multi-agent LangGraph workflow** with five specialised agents:

```
Ideator (generates pool of 10 variant ideas with edit prompts)
   │
   └─► Idea Critic (reviews all ideas, rejects subtle/duplicate ones)
          │
          └─► Pool of approved ideas
                │
                └─► Try in batches of 2:
                      │
                      Editor ──► Critic ──► Pass?
                                              │ no
                                    Refiner ──┘ (revised edit prompt, retry up to 3x)
                      │
                      Keep drawing from pool until 2 accepted or pool exhausted
```

| Agent | Role | Model |
|-------|------|-------|
| **Ideator** | Brainstorms 10 variant ideas with concrete edit prompts | Gemini 2.5 Flash (vision) |
| **Idea Critic** | Reviews all ideas together, rejects subtle/infeasible ones | Gemini 2.5 Flash |
| **Editor** | Generates the edited image | Gemini 2.5 Flash Image |
| **Critic** | SSIM pre-check + blind visual comparison + LLM evaluation | Gemini 2.5 Flash (vision) |
| **Refiner** | Translates critic feedback into a revised edit prompt | Gemini 2.5 Flash |

Three LLM providers supported with automatic fallback: **Gemini**, **OpenAI**, **Claude**. Text and image providers are independently configurable.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/jobs` | Submit a new job (multipart: image + JSON). Returns `202`. |
| `GET` | `/api/v1/jobs` | List all jobs (lightweight summaries). |
| `GET` | `/api/v1/jobs/{job_id}` | Poll job status and results (incremental). |
| `GET` | `/api/v1/jobs/{job_id}/image/{variant_id}` | Fetch full variant image (base64). |
| `GET` | `/api/v1/thumb/{job_id}/{variant_id}` | Fetch JPEG thumbnail. |
| `POST` | `/api/v1/jobs/{job_id}/cancel` | Cancel a running job. |
| `POST` | `/api/v1/jobs/{job_id}/feedback` | Record user feedback on a variant. |
| `GET` | `/api/v1/feedback/stats` | Aggregate feedback statistics. |
| `GET` | `/api/v1/health/dependencies` | Live health check for all providers + DB. |
| `GET` | `/api/v1/experiments/comparison` | Provider comparison data. |
| `POST` | `/api/v1/experiments` | Create an experiment with config variations. |
| `GET` | `/api/v1/experiments` | List all experiments. |
| `GET` | `/api/v1/experiments/{id}` | Experiment detail with variations. |
| `POST` | `/api/v1/experiments/{id}/run` | Run all variations with same input. Returns `202`. |
| `GET` | `/api/v1/experiments/{id}/results` | Per-variation comparison (acceptance rate, scores). |
| `GET` | `/api/v1/metrics/summary` | System metrics summary (JSON). |
| `GET` | `/metrics` | Prometheus scrape endpoint. |
| `GET` | `/health` | Liveness probe with DB status. |

### Creating a Job

```bash
curl -X POST http://localhost:8000/api/v1/jobs \
  -F "image=@creative_1.png" \
  -F "request_body=@request.json"
```

Where `request.json` contains:

```json
{
  "recommendations": [
    {
      "id": "rec_1",
      "title": "Strengthen Headline Impact",
      "description": "Add visual punch to the headline...",
      "type": "contrast_salience"
    }
  ],
  "brand_guidelines": {
    "protected_regions": ["Do not modify or remove the brand logo"],
    "typography": "Maintain existing font style and hierarchy",
    "aspect_ratio": "Maintain original aspect ratio (1572x1720)",
    "brand_elements": "Ensure logo remains visible and legible"
  }
}
```

### Polling for Results

```bash
curl http://localhost:8000/api/v1/jobs/{job_id}
```

Results stream in incrementally -- each variant appears as it completes.

## Configuration

All settings via environment variables (`.env` file):

| Variable | Default | Description |
|----------|---------|-------------|
| `TEXT_PROVIDER` | `openai` | Provider for vision/text calls (`openai`, `gemini`, or `claude`) |
| `IMAGE_PROVIDER` | `gemini` | Provider for image generation (`gemini` or `openai`) |
| `GOOGLE_API_KEY` | *(required)* | Google AI API key |
| `OPENAI_API_KEY` | *(optional)* | OpenAI API key (for fallback or comparison) |
| `ANTHROPIC_API_KEY` | *(optional)* | Anthropic API key (for fallback or comparison) |
| `POSTGRES_USER` | `visrec` | PostgreSQL username |
| `POSTGRES_PASSWORD` | *(required)* | PostgreSQL password |
| `CORS_ORIGINS` | `*` | Comma-separated allowed origins |
| `NUM_VARIANTS` | `2` | Target accepted variants per recommendation |
| `MAX_RETRIES` | `3` | Max edit-evaluate retry attempts per variant |
| `LLM_RATE_LIMIT` | `30` | Starting max LLM API calls per minute (auto-adjusts) |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

## Persistence

Jobs are stored in **PostgreSQL** with heartbeat monitoring. Results are written incrementally after each variant completes. A background cleanup sweep marks jobs as failed if their heartbeat goes stale (5 min timeout). On startup, orphaned jobs from previous sessions are recovered.

## Running Tests

```bash
cd backend
pip install -r requirements.txt pytest
python -m pytest tests/ -v
```

106 tests across 3 files:
- `test_agents.py` (27) -- agent logic, SSIM, blind comparison, routing
- `test_api.py` (63) -- validation, lifecycle, cancel, feedback, thumbnails, DB, metrics, config overrides, experiments, error messages
- `test_rate_limiter.py` (16) -- error detection, adaptive backoff, per-provider isolation

All tests mock LLM calls -- no API keys needed.

## Project Structure

```
backend/
├── app/
│   ├── main.py              # FastAPI app, endpoints, static frontend, job lifecycle
│   ├── config.py            # Pydantic settings (multi-provider config)
│   ├── models.py            # Pydantic models (Job, Variant, Audit, Feedback)
│   ├── db.py                # PostgreSQL persistence + heartbeat + feedback
│   ├── metrics.py           # Prometheus counters, histograms, gauges
│   ├── services/
│   │   ├── image.py         # Image validation (PIL)
│   │   ├── llm.py           # LLM abstraction (3 providers) + adaptive rate limiting + token tracking
│   │   └── cost.py          # Per-model cost estimation tables
│   └── workflow/
│       ├── state.py         # LangGraph state definition
│       ├── graph.py         # Pool-and-batch orchestration + variant duration tracking
│       ├── prompt_versions.py # SHA-256 hashes of all agent prompts
│       └── agents/
│           ├── __init__.py  # timed_agent decorator (duration + token tracking)
│           ├── ideator.py   # Recommendation → 10 variant ideas with edit prompts
│           ├── idea_critic.py # Idea quality gate (rejects subtle/duplicate)
│           ├── editor.py    # Edit prompt → edited image
│           ├── critic.py    # SSIM + blind comparison + LLM evaluation
│           └── refiner.py   # Critic feedback → revised edit prompt
├── static/
│   ├── index.html           # Vanilla JS frontend
│   └── samples/             # Sample creative images
├── tests/
│   ├── test_api.py          # 33 API endpoint tests
│   ├── test_agents.py       # 27 agent + quality check tests
│   └── test_rate_limiter.py # 16 rate limiter tests
├── monitoring/
│   ├── prometheus/          # Scrape config
│   ├── grafana/             # Dashboard + datasource provisioning
│   └── pgadmin/             # Auto-connect config
├── Dockerfile
├── requirements.txt
├── .env.example
└── .gitignore
```

`docker-compose.yml` is at the project root (one level up).

## Design Decisions

- **Pool-and-batch**: Generate 10 ideas upfront, try 2 at a time, draw from pool on failure. No extra LLM calls to generate replacements mid-flight.
- **Three-layer critic**: SSIM pre-check (deterministic) → blind comparison (unbiased LLM) → full evaluation (informed LLM). Catches unchanged images at every level.
- **Adaptive rate limiting**: Per-provider token-bucket that auto-configures from OpenAI response headers and backs off on errors. No manual tuning needed.
- **Independent providers**: Text and image providers are separately configurable. Three providers (OpenAI, Gemini, Claude) with automatic fallback chain.
- **Incremental streaming**: Variants append to results as they complete. Frontend polls and renders partial results with JPEG thumbnails.
- **PostgreSQL + in-memory dual-write**: Active jobs fast (memory); completed jobs durable (DB). Heartbeat monitoring catches orphaned jobs.

## Limitations

- **Rate limiting**: Free-tier API limits constrain throughput. The adaptive limiter helps but can back down to 1 RPM under heavy throttling.
- **No task queue**: Background jobs use `asyncio.create_task`. Production would use Celery for horizontal scaling.
- **No API authentication**: Anyone who can reach the server can submit jobs.
- **Base64 image storage**: Images stored as base64 in PostgreSQL. Production would use object storage (S3/GCS).
```
