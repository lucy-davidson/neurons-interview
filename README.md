# Neurons AI Engineer Assignment — Visual Recommendations

Takes a marketing creative, a set of textual recommendations, and brand guidelines, then generates edited visual variants that apply each recommendation while respecting the brand constraints. Built with FastAPI, LangGraph, and Gemini/OpenAI/Claude.

## Quick Start

```bash
cp backend/.env.example backend/.env    # add your API keys
ln -sf backend/.env .env                # docker-compose reads vars from root .env
docker-compose up --build
```

Frontend at http://localhost:8000. Grafana at http://localhost:3000. pgAdmin at http://localhost:5050.

Full docs in [backend/README.md](backend/README.md).

---

## How it works

Five agents collaborate through a LangGraph state graph:

1. **Ideator** — reads the image and recommendation, brainstorms 10 different variant ideas, each with a concrete edit prompt
2. **Idea Critic** — reviews all 10 ideas as a group, throws out anything too subtle or duplicative
3. **Editor** — sends the edit prompt + original image to Gemini's image generation API
4. **Critic** — compares original vs edited image using SSIM (pixel-level), a blind "what changed?" check (no anchoring bias), then a full LLM evaluation
5. **Refiner** — if the critic rejects, rewrites the edit prompt based on what went wrong and sends it back to the editor

Variants are tried in batches of 2 from the pool of approved ideas. If a batch fails, the system draws the next ideas from the pool. It keeps going until 2 variants pass or the pool runs out. All recommendations are processed concurrently.

---

## Assignment requirements

### Backend service (required)

- FastAPI with async job lifecycle — POST to submit, GET to poll, results stream in as variants complete
- Full input validation — file size (50MB), dimensions (8192px), recommendation count (5), text length (2000 chars), RGBA detection
- Structured JSON logging with job-level correlation IDs
- Docker Compose with 5 services (backend, PostgreSQL, Prometheus, Grafana, pgAdmin)

### Multi-agent workflow (required)

- LangGraph StateGraph with conditional routing (critic → refiner → editor retry loop)
- Agents communicate through a shared typed state dictionary, not direct calls
- Full audit trail — every agent decision is timestamped and logged

### Deliverables (required)

- Documented codebase with docstrings on all modules/classes/functions
- README with setup instructions, `.env.example` with inline comments
- WALKTHROUGH.md with code-level explanations

### Unit tests (optional — implemented)

106 tests across 3 files, all run in ~7 seconds with no API keys:
- 27 agent tests — parsing, fallbacks, SSIM, blind comparison, routing edge cases
- 63 API tests — validation, job lifecycle, cancel, feedback, thumbnails, DB persistence, config overrides, experiments, error messages
- 16 rate limiter tests — error detection, adaptive backoff, per-provider isolation

### Frontend (optional — implemented)

Static HTML/JS served by FastAPI. No framework, no build step. Sample data pre-loads the image + recommendations + brand guidelines in one click. Thumbnails load as JPEGs; full images fetch on click. Feedback buttons (like/dislike/refine/use this) persist in localStorage. Cancel button for running jobs.

---

## What I added beyond the spec

**Multi-provider support** — OpenAI, Gemini, and Claude with independent text/image provider selection. If one goes down, the system falls back to the next automatically. Per-provider adaptive rate limiting reads `x-ratelimit-limit-requests` from OpenAI response headers and adjusts throughput. Backs off on 429s, recovers after consecutive successes.

**Image quality verification** — the biggest problem I ran into was the image editor returning the original image unchanged. SSIM catches pixel-identical images. The blind comparison asks "what's different?" without telling the LLM what the edit should have been — this prevents it from convincing itself it sees a change that isn't there.

**PostgreSQL persistence** — jobs survive backend restarts. A heartbeat system (updated every 30s) catches orphaned jobs — if a worker crashes, a background sweep marks the job as failed after 5 minutes. Results are written to the DB incrementally as each variant completes.

**Observability** — Prometheus metrics for everything (job lifecycle, LLM latency by provider, variant acceptance rates, parse failures, fallback counts). Grafana dashboard with 14 panels, auto-provisioned on first boot. Dependency health checks ping all three LLM providers and PostgreSQL every 30 seconds.

**User feedback** — like/dislike/refine/use-this buttons on each variant. Stored in PostgreSQL with the provider, model, and critic score for later analysis. The idea is to eventually use this to calibrate the critic and tune the ideator prompts.

**Experiment scaffolding** — per-job config overrides let you change the provider, model, or workflow parameters (retries, variant count) without restarting the server. An experiment registry groups related jobs into named variations for A/B testing. Submit the same creative across all variations in one call, then compare acceptance rates, scores, and attempt counts side by side.

---

## What I didn't implement

- **Dynamic tool selection** — everything goes through one image editing API. A real system would route different recommendation types to different tools (inpainting for text changes, style transfer for colour work, etc.)
- **Confidence-based branching** — all variants follow the same pipeline regardless of the critic's confidence level
- **Cross-job learning** — within a job, failed ideas don't get retried. But there's no learning across jobs (e.g., "letter spacing edits never work on this image style")

---

## Project layout

```
backend/
  app/                    # FastAPI app + agents + services
  static/                 # HTML/JS frontend + sample images
  tests/                  # 76 tests
  monitoring/             # Prometheus, Grafana, pgAdmin config
ai_engineer_assignment_2026/  # Assignment PDF + sample data
```
