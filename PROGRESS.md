# Neurons AI Engineer Assignment — Progress Tracker

> Last updated: 2026-03-24

---

## Required — Backend Service

| # | Task | Status | Verified |
|---|------|--------|----------|
| 1 | RESTful API with FastAPI (async job lifecycle) | Done | |
| 2 | POST /api/v1/jobs — submit image + recommendations + brand guidelines | Done | |
| 3 | GET /api/v1/jobs/{job_id} — poll status & results | Done | |
| 4 | GET /api/v1/jobs/{job_id}/image/{rec_id} — fetch edited image | Done | |
| 5 | GET /health — health check | Done | |
| 6 | Async background processing (submit → poll pattern) | Done | |
| 7 | Concurrent recommendation processing (asyncio.gather) | Done | |
| 8 | Error handling (invalid file types, failed processing, max retries) | Done | |
| 9 | Status reporting through API (pending/running/completed/failed) | Done | |
| 10 | Proper logging | Done | |
| 11 | Dockerised with docker-compose | Done | |

## Required — Multi-Agent Workflow (LangGraph)

| # | Task | Status | Verified |
|---|------|--------|----------|
| 12 | LangGraph StateGraph with conditional edges | Done | |
| 13 | Planner Agent — interprets recommendation, produces plan + edit prompt | Done | |
| 14 | Editor Agent — generates edited image via gpt-image-1 | Done | |
| 15 | Critic Agent — evaluates output (recommendation + brand compliance) | Done | |
| 16 | Refiner Agent — translates critic feedback into planner guidance | Done | |
| 17 | Iteration loop (plan→edit→evaluate→retry up to MAX_RETRIES) | Done | |
| 18 | Audit trail — timestamped log of every agent decision | Done | |

## Required — Deliverables

| # | Task | Status | Verified |
|---|------|--------|----------|
| 19 | Source code with clear documentation | Done | |
| 20 | README with setup instructions | Done | |
| 21 | Code walkthrough doc (WALKTHROUGH.md) | Done | |
| 22 | .env.example for config | Done | |

## Optional — Unit Tests

| # | Task | Status | Verified |
|---|------|--------|----------|
| 23 | Health check test | Done | |
| 24 | Invalid image type test | Done | |
| 25 | Invalid JSON body test | Done | |
| 26 | Job not found test | Done | |
| 27 | Create and poll happy path test | Done | |
| 28 | Empty recommendations test | Done | |
| 29 | Agent/workflow unit tests | Not started | |

## Optional — Frontend App (Streamlit/Gradio)

| # | Task | Status | Verified |
|---|------|--------|----------|
| 30 | Drag-and-drop image upload | Not started | |
| 31 | Input recommendations + brand guidelines | Not started | |
| 32 | Display edited variants alongside original | Not started | |
| 33 | Visualise audit trail | Not started | |
| 34 | Loading/processing state (async demo) | Not started | |
| 35 | Add frontend to docker-compose | Not started | |

## Optional — Additional Enhancements

| # | Task | Status | Verified |
|---|------|--------|----------|
| 36 | Tool selection — dynamic tool choice based on rec type (segmentation, inpainting, style transfer) | Not started | |
| 37 | Multiple variants in parallel, ranked by score | Not started | |
| 38 | Confidence-based branching (different strategies by confidence) | Not started | |
| 39 | Memory across iterations (avoid repeating failed approaches) | Not started | |
| 40 | Human-in-the-loop feedback (accept/reject/refine) | Not started | |

## Blockers / Notes

- Job store is in-memory — jobs lost on restart (acceptable for assignment, noted as limitation)
- Image size constrained to IMAGE_SIZE config — original aspect ratio may not be perfectly preserved
- Need to verify end-to-end run with real OpenAI API key
- Need to verify Docker build and compose up work cleanly

---

## How to use this tracker

- **Status**: `Not started` | `In progress` | `Done` | `Blocked`
- **Verified**: Leave blank until user has personally confirmed the feature works. Mark with date when verified.
