# Running Experiments

Step-by-step guide to comparing different model configurations using the experiment system.

---

## 1. Make sure the system is running

```bash
docker-compose up --build -d
```

Check everything is healthy:

```bash
curl http://localhost:8000/health
```

You should see `{"status":"ok","database":"connected"}`.

---

## 2. Create an experiment

An experiment defines the configurations you want to compare. Each "variation" is a different config that will run against the same input.

```bash
curl -X POST http://localhost:8000/api/v1/experiments \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Gemini vs OpenAI text evaluation",
    "description": "Same image and recommendations, different text providers",
    "variations": [
      {
        "name": "gemini-text",
        "config": { "text_provider": "gemini" }
      },
      {
        "name": "openai-text",
        "config": { "text_provider": "openai" }
      }
    ]
  }'
```

This returns an `experiment_id`. Save it — you'll need it for the next steps.

```json
{
  "experiment_id": "abc123...",
  "name": "Gemini vs OpenAI text evaluation",
  "variations": 2
}
```

### What can you vary?

| Field | Values | What it changes |
|---|---|---|
| `text_provider` | `gemini`, `openai`, `claude` | Which LLM evaluates ideas, critiques images, refines prompts |
| `image_provider` | `gemini`, `openai` | Which model generates the edited images |
| `text_model` | any model ID | Specific model (e.g. `gpt-4o`, `gemini-2.5-flash`) |
| `image_model` | any model ID | Specific image model |
| `num_variants` | 1, 2, 3... | How many accepted variants to target per recommendation |
| `max_retries` | 1, 2, 3... | How many edit-evaluate-refine cycles before giving up |

You can combine these. For example, to test whether more retries help:

```json
{
  "name": "Retry budget comparison",
  "variations": [
    { "name": "1-retry", "config": { "max_retries": 1 } },
    { "name": "3-retries", "config": { "max_retries": 3 } },
    { "name": "5-retries", "config": { "max_retries": 5 } }
  ]
}
```

---

## 3. Run the experiment

Submit an image and recommendations. The system creates one job per variation, all using the same input.

```bash
curl -X POST http://localhost:8000/api/v1/experiments/EXPERIMENT_ID/run \
  -F "image=@backend/static/samples/creative_1.png" \
  -F "request_body=@-" << 'EOF'
{
  "recommendations": [
    {
      "id": "rec_1",
      "title": "Strengthen Headline Impact",
      "description": "Add visual punch to the headline through enhanced color contrast, a soft gradient backdrop, or a geometric shape.",
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
EOF
```

This returns a list of jobs, one per variation:

```json
{
  "experiment_id": "abc123...",
  "jobs": [
    { "job_id": "job_aaa...", "variation": "gemini-text" },
    { "job_id": "job_bbb...", "variation": "openai-text" }
  ]
}
```

All jobs run concurrently. For meaningful results, run the same experiment multiple times (LLM output is non-deterministic). Just call the `/run` endpoint again with the same input — the jobs will be added to the same experiment.

---

## 4. Wait for jobs to finish

Poll any individual job to check progress:

```bash
curl http://localhost:8000/api/v1/jobs/JOB_ID | python3 -m json.tool
```

Or watch the logs in real time:

```bash
docker-compose logs -f backend 2>&1 | grep -E "variant_completed|job_completed"
```

---

## 5. Compare results

Once jobs are done, pull the experiment results:

```bash
curl http://localhost:8000/api/v1/experiments/EXPERIMENT_ID/results | python3 -m json.tool
```

This returns per-variation aggregates:

```json
{
  "experiment_id": "abc123...",
  "name": "Gemini vs OpenAI text evaluation",
  "status": "completed",
  "variations": [
    {
      "name": "gemini-text",
      "jobs": 1,
      "variants_total": 4,
      "variants_accepted": 3,
      "acceptance_rate": 0.75,
      "avg_score": 0.82,
      "avg_attempts": 1.5,
      "avg_duration_s": 45.2,
      "total_tokens": 12400,
      "total_cost_usd": 0.0052,
      "prompt_versions": { "ideator": "a3f8b2...", "critic": "7b2e9f...", "..." : "..." }
    },
    {
      "name": "openai-text",
      "jobs": 1,
      "variants_total": 4,
      "variants_accepted": 2,
      "acceptance_rate": 0.5,
      "avg_score": 0.71,
      "avg_attempts": 2.1,
      "avg_duration_s": 38.7,
      "total_tokens": 18200,
      "total_cost_usd": 0.0891,
      "prompt_versions": { "ideator": "a3f8b2...", "critic": "7b2e9f...", "..." : "..." }
    }
  ]
}
```

### What to look at

| Metric | What it tells you |
|---|---|
| `acceptance_rate` | What fraction of variants passed the critic. Higher = better quality edits. |
| `avg_score` | Average critic evaluation score (0-1). |
| `avg_attempts` | How many edit-evaluate-refine cycles per variant. Lower = less wasted work. |
| `avg_duration_s` | Average wall-clock time per variant. |
| `total_tokens` | Total LLM tokens consumed across all variants. |
| `total_cost_usd` | Estimated API cost. |
| `prompt_versions` | Hash of each agent's prompt. Same hashes = same prompts. If you change a prompt between runs, the hash changes. |

---

## 6. Dig into the details

### Per-variant breakdown

Each variant in a job response includes detailed tracking:

```bash
curl http://localhost:8000/api/v1/jobs/JOB_ID \
  | python3 -c "
import sys, json
data = json.load(sys.stdin)
for rec in data['results']:
    for v in rec['variants']:
        print(f\"{v['variant_id']}: status={v['status']}, score={v.get('evaluation_score')}, duration={v.get('duration_s')}s, tokens={v.get('total_tokens')}, cost=\${v.get('total_cost_usd', 0):.4f}\")
"
```

### Agent-level timing

See how long each agent took within a variant:

```bash
curl http://localhost:8000/api/v1/jobs/JOB_ID \
  | python3 -c "
import sys, json
data = json.load(sys.stdin)
for rec in data['results']:
    for v in rec['variants']:
        print(f\"--- {v['variant_id']} ---\")
        for t in v.get('agent_timings', []):
            print(f\"  {t['agent']}: {t['duration_s']}s, {t['tokens']} tokens, \${t['cost_usd']:.4f}\")
"
```

### Critic evaluation history

See every evaluation attempt (not just the final one):

```bash
curl http://localhost:8000/api/v1/jobs/JOB_ID \
  | python3 -c "
import sys, json
data = json.load(sys.stdin)
for rec in data['results']:
    for v in rec['variants']:
        print(f\"--- {v['variant_id']} ---\")
        for e in v.get('critic_evaluations', []):
            print(f\"  attempt {e['attempt']}: passed={e['passed']}, score={e['score']}, ssim={e.get('ssim_score', 'N/A')}\")
"
```

---

## 7. View logs in Grafana

Open the Logs Explorer dashboard at http://localhost:3000/d/visrec-logs to see structured log panels for agent performance, critic evaluations, token usage, and more.

To filter logs for a specific job, go to Explore (http://localhost:3000/explore), switch the datasource to Loki, and query:

```
{service="backend"} | json | job_id = "YOUR_JOB_ID"
```

---

## Tips

- **Change one variable at a time.** If you change both the text provider and max_retries, you won't know which one caused the difference.
- **Run multiple times.** LLM output is non-deterministic. A single run tells you very little. Run each variation 3-5 times with the same input.
- **Use the same test input.** The two sample creatives with their 3 recommendations each are a good baseline. Consistent input makes results comparable.
- **Check prompt_versions.** If you change an agent's prompt between experiment runs, the hash will change. This tells you whether you're comparing the same prompts or different ones.
- **Cost adds up.** Each variant costs roughly $0.01-0.10 depending on the provider. An experiment with 3 variations, 3 recommendations, and 5 runs = 45+ jobs. Check `total_cost_usd` in the results.
