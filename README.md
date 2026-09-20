# SchemaForge

**A fine-tuned, open-weight Text-to-SQL model — taken from training data to a deployed, safety-checked product.**

SchemaForge specializes `Qwen/Qwen3-4B-Instruct-2507` with 4-bit QLoRA to turn a natural-language
question, a database schema and optional business context into **one read-only SQL query**, then
serves it through a production-style stack: quantized GGUF weights on a persistent llama.cpp
server, deterministic SQL safety, and a web frontend — deployed on AWS EC2 and Vercel and verified
from a phone on LTE. *(Public name: SchemaForge. The Python package and some internal docs use the
legacy name `localsql`.)*

```text
DATA -> TRAINING -> EVALUATION -> QUANTIZATION -> SERVING -> SAFETY -> APPLICATION -> CLOUD DEPLOYMENT
```

<p align="center">
  <img src="docs/assets/phase11/mobile-query-result-625000.png" alt="SchemaForge on an iPhone over LTE: generated SQL, safety and read-only badges, result 625,000" width="280">
</p>

<sub>Real screenshot from the production proof (iPhone, LTE, Wi-Fi off). The single request took 16.0 s at 3.4 tok/s on 2 vCPU — an observation, **not a benchmark**.</sub>

## The question, and why Text-to-SQL

**Research question:** can a compact (~4B) open-weight model, specialized with QLoRA on a filtered
slice of BIRD, generate correct SQL for **database schemas it never saw in training**?

Text-to-SQL is the vehicle because it is a task with an *objective, executable* check: SQL can be
parsed, validated against a real schema, run on a real database and compared by result — so
"did fine-tuning help?" can be answered with measurements instead of impressions. This is an
**open-weight model-engineering project, not an API wrapper**: the weights were fine-tuned,
evaluated, quantized and served by this repository's own pipeline.

## Results (frozen)

Three separate evaluations — different data, metrics and evaluators; never combined.

| Evaluation | Metric | Base | Fine-tuned | Δ |
|---|---|---|---|---|
| **A.** Seen-training specialization (500 fixed training examples) | Normalized SQL exact match | 4.20% | 37.40% | +33.20 pp |
| **B.** Schema-held-out (534 examples, 7 unseen DBs) | Execution accuracy* | 39.14% | 44.57% | +5.43 pp |
| **C.** External BIRD Mini-Dev (500, untouched) | Execution accuracy (official) | 43.6% | 44.8% | +1.2 pp |

\*SchemaForge's SQLite execution comparator, not the official BIRD evaluator.

**Read honestly:** A is *specialization*, not generalization. B is the strongest internal
generalization evidence (61 examples gained vs 32 lost). C's +1.2 pp is real but **modest**, and
official **Soft-F1 did not improve** (47.70 -> 47.18); what changed most is output reliability
(parse success 0.942 -> 0.984, execution success 0.82 -> 0.87). No state-of-the-art claim is made,
and the deployed Q4_K_M model was not re-scored. Full tables, paired counts and caveats:
[`docs/RESULTS.md`](docs/RESULTS.md).

## What was built

| Stage | Highlights | Docs |
|---|---|---|
| **Data** | Deterministic BIRD pipeline, one canonical prompt/completion, seeded business-context dropout, explicit validation (nothing silently dropped) | [`DATA_CONTRACT`](docs/DATA_CONTRACT.md) |
| **Leakage-safe evaluation** | Splits by `db_id` with zero overlap asserted in code; Mini-Dev is evaluation-only; gold isolated from generation; official evaluator vendored unmodified | [`EVALUATION`](docs/EVALUATION.md), [`BASELINE`](docs/BASELINE.md) |
| **Training** | 4-bit NF4 QLoRA, LoRA r=16/alpha=32 on 7 projections, completion-only loss, adaptive schema compaction so nothing is truncated at 4096 tokens, 1,518 steps, resumable across Kaggle sessions | [`TRAINING`](docs/TRAINING.md), [`CONTEXT_BUDGET`](docs/CONTEXT_BUDGET.md) |
| **Quantization** | HF -> F16 GGUF -> **Q4_K_M** (2.33 GB) + F16 LoRA GGUF; ~2.39 GB effective; SHA-256 pinned | [`PHASE7`](docs/PHASE7.md) |
| **Serving** | Persistent llama.cpp server: model loaded once, private network, hot LoRA, CPU-first (GPU optional) | [`PHASE11`](docs/PHASE11.md) |
| **Safety** | The model only generates SQL; deterministic AST policy, schema-aware preflight, read-only SQLite executor, timeouts, row cap | [`PHASE8`](docs/PHASE8.md), [`PHASE10`](docs/PHASE10.md) |
| **Backend reliability** | Bounded concurrency, cancellation, generation timeout, rate/body limits, request IDs, readiness, error taxonomy | [`PHASE10`](docs/PHASE10.md) |
| **Frontend** | Vite + React + TypeScript SPA using only the public API — no SQL, DB or model logic in the browser | [`PHASE9`](docs/PHASE9.md) |
| **Cloud** | Vercel -> HTTPS -> AWS EC2 (Caddy) -> FastAPI -> private llama.cpp; validated from a phone on LTE | [`evidence`](docs/evidence/phase11-production-proof.md) |

## Architecture

```mermaid
flowchart LR
    B["Browser"] --> V["Vercel frontend"] -->|HTTPS| E["AWS EC2 / Caddy"] --> F["FastAPI"]
    F --> L["Persistent llama.cpp<br/>Qwen3-4B Q4_K_M + LoRA (CPU)"]
    L -->|SQL text| S["AST safety"] --> P["Schema-aware preflight"] --> X["Read-only SQLite executor"]
```

Deployed on AWS EC2 with persistent llama.cpp serving: the model runs on the server, **never in
the browser or on the visitor's device**. Full offline + online diagram and per-component
responsibilities: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Trust boundary: the model does not judge its own SQL

The LLM generates SQL. It does **not** decide whether that SQL is safe. Deterministic software
parses it, enforces a single read-only query (AST allow-list, DDL/DML rejected), checks it against
the real schema with an `EXPLAIN` preflight, and executes it on a `mode=ro`, `query_only`,
authorizer-guarded SQLite connection with a timeout and row cap.

**Safe != correct.** Every response reports `semantic_correctness = "not_verified"` and
`confidence = null`; a confidence score is intentionally not invented.

## Production engineering

Model loaded once (no per-query reload) · private, internal-only model-server network · FastAPI as
the single application boundary · Caddy automatic HTTPS · exact-origin CORS enforced at startup ·
`/health` and `/ready` (readiness never runs inference) · bounded concurrency with `model_busy`
back-pressure · cancellation and generation timeout · rate limiting and body limits · request IDs ·
API docs hidden in production · non-root, capability-dropped, read-only containers · SHA-256
verification of model artifacts at every start.

## Deployment proof and honest framing

Phase 11 proved the whole path on real infrastructure on 2026-09-19: an AWS EC2 `m7i-flex.large`
(2 vCPU, 8 GiB, CPU-only) serving the frozen model behind Caddy, a Vercel frontend, and a real
question answered on an iPhone over LTE (result `625000`). Evidence:
[`docs/evidence/phase11-production-proof.md`](docs/evidence/phase11-production-proof.md).

The proof instance was terminated after evidence capture to avoid cost, and the AWS backend is
**recreated for demonstrations** from the checked-in runbook; when it is not running, the static
frontend cannot answer queries. Deployment validation is kept separate from research results.

## Quickstart

```bash
uv sync && uv run pytest -q                       # no models or GPU needed
cd frontend && npm ci && npm test                 # frontend checks
```

Providing model artifacts, local serving, Docker Compose, recreating AWS and deploying Vercel:
[`docs/QUICKSTART.md`](docs/QUICKSTART.md). GGUF/model files are **not** in Git; they are
identified by SHA-256 ([`docs/TRAINING.md`](docs/TRAINING.md)). No secrets are required or stored.

## Limitations

External gain is modest (+1.2 pp EX) and Soft-F1 did not improve · CPU inference is slow
(one observed request: 16.0 s, 3.4 tok/s — not a benchmark) · one model slot, no HA or autoscaling ·
SQLite-first, no PostgreSQL/bring-your-own DB, no authentication · semantic correctness is not
verified and confidence is unclaimed · GPU serving path exists but was not exercised. Full list:
[`docs/LIMITATIONS.md`](docs/LIMITATIONS.md).

## Documentation map

| Start here | |
|---|---|
| [`docs/RESULTS.md`](docs/RESULTS.md) | Authoritative results |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Finished system architecture |
| [`docs/FINAL_CERTIFICATION.md`](docs/FINAL_CERTIFICATION.md) | Phase-by-phase evidence matrix |
| [`docs/PORTFOLIO.md`](docs/PORTFOLIO.md) | Project summaries and interview notes |
| [`docs/QUICKSTART.md`](docs/QUICKSTART.md) · [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) | Reproduce it · what it does not do |
| [`PROJECT.md`](PROJECT.md) | Chronological engineering journal |

Phase documents: [`DATA_CONTRACT`](docs/DATA_CONTRACT.md) · [`EVALUATION`](docs/EVALUATION.md) ·
[`BASELINE`](docs/BASELINE.md) · [`TRAINING`](docs/TRAINING.md) · [`CONTEXT_BUDGET`](docs/CONTEXT_BUDGET.md) ·
[`PHASE7`](docs/PHASE7.md) · [`PHASE8`](docs/PHASE8.md) · [`PHASE9`](docs/PHASE9.md) ·
[`PHASE10`](docs/PHASE10.md) · [`PHASE11`](docs/PHASE11.md).

## Repository layout

```text
src/localsql/     data/ schema_context/ benchmark/ model/ train/ deploy/ backend/   (package name is legacy `localsql`)
frontend/         Vite + React + TypeScript SPA (Vercel)
docker/ deploy/   Dockerfiles, Caddyfile, env template, EC2 scripts;  docker-compose*.yml
configs/          data / benchmark / model / train / phase7 / backend constants
scripts/          pipeline, evaluation, conversion, canary and smoke scripts
tests/            offline unit + fixture tests
docs/             documentation, evidence/, assets/
```
