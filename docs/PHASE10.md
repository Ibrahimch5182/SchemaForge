# Phase 10 — Reliability, Calibration and Product Hardening

**Status: implemented and automatically tested. Phase 10 PASS additionally requires the
real product reliability canary (`scripts/canary_phase10.py`) to exit 0.**

Phase 10 hardens the existing flow; it adds no features, no retraining and no model
claims.

```
request -> database/schema -> generation -> normalization -> safety policy
        -> database-aware preflight -> read-only execution -> reliability metadata -> API/UI
```

## 1. Safety is not correctness (the trust model)

| Stage | What passing means | What it does NOT mean |
|---|---|---|
| Model generation | The model produced text | That the text is SQL, or right |
| Safety policy (AST) | One read-only SELECT/WITH | That it answers the question |
| **Preflight** (new) | The SQL compiles against **this** database's real schema | That it answers the question |
| Read-only execution | It ran, within time/row limits | That the rows are the right answer |
| **Semantic correctness** | — | **Never verified.** No calibrated probability exists |

`QueryResponse.reliability` (additive, backward compatible) records exactly which
deterministic stages ran: `safety`, `preflight`, `execution` ∈ `passed | failed | not_run`,
`semantic_correctness` is **always** `"not_verified"`, and `confidence` is **always**
`null`. There is deliberately no confidence score: the project has no scientifically
calibrated signal for semantic correctness, and inventing one would be misleading.
The frontend parser refuses to let a payload upgrade `semantic_correctness` or supply
a confidence.

UI wording: **Safety verified**, **Read-only execution**, **Query executed
successfully**, plus a concise note — *Verified: safe, valid for this database,
read-only. Not verified: that this answers your question.* No percentages, no
"correct".

## 2. Database-aware preflight

`localsql.backend.preflight.SQLitePreflight` runs `EXPLAIN QUERY PLAN <sql>` on the same
hardened read-only connection (`mode=ro`, `query_only`, authorizer, progress-handler
deadline) between safety and execution. SQLite compiles the statement against the
real schema — resolving tables, columns and functions — **without running it**, so
there is no duplicate execution and no optimizer framework. A statement that would write
is refused by the authorizer here too.

Failures become structured codes (`status: "validation_error"`, `stage: "preflight"`):
`unknown_table`, `unknown_column`, `ambiguous_column`, `unknown_function`,
`invalid_syntax`, `invalid_sql`, `authorization_denied`. `error.detail` carries only a
short identifier (e.g. the missing column name); raw SQLite messages never leave the
module. Unsafe SQL never reaches preflight; invalid SQL never reaches execution.

## 3. Runtime reliability, concurrency and timeouts

`LlamaCppRuntime` (still process-per-request; the persistent server is Phase 11):

- **Bounded concurrency** — `InferenceGate`: at most `max_concurrent_generations` (default
  1) model processes run; at most `max_waiting_requests` (default 2) may queue, each for at
  most `queue_wait_seconds` (default 20). Beyond that, requests are rejected immediately as
  `model_busy` (HTTP 429, `Retry-After: 5`). No unbounded pile-up of multi-GB processes.
- **Timeouts** — the generation subprocess is killed at `runtime.timeout_seconds` (now 120)
  and reported as `model_timeout` (504). Execution has its own limit (`timeout`, 504).
  Ordering is coherent end to end: SQLite `execution.timeout_seconds` (10) <
  generation 120 (+ ≤20 queue) < frontend `VITE_QUERY_TIMEOUT_MS` (180), so the backend gives
  up before the browser does.
- **Abnormal exit / crash** — a non-zero exit becomes `model_error` with a generic message;
  exit code and a stderr tail go only to the structured log (`runtime.failure`).
- **Empty / malformed output** — `malformed_model_output` (see taxonomy).
- **Cleanup** — the process is killed on timeout, cancel, or any unexpected exception inside
  the runner; the prompt temp directory is always removed.
- **No per-request hashing** of multi-GB artifacts.

## 4. Cancellation

Cooperative and thread-based (the sync pipeline runs in a worker thread; no async rewrite):

- `CancelToken` is created per request and registered by request id in `QueryService`.
- `POST /query/{request_id}/cancel` (idempotent, always 200, `cancelled: false` for
  unknown/finished ids) sets the token. It is honored while waiting in the gate, by
  killing the llama.cpp process, by interrupting SQLite (progress handler) in preflight and
  execution, and between stages. The request then completes as `status: "cancelled"` (409).
- The frontend sends its own `X-Request-ID` and, when the user cancels **or** the client
  timeout fires, calls the cancel endpoint (fire-and-forget), so a browser-side give-up does
  not leave a model process running for nobody.
- Not covered: a tab closed without cancelling. The generation and execution time limits
  bound that case.

## 5. Health and readiness

- `GET /health` (liveness + info; always 200): `model_runtime` gains `ready`, `checks`
  (`executable` / `base_gguf` / `lora_gguf` file presence — three `stat` calls, no model
  load, no hashing, no paths) and `availability` (`idle | busy | saturated`, running/waiting
  counts and limits).
- `GET /ready`: `200 {"ready": true}` only when a runtime is configured with its artifacts
  present and a database is registered; otherwise `503` with `reasons`
  (`model_not_configured`, `model_artifacts_missing`, `no_databases_registered`). Never runs
  inference.
- The frontend status pill shows *Local model ready*, *Model busy/saturated*, *Model files
  missing*, *Model not configured* or *Backend offline*, and blocks **Run** when files are
  missing.

## 6. Error taxonomy

`status` says which stage failed; `error.code` says why. One table
(`localsql/backend/taxonomy.py`) drives HTTP status; the frontend presentation
(`frontend/src/lib/failure.ts`) follows it. Established names were kept.

| `status` | `error.code` | HTTP | Meaning |
|---|---|---|---|
| `ok` | — | 200 | ran; correctness not verified |
| `model_error` | `model_not_configured` | 503 | no runtime attached |
| `model_error` | `model_busy` | 429 | gate full (retry) |
| `model_error` | `model_timeout` | 504 | generation limit hit |
| `model_error` | `model_error` | 502 | runtime failed / crashed |
| `model_error` | `malformed_model_output` | 502 | empty or not a SQL query |
| `model_error` | `internal_error` | 502 | unexpected; generic message |
| `unsafe_sql` | `not_read_only_query`, `multiple_statements`, `denied_function`, `mutation`, `ddl`, `attach_detach`, `pragma`, `transaction`, `session_command`, `select_into`, `parameters` | 422 | safety policy rejection |
| `validation_error` | `unknown_table`, `unknown_column`, `ambiguous_column`, `unknown_function`, `invalid_syntax`, `invalid_sql`, `authorization_denied` | 422 | invalid for this database |
| `execution_error` | `timeout` | 504 | execution time limit |
| `execution_error` | `sql_error`, `authorization_denied`, `database_unavailable`, `internal_error` | 422 / 503 | execution failure |
| `schema_error` | `schema_error`, `internal_error` | 500 | schema unreadable |
| `cancelled` | `cancelled` | 409 | cancelled; work stopped |

Non-pipeline errors use the JSON envelope: `unknown_database` (404),
`database_unavailable` (503), `invalid_request` (422), `not_found` (404),
`internal_error` (500); Phase 11 adds the public-demo protections `rate_limited` (429, with
`Retry-After`) and `payload_too_large` (413). "Malformed" safety codes (`empty_sql`, `parse_error`,
`invalid_characters`, `too_long`) are reclassified from `unsafe_sql` to
`malformed_model_output`: garbage output is a model problem, not a policy violation
(so the earlier `: SELECT …` artifact now reads correctly). A test keeps this document in
sync with the taxonomy. Responses never contain stack traces, machine paths, secrets or
subprocess output.

## 7. Observability

One JSON line per event, always with `request_id`; never result rows, questions, SQL,
prompts or secrets (key denylist + tests). Stage latencies (`schema`, `model`, `safety`,
`preflight`, `execution`, `total`) and the `reliability` stage states ride on
`query.completed`. Specific events: `query.model_busy`, `query.model_timeout`,
`query.model_failure`, `query.malformed_output`, `query.safety_rejected`,
`query.preflight_rejected`, `query.execution_timeout`, `query.execution_failure`,
`query.cancelled`, `runtime.timeout`, `runtime.failure` (exit code + stderr tail,
server-side only). No monitoring platform was added.

## 8. Product reliability canary

```powershell
uv run python scripts/canary_phase10.py
```

**PRODUCT RELIABILITY CANARY — NOT A MODEL BENCHMARK.** It measures the product's failure
handling, not how often the model is right.

- **Part A (no model):** scripted runtimes drive malformed/empty output, unsafe SQL,
  unknown table/column, runaway-query timeout, model busy/timeout/error, cancellation,
  bounded concurrency, and a *real* crashing subprocess through the real registry,
  safety, preflight and executor; checks the taxonomy, no leakage, and that the database
  file is byte-identical afterwards.
- **Part B (real Q4_K_M + LoRA, 3 generations):** readiness without inference; three
  representative queries must complete the pipeline with honest trust metadata
  (`not_verified`, no confidence, `passed/passed/passed`); two easy answers (count = 12,
  Engineering total = 625000.0) must match; the group-by answer is reported but not gating;
  no inference slots leak; the database is unchanged.
- Report: `.artifacts/phase10/canary_report.json`. Non-zero exit if the gate fails.

## 9. Configuration

`configs/backend.yaml → runtime`: `timeout_seconds: 120`, `max_concurrent_generations: 1`,
`max_waiting_requests: 2`, `queue_wait_seconds: 20`. Frontend: `VITE_QUERY_TIMEOUT_MS`
(default 180000).

## 10. Limitations

- Process-per-request inference still reloads the model each time (Phase 11); the gate
  bounds it but does not remove the ~10 s cost, and `model_busy` is expected under load.
- Preflight validates against the schema; it cannot detect semantically wrong but valid SQL
  (wrong join, wrong filter). That is exactly why `semantic_correctness` stays
  `not_verified`.
- Cancellation needs the client to call the cancel endpoint; closing a tab silently relies on
  the time limits. Cancelling during the tiny `EXPLAIN` phase may finish as a validation
  result instead.
- Multiple backend workers would each have their own gate (single-process assumption).
- The canary is a reliability gate on a tiny synthetic fixture, not a quality measurement.

## Phase 10 PASS criteria

- Reliability states are explicit and stable (taxonomy above; docs/test in sync).
- Safety success is never presented as semantic correctness (`reliability` + UI wording).
- DB-aware preflight works and precedes execution.
- Runtime failure/timeout behavior is controlled; inference concurrency is bounded.
- `/health` and `/ready` are useful without running inference.
- Cancellation/cleanup work end to end where practical.
- Frontend represents the states; frontend checks and `uv run pytest -q` pass.
- `scripts/canary_phase10.py` exits 0 on the real runtime.
