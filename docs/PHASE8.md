# Phase 8 — Production Backend

**Status: Phase 8 PASSED.** The real end-to-end smoke
(`scripts/smoke_phase8.py`, Phase 7 Q4_K_M + LoRA runtime) exited 0. Both real
model queries completed model generation -> safety -> read-only execution
(`SELECT COUNT(emp_id) FROM employees` -> 12; the Engineering salary join ->
625000.0). Answer correctness on the demo questions remains informational, not
a benchmark.

## Goal

```
registered database -> schema introspection -> canonical SchemaForge prompt
 -> model SQL generation -> deterministic SQL safety -> read-only execution
 -> structured result -> FastAPI / CLI
```

The model generates SQL text only. Deterministic software owns database
access, safety, execution limits, and errors.

## Architecture (`src/localsql/backend/`)

| Module | Responsibility |
|---|---|
| `registry.py` — `DatabaseRegistry` | Logical id -> vetted path inside one allowed root. |
| `introspection.py` — `SchemaIntrospector` / `SQLiteSchemaIntrospector` | Tables, columns/types, PKs (incl. composite), FKs -> the repo's `DatabaseSchema`. |
| `runtime.py` — `ModelRuntime` / `LlamaCppRuntime` | Prompt in, raw completion out. llama.cpp hot-LoRA via Phase 7 `run_gguf_once`. |
| `safety.py` — `SQLSafetyPolicy` | sqlglot AST allowlist; structured `SafetyDecision`. |
| `executor.py` — `SQLiteReadOnlyExecutor` | Independent read-only, time- and row-bounded execution. |
| `service.py` — `QueryService` | The one pipeline. No FastAPI import. |
| `api.py` | Thin FastAPI adapter (DI, request ids, safe errors). |
| `bootstrap.py`, `config.py` | Composition root; `configs/backend.yaml` + env. |
| `observability.py` | JSON-line structured logging. |

`QueryService` flow: resolve DB -> introspect -> `serialize_schema` +
`build_prompt` (the **same** canonical builder used in training, evaluation
and Phase 7; no serving-only template) -> `ModelRuntime.generate` ->
`normalize_predicted_sql` (whitespace trim only, never repaired) -> safety ->
executor -> `QueryResponse`.

Extension points: a persistent/cloud model server is a new `ModelRuntime`;
PostgreSQL is a new `DialectBackend(introspector, executor)` plus registry
support. Neither requires changing `QueryService`. PostgreSQL is intentionally
**not** implemented (the registry rejects any dialect but `sqlite`).

## Trust boundary

| Untrusted | Trusted (deterministic) |
|---|---|
| The user's `question` / `business_context` | Registry (config-defined ids and paths) |
| The model's output (SQL text) — including anything a prompt injection coaxes out | Safety policy, executor, limits, logging |

Callers never supply paths or connection strings; `QueryRequest` has no such
field (`extra="forbid"`). The prompt is *not* a security control: a malicious
question may make the model emit anything, so all protection is downstream of
the model.

## Security controls (defense in depth)

1. **Registry / path security** — ids match `[A-Za-z0-9][A-Za-z0-9_-]{0,63}`;
   config paths must be relative, without `..`, drive letters, or absolute
   roots; every `resolve()` re-resolves symlinks and requires the result to be
   inside the allowed root; unknown ids give a generic error; paths never
   appear in responses.
2. **Safety policy (AST)** — exactly one statement; root must be
   SELECT / WITH…SELECT / UNION-INTERSECT-EXCEPT of selects; the **entire**
   tree is walked so a mutation nested in a CTE or subquery is rejected.
   Rejected: INSERT/UPDATE/DELETE/REPLACE/MERGE, DDL, ATTACH/DETACH, PRAGMA
   (and `pragma_*` table functions), transactions/savepoints, VACUUM/REINDEX/
   ANALYZE/EXPLAIN, `SELECT … INTO`, bound parameters, `load_extension`,
   `readfile`/`writefile`/`edit`/`fts3_tokenizer`, NUL bytes, over-long SQL,
   anything sqlglot cannot parse as a plain query. Markdown fences are not
   repaired — they are rejected.
3. **Independent read-only executor** (holds even if layer 2 is bypassed; see
   `test_db_cannot_be_mutated_even_if_ast_safety_is_bypassed`):
   SQLite `mode=ro` URI; `PRAGMA query_only=ON`; extension loading disabled;
   an **authorizer** that denies everything except SELECT/READ/safe FUNCTION/
   RECURSIVE (so PRAGMA, ATTACH, DML, DDL, transactions are refused by SQLite
   itself); a wall-clock **progress-handler timeout**; a per-value
   `SQLITE_LIMIT_LENGTH` cap; single-statement execution; `max_rows + 1`
   fetch for deterministic truncation; the connection is always closed.
4. **Safe errors** — typed `BackendError`s with generic messages; the API
   validation handler reports field names/issue types only (never input
   values); unhandled exceptions become a generic 500; model stderr is never
   forwarded.
5. **Observability without leakage** — one JSON line per query
   (`request_id`, `database_id`, `stage`, `status`, error/safety codes,
   latency). Questions, SQL, prompts, and result rows are never logged
   (enforced by a key denylist and a test).
6. **Serialized model use** — one llama.cpp process at a time (multi-GB each).

Known parser-differential note: safety uses sqlglot, executes the *original*
SQL string, and SQLite may parse edge cases differently. That is exactly why
the executor layer exists.

## API

```
GET  /health      -> {status, model_runtime}   (works without a model configured)
GET  /databases   -> {databases:[{id, dialect, description}]}
POST /query       {database_id, question, business_context?} -> QueryResponse
```

`QueryResponse`: `request_id, database_id, status, generated_sql, safety
{allowed, reasons[]}, result {columns, rows, returned_row_count, truncated,
max_rows, elapsed_ms}, error {stage, code, message}, timings {schema_ms,
model_ms, safety_ms, execution_ms, total_ms}, model {…runtime metadata…},
prompt_sha256, dialect`. No total row count is reported (none is known):
`truncated=true` means at least one more row existed beyond `max_rows`.

| `status` | HTTP | Meaning |
|---|---|---|
| `ok` | 200 | generated, safe, executed |
| `unsafe_sql` | 422 | safety rejected; nothing executed |
| `execution_error` | 422 (504 on `timeout`) | SQL failed at execution |
| `model_error` | 502 (503 `model_not_configured`) | runtime failed / not configured |
| `schema_error` | 500 | schema could not be read |
| — | 404 / 503 | unknown / unavailable database id |
| — | 422 `invalid_request` | validation failure (no input echoed) |

`X-Request-ID` is accepted (`[A-Za-z0-9_.-]{8,64}`) or generated and returned.

Run: `uv run uvicorn localsql.backend.api:create_app_from_env --factory`

## CLI

```powershell
uv run python scripts/query_local.py --database-id demo --question "How many employees are there?"
```

Same `QueryService`; exit 0 only for `status == ok`. `--list-databases` lists ids.

## Configuration

`configs/backend.yaml` (databases + root, execution limits, runtime kind,
logging). Machine-specific values come from the environment and are never
committed:

| Variable | Meaning |
|---|---|
| `SCHEMAFORGE_LLAMA_EXE` | `llama-completion` executable |
| `SCHEMAFORGE_BASE_GGUF` | Q4_K_M base GGUF |
| `SCHEMAFORGE_LORA_GGUF` | LoRA GGUF (`--lora`) |
| `SCHEMAFORGE_LLAMA_THREADS`, `SCHEMAFORGE_LLAMA_NGL` | optional |
| `SCHEMAFORGE_DB_ROOT`, `SCHEMAFORGE_BACKEND_CONFIG` | optional overrides |

Inference settings (context 8192, 512 new tokens, seed 42, greedy) come from
`configs/phase7.yaml`, so serving matches what Phase 7 benchmarked.

## Limitations

- **Latency**: the Phase 7 runtime starts a llama.cpp process per request and
  reloads the model (~20 s on the target CPU). A persistent server runtime is
  the intended next step (new `ModelRuntime`; service unchanged).
- SQLite only. PostgreSQL is designed-for, not implemented. Views are not
  introspected. Schema is re-introspected per request (cheap for SQLite).
- No auth, rate limiting, or multi-tenant isolation (out of scope) (nor an HTTP body-size cap beyond field length limits; put a reverse proxy in front for that).
- Model quality on the quantized deployment is **not** re-evaluated here;
  Phase 7 measured that Q4_K_M changes outputs materially vs F16. The safety
  layer guarantees *safety*, not *correctness*, of generated SQL.
- No autonomous repair/retry: a failed or unsafe generation is reported as is.
- `sqlite_master` remains readable (it only exposes the schema the model is
  already given).

## Integration finding: llama.cpp prompt-file trailing newline

The first real smoke run exposed one integration failure: a generation of
`: SELECT COUNT(emp_id) FROM employees`, correctly rejected by the safety layer
as `parse_error`.

- **Root cause:** llama.cpp's `-f` prompt-file option strips the final newline,
  so the ChatML assistant prefix `<|im_start|>assistant<newline>` reached the model as
  `...assistant` (measured on b10964: 9 vs 10 tokens). This created train/serve
  prompt drift, and the model's first output token became unstable (leading
  newline, space, or `:`).
- **Fix (at the source):** `LlamaSettings.guard_prompt_trailing_newline`
  writes one extra newline so exactly `assistant<newline>` is tokenized. The
  production backend now enables `guard_prompt_trailing_newline=True`; the
  Phase 7 default remains `False` (see `docs/PHASE7.md`).
- **Not done, deliberately:** no SQL label stripping (`:` / `SQL:`) and no
  permissive normalization were added; `normalize_predicted_sql` is still
  whitespace-trim only. `SQLSafetyPolicy` is unchanged and strict: label
  artifacts, e.g. `: SELECT ...` or `SQL: DROP ...`, are still rejected
  (regression-tested).
- **Verification:** after the fix the real smoke passed with both queries
  executing.

## Real smoke (`scripts/smoke_phase8.py`)

Creates the deterministic demo DB at `.artifacts/phase8/databases/demo.sqlite`
(departments, employees with an FK, project_assignments with a composite PK),
then gates on:

1. schema introspection (tables, composite PK, FK);
2. executor refuses a DELETE and the DB is unchanged; safety rejects DROP;
3. two real questions (one with business context) each pass model generation
   -> safety -> execution with the Phase 7 Q4_K_M + LoRA runtime.

Answer correctness on the demo questions is printed as *informational*, not
gating. A JSON report is written to `.artifacts/phase8/smoke_report.json`.

## Phase 8 PASS criteria (met)

- `uv run pytest -q` passes (backend tests use real temporary SQLite DBs; only
  the model runtime is faked).
- `scripts/smoke_phase8.py` exits 0 on the real Q4_K_M + LoRA runtime (done).
- No claim of SQL correctness beyond what the smoke and later evaluation show.
