# Phase 9 — Product Frontend

**Status: PASSED (tag `phase-9-pass`); later deployed on Vercel in Phase 11. Phase-9-era statements
below about local/reload behavior are historical; the current deployment is described in
[`ARCHITECTURE.md`](ARCHITECTURE.md).**

## Goal

The user-facing SchemaForge product: an impressive landing page that states what
the project is and shows its *measured* results, and a polished query workspace
that drives the Phase 8 backend end to end. The frontend contains **no** database
access, schema introspection, SQL safety, SQL execution or model logic — it only
calls the Phase 8 API.

## Architecture (`frontend/`)

Standalone TypeScript SPA: **Vite + React 19**, hand-written CSS (design tokens, no
UI framework), self-hosted fonts (`@fontsource-variable/*`, so it works offline),
no charting or syntax-highlighting library (both are small, purpose-built,
dependency-free modules). Kept separate from the Python package; nothing in
`src/localsql` imports it.

```
frontend/src/
  api/        types.ts (wire contract) · guards.ts (runtime validation) ·
              errors.ts (ApiError) · client.ts (the ONLY fetch code) · config.ts (env)
  state/      api-context (DI) · useHealth · useDatabases · useQueryRunner · history
  lib/        failure (error taxonomy) · chartability · sqlHighlight · format ·
              modelMeta · clipboard · route · hooks
  components/ StatusPill · DatabasePicker · QueryForm · LoadingPanel · SqlBlock ·
              ResultView · ResultTable · ChartPanel · MetaStrip · FailurePanel ·
              HistoryList · ForgeDemo · SiteHeader · icons
  pages/      Landing · Workspace
  data/       results.ts (frozen benchmark numbers — the single source)
  styles/     tokens · base · landing · workspace
```

Boundaries: components never call `fetch`; they receive a `SchemaForgeApi` from
context (`ApiProvider`), which is how tests inject mocks. State hooks own
in-flight requests (real `AbortController` cancellation). Routing is a ~40-line
History-API router (two routes do not justify a dependency).

## API integration

Uses the Phase 8 contract **exactly** (`docs/PHASE8.md`): `GET /health`,
`GET /databases`, `POST /query {database_id, question, business_context?}`. The
frontend never sends or displays filesystem paths and offers no way to submit SQL.

- **Config:** `VITE_API_BASE_URL` (default `http://127.0.0.1:8000`) and
  `VITE_QUERY_TIMEOUT_MS` (default 180 000); see `frontend/.env.example`.
- **Pipeline outcomes are results, not transport errors.** The backend returns
  `unsafe_sql` / `execution_error` / `model_error` / `schema_error` as a full
  `QueryResponse` body with a non-2xx status; the client parses those and the UI
  renders them as designed states (with the blocked/failed SQL shown).
- **Structured errors:** the `{request_id, error:{code,message,fields?}}` envelope
  becomes an `ApiError` (`unknown_database`, `database_unavailable`,
  `invalid_request`, `internal_error`, …).
- **Defensive parsing:** every payload is validated at the boundary
  (`guards.ts`); off-contract data becomes a `malformed` error, and non-primitive
  cells are stringified so an object can never reach the table. Non-JSON error
  bodies (proxy pages) are handled.
- **Network / timeout / cancel** are distinct: `network`, `timeout` (client
  wait elapsed), `aborted` (user cancelled).
- **Health** is checked on mount, on demand ("Re-check"), on window focus only if
  the last check is >60 s old, and after a network failure. No interval polling.
- **CORS:** the one small backend change. `create_app(..., cors_origins=...)`
  adds an explicit allow-list (`configs/backend.yaml` → `api.cors_allowed_origins`,
  override `SCHEMAFORGE_CORS_ORIGINS`), GET/POST only, no credentials, exposing
  `X-Request-ID`. Default: `http://localhost:5173` and `http://127.0.0.1:5173`.
  Without origins configured, no CORS headers are emitted (Phase 8 behavior).

## Security posture

- Generated SQL and result cells are rendered strictly as text nodes; ESLint forbids
  `dangerouslySetInnerHTML`. Tests assert markup in SQL/cells stays inert.
- The SQL highlighter is a lossless tokenizer producing `<span>` text children; it
  never parses or executes SQL.
- Backend messages for opaque failures are replaced by product copy; only the
  (safe) rejection reasons and SQLite error text from Phase 8 are shown.
- History lives in `sessionStorage` only (per tab, capped at 15, corrupt entries
  dropped, quota errors tolerated). No accounts, no server-side history.

## UI states

| Situation | Presentation |
|---|---|
| Backend/model status | Header pill: *<host> model server ready* (Phase 11 wording; originally *Local model ready*) / *Model not configured* / *Backend offline* / checking, with a details popover (runtime, deployment, GGUF names, context) |
| Databases loading / empty / unreachable | Skeleton · empty state · designed error with retry |
| Running | Indeterminate animation + **real elapsed clock**; explicit "no live progress signal" copy (never a fake percentage); Cancel |
| Success | Banner, SQL artifact (highlight, copy, *Safety passed* / *Read-only* badges), result table, run summary, runtime details disclosure |
| `unsafe_sql` | Danger card with rejection reasons and the **blocked, never-executed** SQL |
| `execution_error` / timeout | Card with next-step hint; SQLite message shown as detail; timeout is its own state |
| `model_error` / not configured | Distinct cards; not-configured points to the backend env vars |
| Network / client timeout / cancelled | Backend-unavailable, wait-limit, and neutral cancelled cards |
| Malformed response / validation / unknown database | Their own cards (no raw payloads) |

Result table: sticky header + row index, tabular numerals, `NULL` / boolean pills,
horizontal scroll inside a focusable region, progressive rendering (100 rows, +200
per click), truncation banner ("first N rows … total isn't known" — no total is
ever claimed), designed empty result.

## Visualization rules

Deterministic client-side logic (`lib/chartability.ts`), no model call. The table
is always the source of truth; a **Chart** view is offered only when:

- ≥ 2 columns and 2–24 rows;
- a label column exists: the first text column, else a numeric `*_id` column;
  labels are unique;
- ≥ 1 numeric measure that is not all-NULL (real measures preferred over id
  columns; a select lets the user switch measure);
- labels that are all `YYYY`, `YYYY-MM` or `YYYY-MM-DD` render as a line chart;
  everything else as horizontal bars (negatives supported).

Otherwise no chart is shown and a one-line reason is given.

## Benchmark presentation

Numbers live only in `frontend/src/data/results.ts`; nothing is computed or
averaged. Three separate cards, each with its own metric, scope, sample size and
caveat; bars use a fixed 0–100 axis so small gains look small.

The table below documents the values shown on the landing-page cards (`frontend/src/data/results.ts`).
They are a presentation layer; the frozen scientific results are in [`RESULTS.md`](RESULTS.md).

| Card | Metric | Base → Fine-tuned | Δ |
|---|---|---|---|
| Seen training-set specialization (n=500) | Normalized SQL exact match | 34.20% → 67.40% | +33.20 pp — *seen data, not generalization* |
| Schema-held-out validation (n=534) | Execution accuracy, SchemaForge SQLite comparator | 37.40% → 60.60% | +23.20 pp |
| BIRD MiniDev (n=500) | Execution accuracy | 43.60% → 59.03% | +15.43 pp |

Deployment: F16 base 7.50 GB · Q4_K_M base 2.33 GB (~69% smaller) · Q4_K_M + LoRA
2.39 GB effective · 10.86 tok/s CPU generation · ~5.7 GB peak process memory.
The page states the caveat that Q4_K_M matched F16 output exactly on only 4 of 10
sanity prompts and that equivalence is **not** claimed. No users, customers, uptime
or traffic figures appear anywhere. The hero demo is labelled as an example
captured from the Phase 8 smoke run.

## Local development

```powershell
# 1. Backend (from the repo root; model env vars as in docs/PHASE8.md)
uv run uvicorn localsql.backend.api:create_app_from_env --factory --port 8000

# 2. Frontend
cd frontend
npm install
# no env file needed: dev defaults already point at 127.0.0.1:8000
npm run dev                       # http://localhost:5173
```

Scripts: `npm run dev | build | preview | typecheck | lint | test`.

## Testing

Vitest + Testing Library (jsdom), behavior-focused, all backend calls mocked — no
snapshots. Coverage: API client (success, pipeline-outcome-on-non-2xx, error
envelope, non-JSON error, network, timeout, cancel, malformed payloads),
failure taxonomy, chart eligibility, SQL tokenizer (lossless, inert markup),
history storage (corrupt/quota), and the workspace flows (database loading,
health/not-configured/offline, successful query, business-context submission,
Ctrl+Enter, loading state, cancel, SQL display, table, truncation, copy,
NULL/boolean/number rendering, XSS inertness, chart toggle, every failure state,
retry, history reopen) plus the landing page's benchmark accuracy and honesty
assertions. Python side: `tests/backend/test_api.py` covers the CORS allow-list.

## Real integration smoke (manual)

frontend → Phase 8 FastAPI → real Q4_K_M + LoRA runtime → demo SQLite DB → rendered
result. Steps are in the launch sequence below; verify in the browser:

1. Landing page renders; "Open the workspace" navigates to `/workspace`.
2. Header pill turns green (model server ready); the `demo` database is listed.
3. Click the *Engineering payroll* example → **Run query**; the loading state shows
   a running clock for ~10–30 s.
4. The result shows the generated SQL (copyable), *Safety passed / Read-only*,
   a table with `625,000`, and timings/runtime details.
5. Ask something the model cannot answer safely or refers to a missing column and
   confirm the failure card renders (no raw errors).

```powershell
# terminal 1 (repo root)
$P = (Get-Location).Path
$env:SCHEMAFORGE_LLAMA_EXE = "C:\SchemaForge-Phase7\tools\llama-b10964-win-cpu-x64\llama-completion.exe"
$env:SCHEMAFORGE_BASE_GGUF = "$P\.artifacts\phase7\q4_k_m\base-Q4_K_M.gguf"
$env:SCHEMAFORGE_LORA_GGUF = "$P\.artifacts\phase7\lora-gguf\lora-1518-f16.gguf"
uv run python scripts/smoke_phase8.py            # (re)creates the demo DB
uv run uvicorn localsql.backend.api:create_app_from_env --factory --port 8000
# terminal 2
cd frontend; npm install; npm run dev
```

## Limitations

- *Historical (Phase 9):* each query reloaded the model (~20 s+). **Superseded in Phase 11**:
  the deployed backend uses a persistent llama.cpp server (model loaded once; see
  [`PHASE11.md`](PHASE11.md)); the subprocess runtime remains as the development fallback.
- Frontend tests do not use a real browser; visual quality is verified manually
  (screenshots) rather than by snapshot tests.
- Single dark theme; no auth, no server-side history, no arbitrary-SQL editor
  (by design).

## Phase 9 PASS criteria

- `npm test`, `npm run typecheck`, `npm run lint`, `npm run build` all succeed.
- `uv run pytest -q` remains green.
- The manual real-integration smoke above completes: browser shows a rendered
  result from the real Q4_K_M + LoRA runtime on the demo database.
