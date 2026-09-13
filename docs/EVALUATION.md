# LocalSQL Evaluation (Phase 2): BIRD Mini-Dev

Phase 2 builds a scientifically clean, reproducible external evaluation
system. **No model inference happens in this phase** -- it only prepares the
benchmark and can score an existing prediction file.

## Benchmark variant (locked)

**BIRD Mini-Dev, original 500 SELECT-only SQLite examples, 11 databases.**

The upstream `bird-bench/mini_dev` repo also ships "Mini-Dev V2" /
LiveSQLBench: 270 additional CRUD instances across 18 new databases
(`live_sql_bench_sqlite/`). LocalSQL explicitly does **not** use that
variant -- see `excluded_variant` in `configs/benchmark.yaml`.

## Source / revision

- Question metadata (`db_id`, `question`, `evidence`, `difficulty`):
  `birdsql/bird_mini_dev` on Hugging Face, `mini_dev_sqlite` split
  (revision recorded at setup time). Its `SQL` field is **diagnostic only**
  -- see "Canonical grading truth" below.
- 11 SQLite database files, the Spider-format `dev_tables.json` schema,
  per-database `database_description/*.csv` column descriptions, and the
  evaluator-ready `mini_dev_sqlite_gold.sql`: the official `minidev.zip`
  archive (`bird-bench.oss-cn-beijing.aliyuncs.com`). The archive is ~800MB
  total; only the ~346MB of entries actually needed are extracted via HTTP
  range requests (same technique as Phase 1's train.zip handling).
- Official evaluator (`evaluation_ex.py`, `evaluation_f1.py`,
  `evaluation_utils.py`): `bird-bench/mini_dev`, pinned commit
  `abd11b6db92a1c9f809b32f7564c7c71b34d67f0`, vendored unmodified into
  `data/benchmarks/bird_mini_dev/third_party/mini_dev_eval/` with recorded
  sha256 checksums (`SOURCE.json`).

**BIRD Mini-Dev is never used as training data.**

## Canonical grading truth (Phase 2A decision)

`GradingExample.sql` is sourced **only** from the archive's
`mini_dev_sqlite_gold.sql` -- the exact file the official evaluator scores
against. HF's `SQL` field is diagnostic-only and must never determine
grading.

This was decided after diagnosing an oracle-sanity run that scored 96% EX
instead of ~100%: `analyze_source_consistency` (kept as a permanent setup
check, reported in `setup_report.json`) proved HF's `mini_dev_sqlite` split
diverges from the archive's own gold file at **18/500 rows** -- 2 rows with
genuinely different/corrected SQL for the same question, plus a 16-row
block where the archive's own financial-db questions contain duplicates
that HF's set does not. The archive's own question file and its own gold
SQL agree with each other on all 500 rows (verified), so the archive is
internally self-consistent and is the correct anchor for grading, since
that is what official EX/Soft-F1 actually score against.

The remaining 2/500 oracle-sanity discrepancies (`bird-mini-dev-sqlite-0340`,
`-0393`) are **not** a source mismatch: their gold SQL is byte-identical
between predictions and the archive, but each independently exceeds the
30-second execution timeout on this environment (`-0393` completes in
~83.5s at a larger timeout; `-0340` still exceeds 120s) -- a genuine
slow-query/timeout limitation, reproduced with zero multiprocessing
involved, not adapter or resource-contention behavior. The timeout is
retained at 30s (not raised to chase a higher score); both examples stay in
the 500-example denominator, uncorrected and un-special-cased. Expected
full-oracle-sanity ceiling on this environment: **~498/500 (99.6%) EX**.

## Gold isolation

Two separate artifacts, joined only by a stable `example_id`
(`bird-mini-dev-sqlite-0000` .. `-0499`, assigned by official row order):

- `data/benchmarks/bird_mini_dev/generation/manifest.jsonl` -- gold-free.
  Fields: `example_id`, `db_id`, `dialect`, `question`, `business_context`,
  `serialized_schema`, `prompt`, `difficulty`. The Pydantic model
  (`GenerationExample`, `extra="forbid"`) has no SQL/answer field at all, and
  `scripts/setup_bird_minidev.py` self-checks the dumped keys against
  `FORBIDDEN_GENERATION_FIELDS` before finishing. A dedicated test
  (`tests/benchmark/test_manifest_and_isolation.py`) additionally checks the
  gold SQL text itself doesn't appear anywhere in the serialized record.
- `data/benchmarks/bird_mini_dev/grading/reference.jsonl` -- isolated. Adds
  the gold `sql`. Only evaluation code should read this file.

Schema serialization and the prompt template are **reused unmodified** from
Phase 1 (`localsql.data.schema_serializer.serialize_schema`,
`localsql.data.prompt_builder.build_prompt`) -- Mini-Dev schema comes from
the official `dev_tables.json` (same Spider format as Phase 1's
`train_tables.json`), never inferred from gold SQL.

Business context: Mini-Dev's `evidence` maps directly to
`business_context`, kept whenever present. Phase 1's 50% training dropout is
**not** applied to this external benchmark.

## Prediction contract

```json
{"example_id": "bird-mini-dev-sqlite-0000", "db_id": "formula_1", "predicted_sql": "SELECT ..."}
```

Optional metadata (`latency_ms`, `prompt_tokens`, `completion_tokens`,
`model_id`, `context_mode`) is accepted but never required by the grader.
`localsql.benchmark.prediction.validate_predictions` reports duplicates,
unknown ids, `db_id` mismatches, non-string SQL, and missing ids explicitly.

## Metrics

- **Execution Accuracy (EX)** -- primary. Official evaluator, unmodified.
- **Soft-F1** -- secondary. Official evaluator, unmodified.
- **R-VES** -- deferred. Hardware/timing-sensitive; not used for model
  selection in this phase and not benchmarked repeatedly. Will be revisited
  once deployment hardware is fixed.
- **LocalSQL diagnostics** (parse rate, execution success rate, timeout/error
  counts) -- clearly labeled as diagnostics, never conflated with official
  correctness. `parseable != executable != correct`.

## Official evaluator integration

`localsql.benchmark.official_adapter` imports the vendored, unmodified
`evaluation_ex.py` / `evaluation_f1.py` directly (rather than shelling out
and scraping stdout) so their own `compute_acc_by_diff` /
`compute_f1_by_diff` functions can be reused for structured, per-example
results (needed for the by-database breakdown). This was verified to work
on Windows, including the `multiprocessing.Pool`-based execution these
scripts use internally (see smoke test results below). One known upstream
limitation: `compute_acc_by_diff`/`compute_f1_by_diff` divide by the count
of each difficulty bucket, so an evaluation slice (`--limit`) missing an
entire difficulty tier raises `ZeroDivisionError` inside the *unmodified*
upstream code -- the adapter catches this, marks EX/Soft-F1 `UNAVAILABLE`
with the reason, and never substitutes a custom metric.

`evaluation_utils.py` unconditionally imports `psycopg2` and `pymysql` even
though only SQLite is evaluated here; both are installed (with
`func_timeout`) in the optional `eval` dependency group purely so the
unmodified file can be imported (`uv sync --group eval`).

## Oracle sanity mode

`--oracle-sanity` feeds the grading reference's own gold SQL back through
the grader as "predictions," solely to prove the evaluator plumbing is
wired correctly. On this environment, correct wiring scores **~99.6% EX
(498/500)** -- not 100% -- because of the two known gold-query timeouts
above; any other discrepancy indicates a real plumbing bug. It is never a
model result and is clearly marked `oracle_sanity_run: true` in the report.

## Commands

```powershell
uv sync --group eval
uv run python scripts/setup_bird_minidev.py
uv run python scripts/evaluate_bird_minidev.py --predictions path\to\predictions.jsonl
uv run python scripts/evaluate_bird_minidev.py --oracle-sanity   # full 500, run manually
```

See `CLAUDE.md` for the quick command reference and `configs/benchmark.yaml`
for every pinned constant.
