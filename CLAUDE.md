# CLAUDE.md

Guidance for future Claude sessions working in this repository.

## What LocalSQL is

A capstone project to fine-tune a compact open-weight LLM (QLoRA) that
converts `question + schema + optional business context` into one
executable read-only SQL query. Target model: `Qwen/Qwen3-4B-Instruct-2507`
(smaller Qwen3-family fallback if compute requires it). Training data:
`birdsql/bird23-train-filtered`. Evaluation: BIRD Mini-Dev.

## Current phase

**Phase 2 complete: training-data pipeline (Phase 1) + external BIRD
Mini-Dev evaluation system (Phase 2).** No model training, no inference, no
application code exists yet. Do not implement later phases unless
explicitly asked.

## Architecture (see `docs/ARCHITECTURE.md`, `docs/DATA_CONTRACT.md`, `docs/EVALUATION.md`)

```
BIRD --> prepare --> baseline --> QLoRA --> evaluate --> quantize   (future)
User question --> schema introspection --> fine-tuned model
  --> SQL safety validation --> read-only execution --> result       (future)
```

## Important invariants

1. BIRD Mini-Dev is evaluation-only -- never load it as training data. Only
   the locked variant (original 500 SELECT-only SQLite examples, 11 DBs) is
   used; the newer Mini-Dev V2 / LiveSQLBench CRUD additions are excluded.
   Grading truth (`GradingExample.sql`) is sourced ONLY from the official
   archive's `mini_dev_sqlite_gold.sql` -- never from HF's `SQL` field,
   which is diagnostic-only (HF/archive diverge on 18/500 rows; see
   `docs/EVALUATION.md`). Official timeout stays 30s; expected oracle
   ceiling is ~99.6% (498/500), not 100%, due to two genuinely slow gold
   queries -- don't "fix" this by raising the timeout or special-casing IDs.
2. Train/validation splitting is always by `db_id` (never per-example) to
   measure generalization to unseen schemas. Zero `db_id` overlap is a hard
   requirement, checked and asserted in code.
3. Model output contract is SQL only: no markdown fences, no `SQL:`
   prefix, no chain-of-thought, no commentary.
4. SQL execution will eventually be read-only and deterministically
   validated (not implemented yet).
5. No LangGraph, no agent framework, no RAG/vector DB, no MCP.
6. Do not alter the ML methodology (model choice, QLoRA, dataset choice)
   without explicit user instruction.
7. Do not add dependencies casually. Stays on lightweight data-engineering
   deps (`huggingface_hub`, `pydantic`, `sqlglot`, `pyyaml`, `fsspec`,
   `pytest`), plus an optional `eval` group (`func_timeout`, `pymysql`,
   `psycopg2-binary`) needed only to import the unmodified official BIRD
   evaluator. No `torch`/`transformers`/`trl`/`peft`/`bitsandbytes` until a
   training phase is authorized.
8. Do not implement future phases early (no FastAPI, no frontend, no
   agents, no model download/training).
9. The official BIRD Mini-Dev evaluator (`evaluation_ex.py`,
   `evaluation_f1.py`, `evaluation_utils.py`) is vendored unmodified at a
   pinned commit and imported directly, never copy-pasted/edited. If it
   can't run, report the precise blocker -- never substitute a custom metric
   and call it official EX/Soft-F1.

## Environment note

This machine's Windows Application Control policy blocks `pyarrow`'s and
`pandas`'s compiled extensions from loading (`numpy` is fine). The `datasets`
library therefore is NOT a project dependency (removed after Phase 1 -- it
was never actually imported). Raw BIRD/Mini-Dev JSON and schema files are
downloaded via `huggingface_hub` / `fsspec` and parsed with the standard
library `json` module. Keep it that way unless this constraint is
re-verified as no longer applicable. `func_timeout`/`pymysql`/
`psycopg2-binary` (the `eval` dependency group) do install and import fine
on this machine (Windows wheels available).

## Relevant commands

```powershell
# Environment setup (Python 3.11 via uv)
uv sync
uv sync --group eval   # only needed to run the official BIRD evaluator

# Phase 1: inspect / prepare BIRD training data
uv run python scripts/inspect_bird.py
uv run python scripts/prepare_bird.py

# Phase 2: set up / evaluate against BIRD Mini-Dev
uv run python scripts/setup_bird_minidev.py
uv run python scripts/evaluate_bird_minidev.py --predictions path\to\predictions.jsonl
uv run python scripts/evaluate_bird_minidev.py --oracle-sanity   # plumbing check, NOT a model result

# Tests (no network required -- uses committed synthetic/sample fixtures)
uv run pytest -q
```

## Where decisions/docs live

- `docs/DATA_CONTRACT.md` -- Phase 1: raw sources, internal schema, prompt
  format, split methodology, generated artifacts.
- `docs/EVALUATION.md` -- Phase 2: BIRD Mini-Dev variant, gold isolation,
  prediction contract, evaluator integration.
- `docs/ARCHITECTURE.md` -- full future architecture (training/app not yet
  built).
- `configs/data.yaml` -- Phase 1 pipeline constants.
- `configs/benchmark.yaml` -- Phase 2 benchmark constants (source revisions,
  expected counts, metric config). Don't scatter magic numbers into Python.
