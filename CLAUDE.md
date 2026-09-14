# CLAUDE.md

Guidance for future Claude sessions working in this repository.

## What LocalSQL is

A capstone project to fine-tune a compact open-weight LLM (QLoRA) that
converts `question + schema + optional business context` into one
executable read-only SQL query. Target model: `Qwen/Qwen3-4B-Instruct-2507`
(smaller Qwen3-family fallback if compute requires it). Training data:
`birdsql/bird23-train-filtered`. Evaluation: BIRD Mini-Dev.

## Current phase

**Phase 3 IN PROGRESS: baseline-inference infrastructure implemented; the
real GPU baseline run has not yet happened.** Phases 1 (training-data
pipeline) and 2 (external BIRD Mini-Dev evaluation) are complete. No
fine-tuning/training, no application code exists yet. See `PROJECT.md` for
the full chronological engineering journal. Do not implement later phases
unless explicitly asked.

## Architecture (see `docs/ARCHITECTURE.md`, `docs/DATA_CONTRACT.md`, `docs/EVALUATION.md`, `docs/BASELINE.md`)

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
   `pytest`) by default, plus two optional groups: `eval`
   (`func_timeout`, `pymysql`, `psycopg2-binary` -- to import the official
   BIRD evaluator) and `model` (`torch`, `transformers`, `accelerate`,
   `bitsandbytes` -- Phase 3 baseline inference only, not installed on this
   Windows machine; `localsql.model.*` uses lazy imports so it stays
   importable without it). No `trl`/`peft`/LoRA until a training phase is
   authorized.
8. Do not implement future phases early (no FastAPI, no frontend, no
   agents, no model download/training on this machine).
9. The official BIRD Mini-Dev evaluator (`evaluation_ex.py`,
   `evaluation_f1.py`, `evaluation_utils.py`) is vendored unmodified at a
   pinned commit and imported directly, never copy-pasted/edited. If it
   can't run, report the precise blocker -- never substitute a custom metric
   and call it official EX/Soft-F1.
10. Baseline model is locked: `Qwen/Qwen3-4B-Instruct-2507`, untouched
    (no fine-tuning), 4-bit NF4 (matches the future QLoRA base
    representation so the comparison isolates fine-tuning). Reuses the
    Phase 1 canonical prompt as a single chat user message wrapped only by
    Qwen's own chat template -- never a second Text-to-SQL prompt.
    `predicted_sql` is `raw_completion.strip()` only, never repaired.
    `scripts/run_baseline.py` reads only the Phase 2 gold-free generation
    manifest -- never grading/gold files. The real GPU run happens on a
    cloud/Kaggle CUDA machine, run by the user, not by Claude.

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
uv sync --group eval    # only needed to run the official BIRD evaluator
uv sync --group model   # only on a CUDA cloud machine, for the real baseline

# Phase 1: inspect / prepare BIRD training data
uv run python scripts/inspect_bird.py
uv run python scripts/prepare_bird.py

# Phase 2: set up / evaluate against BIRD Mini-Dev
uv run python scripts/setup_bird_minidev.py
uv run python scripts/evaluate_bird_minidev.py --predictions path\to\predictions.jsonl
uv run python scripts/evaluate_bird_minidev.py --oracle-sanity   # plumbing check, NOT a model result

# Phase 3: baseline inference (dry-run works without CUDA/model deps)
uv run python scripts/run_baseline.py --manifest data\benchmarks\bird_mini_dev\generation\manifest.jsonl --run-id qwen3-4b-base-nf4-smoke --dry-run --limit 5
# Real run (cloud/Kaggle CUDA only): drop --dry-run/--limit

# Tests (no network required -- uses committed synthetic/sample fixtures)
uv run pytest -q
```

## Where decisions/docs live

- `PROJECT.md` -- chronological engineering journal across all phases;
  append new phases here rather than recreating it.
- `docs/DATA_CONTRACT.md` -- Phase 1: raw sources, internal schema, prompt
  format, split methodology, generated artifacts.
- `docs/EVALUATION.md` -- Phase 2: BIRD Mini-Dev variant, gold isolation,
  prediction contract, evaluator integration.
- `docs/BASELINE.md` -- Phase 3: baseline model/runtime rationale, prompt
  envelope, resume/provenance design, cloud workflow.
- `docs/ARCHITECTURE.md` -- full future architecture (training/app not yet
  built).
- `configs/data.yaml` -- Phase 1 pipeline constants.
- `configs/benchmark.yaml` -- Phase 2 benchmark constants (source revisions,
  expected counts, metric config).
- `configs/model.yaml` -- Phase 3 baseline model/runtime/generation
  constants. Don't scatter magic numbers into Python.
