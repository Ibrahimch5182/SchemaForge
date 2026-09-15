# CLAUDE.md

Guidance for future Claude sessions working in this repository.

## What LocalSQL is

A capstone project to fine-tune a compact open-weight LLM (QLoRA) that
converts `question + schema + optional business context` into one
executable read-only SQL query. Target model: `Qwen/Qwen3-4B-Instruct-2507`
(smaller Qwen3-family fallback if compute requires it). Training data:
`birdsql/bird23-train-filtered`. Evaluation: BIRD Mini-Dev.

## Current phase

**Phase 5A IN PROGRESS: candidate training-context policy finalized, NOT
yet confirmed with the real tokenizer.** Phases 1-4 are complete (Phase 4:
real Kaggle QLoRA smoke test, 20/20 steps, adapter saved + reload-
verified). **Selected candidate policy**: adaptive per-database full-
schema compaction (`localsql.schema_context.db_policy`,
`scripts/build_phase5_candidate.py`) -- 9/62 train DBs (1,502/6,067
examples, real-Kaggle-data-derived) and 2/7 validation DBs (173/534,
local-estimate-derived) get the compact serializer; everything else is
byte-identical to Phase 1. The question-conditioned schema budgeter
(98.17% gold retention, not 100%) was evaluated and explicitly NOT
selected -- it remains in the repo as documented research tooling only.
**Candidate `max_seq_length=4096` is provisional until a real-tokenizer
Kaggle profile of the candidate dataset confirms zero over-4096 examples
and a longest-example GPU memory certification succeeds -- see
`PROJECT.md` (Phase 5A) before treating it as final.** A prior version of
this document's token counts contained an uncorrected local-estimate
number (1,626) presented ambiguously next to real numbers; the real
Phase 4 figure is 1,398 -- see `PROJECT.md` for the full correction. No
full training run and no application code exist yet. See `PROJECT.md` for
the full chronological
engineering journal. Do not implement later phases unless explicitly
asked.

## Architecture (see `docs/ARCHITECTURE.md`, `docs/DATA_CONTRACT.md`, `docs/EVALUATION.md`, `docs/BASELINE.md`, `docs/TRAINING.md`, `docs/CONTEXT_BUDGET.md`)

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
   `pytest`) by default, plus three optional groups: `eval`
   (`func_timeout`, `pymysql`, `psycopg2-binary` -- to import the official
   BIRD evaluator), `model` (`torch`, `transformers`, `accelerate`,
   `bitsandbytes` -- Phase 3 baseline inference), and `train` (`peft`,
   `trl` only -- Phase 4 QLoRA; deliberately does not re-list torch/
   transformers/accelerate/bitsandbytes, install alongside `model`, never
   reinstall Kaggle's own torch build). Not installed on this Windows
   machine; `localsql.model.*` and `localsql.train.*` use lazy imports so
   both stay importable without them.
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
    cloud/Kaggle CUDA machine, run by the user, not by Claude. Real result
    (Kaggle, resolved revision `cdbee75f17c01a7cc42f958dc650907174af0554`):
    official EX 43.6, Soft-F1 47.6975, 500/500 generated, 0 failures.
11. Phase 4 QLoRA is a starting configuration, not a final one: LoRA
    r=16/alpha=32/dropout=0.05 on all 7 attention+MLP projections,
    completion-only loss (prompt tokens label -100, explicit and tested,
    never an SFT framework's default). `scripts/run_qlora_smoke.py` reads
    only Phase 1's `train.jsonl` -- never BIRD Mini-Dev. Adapter-reload
    verification is a plumbing check, not an accuracy evaluation; Mini-Dev
    scoring happens later, via the existing Phase 2 evaluator, only once a
    real fine-tuned checkpoint exists. Real Kaggle smoke result (200
    examples, 20/20 steps): final loss 0.410, peak GPU memory 11,550.2 MB,
    adapter saved + reload-verified; required
    `PYTORCH_ALLOC_CONF=expandable_segments:True` after an initial
    allocator-fragmentation OOM (no hyperparameter change). **Real token
    profile over all 6,067 training examples: 1,398 (~23%) exceed 4096
    tokens, 539 (~8.9%) exceed 8192, concentrated in `works_cycles` and
    `hockey`. Neither 4096 nor 8192 is approved for Phase 5 full training
    -- the context-length/schema strategy is an explicit unresolved Phase 5
    decision. Never silently truncate gold SQL or silently drop a
    database.**

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
uv sync --group eval             # only needed to run the official BIRD evaluator
uv sync --group model            # only on a CUDA cloud machine (baseline)
uv sync --group model --group train  # only on a CUDA cloud machine (QLoRA smoke test)

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

# Phase 4: QLoRA smoke test (dry-run/token-profile work without CUDA where noted)
uv run python scripts/run_qlora_smoke.py --run-id qlora-smoke-check --dry-run --max-train-examples 50
# Real run (cloud/Kaggle CUDA only):
#   uv run python scripts/run_qlora_smoke.py --run-id qlora-smoke-1 --max-train-examples 200 --max-steps 20

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
  envelope, resume/provenance design, cloud workflow, real results.
- `docs/TRAINING.md` -- Phase 4: QLoRA smoke-test rationale, SFT
  formatting, completion-only masking, token profiling, cloud workflow.
- `docs/CONTEXT_BUDGET.md` -- Phase 5A: schema-context analysis, compact
  serialization, deterministic budgeter, open decisions (not final).
- `docs/ARCHITECTURE.md` -- full future architecture (app/production not
  yet built).
- `configs/data.yaml` -- Phase 1 pipeline constants.
- `configs/benchmark.yaml` -- Phase 2 benchmark constants (source revisions,
  expected counts, metric config).
- `configs/model.yaml` -- Phase 3 baseline model/runtime/generation
  constants.
- `configs/train.yaml` -- Phase 4 QLoRA smoke-test constants (LoRA,
  optimization, provisional max_seq_length). Don't scatter magic numbers
  into Python.
