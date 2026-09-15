# LocalSQL

LocalSQL is a capstone project exploring whether a compact, open-weight LLM
can be fine-tuned (QLoRA) into a reliable, schema-generalizing Text-to-SQL
model: given a natural-language question, a relational database schema, and
optional business context, produce one executable, read-only SQL query.

## Research question

Can a ~4B-parameter open model, fine-tuned with 4-bit QLoRA on a filtered
slice of the BIRD training set, generate correct SQL for **database schemas
it never saw during training** -- and does giving it optional business
context (vs. schema alone) meaningfully change accuracy? Answering this
requires a training-data pipeline that splits by database (not by example)
and treats business context as something to sometimes withhold, not always
provide. This repository has not yet trained or evaluated a model, so no
accuracy numbers exist yet.

## Architecture (high level)

```text
Offline ML:   BIRD --> prepare --> baseline --> QLoRA --> evaluate --> quantize
Production:   question --> schema introspection --> fine-tuned model
                --> SQL safety validation --> read-only execution --> result
```

See `docs/ARCHITECTURE.md` for the full plan, `docs/DATA_CONTRACT.md` for
the training-data pipeline, `docs/EVALUATION.md` for the benchmark,
`docs/BASELINE.md` for baseline inference, and `docs/TRAINING.md` for the
QLoRA smoke test. `PROJECT.md` is the full chronological engineering
journal.

## Current phase: Phase 4 IN PROGRESS -- QLoRA smoke-test infrastructure

Phases 1, 2, and 3 are complete (Phase 3's real Kaggle baseline: official
EX 43.6, Soft-F1 47.6975). Phase 4 (below) has implemented QLoRA
smoke-test infrastructure; **no real Kaggle GPU training run has happened
yet** -- that is a manual step the user performs next.

**Phase 1 -- reproducible training-data pipeline**
- Loads the real `birdsql/bird23-train-filtered` dataset (6,601 rows / 69
  databases).
- Sources official BIRD schema metadata and merges in column descriptions.
- Deterministically serializes each database schema and builds the single
  canonical prompt/completion pair every later phase reuses.
- Splits by `db_id` (90/10 by default), zero leakage enforced in code.
- Deterministic, seedable dropout of BIRD's `evidence` field.
- Validates every row explicitly -- nothing is silently dropped.

**Phase 2 -- external BIRD Mini-Dev evaluation system**
- Sets up the official, locked **original 500 SELECT-only SQLite** Mini-Dev
  benchmark (11 databases) -- explicitly not the newer Mini-Dev V2 /
  LiveSQLBench CRUD additions.
- Builds a gold-free generation manifest (reusing the Phase 1 schema
  serializer and prompt builder) and an isolated grading reference, joined
  by stable `example_id`s -- gold SQL never appears in the generation-side
  artifact (enforced in code and tested).
- Integrates the unmodified, pinned-commit official EX and Soft-F1
  evaluators; adds LocalSQL-only diagnostics (SQL parse rate, execution
  success rate) that are never conflated with official correctness.
- R-VES is deferred until deployment hardware is fixed.

**Phase 3 -- baseline inference infrastructure (IN PROGRESS)**
- Implements a single Qwen3-4B-Instruct-2507 (untouched, 4-bit NF4) backend
  and a resumable runner that reads *only* the Phase 2 gold-free
  generation manifest.
- Reuses the Phase 1 canonical prompt verbatim as a single chat message,
  wrapped only by Qwen's own chat template.
- Deterministic decoding; raw completions preserved; `predicted_sql` is
  whitespace-trimmed only, never repaired.
- Full model/software/hardware provenance recorded per run; safe
  crash-resume; `--dry-run` validates everything without CUDA/model deps.
- **Real result (Kaggle)**: official EX 43.6, Soft-F1 47.6975, 500/500
  generated, 0 failures. See `docs/BASELINE.md` / `PROJECT.md`.

**Phase 4 -- QLoRA smoke-test infrastructure (IN PROGRESS)**
- Reuses Phase 1's `train.jsonl` verbatim (prompt/completion/evidence
  dropout already baked in) -- never re-split or re-derived.
- Explicit, unit-tested completion-only loss masking (prompt tokens
  label `-100`, only gold-SQL completion tokens trainable).
- A bounded-step smoke runner (`--max-train-examples`, `--max-steps`)
  attaches a LoRA adapter to the same 4-bit NF4 base as the Phase 3
  baseline, trains, saves the adapter, and verifies it reloads correctly
  (not an accuracy evaluation -- BIRD Mini-Dev is untouched here).
- `--token-profile` reports the real training-prompt token-length
  distribution without ever auto-adjusting `max_seq_length`.
- **No training has been run yet** -- that's a manual step on a
  cloud/Kaggle CUDA machine (see `docs/TRAINING.md`).

No actual training run or application code exists yet.

## Setup

Requires Python 3.11 and [`uv`](https://docs.astral.sh/uv/).

```powershell
uv sync
uv sync --group eval                   # only needed to run the official BIRD evaluator
uv sync --group model                  # only on a CUDA cloud machine (baseline)
uv sync --group model --group train    # only on a CUDA cloud machine (QLoRA smoke test)
```

## Commands

```powershell
# Phase 1: inspect / prepare BIRD training data
uv run python scripts/inspect_bird.py
uv run python scripts/prepare_bird.py

# Phase 2: set up / evaluate against BIRD Mini-Dev
uv run python scripts/setup_bird_minidev.py
uv run python scripts/evaluate_bird_minidev.py --predictions path\to\predictions.jsonl

# Phase 3: baseline inference (dry-run needs no GPU/model deps)
uv run python scripts/run_baseline.py --manifest data\benchmarks\bird_mini_dev\generation\manifest.jsonl --run-id qwen3-4b-base-nf4-smoke --dry-run --limit 5

# Phase 4: QLoRA smoke test (dry-run/token-profile need no CUDA where noted)
uv run python scripts/run_qlora_smoke.py --run-id qlora-smoke-check --dry-run --max-train-examples 50

# Run tests (no network required)
uv run pytest -q
```

## Project layout

```text
PROJECT.md                Chronological engineering journal (all phases)
configs/data.yaml         Phase 1 pipeline constants
configs/benchmark.yaml    Phase 2 benchmark constants
configs/model.yaml        Phase 3 baseline model/runtime constants
configs/train.yaml        Phase 4 QLoRA smoke-test constants
src/localsql/data/        Phase 1: typed models, loaders, serializer,
                           prompt builder, splitter, validator
src/localsql/benchmark/   Phase 2: Mini-Dev loader, manifest builder,
                           prediction contract, diagnostics, official
                           evaluator adapter, report assembly
src/localsql/model/       Phase 3: model config, Qwen backend, generation
                           envelope/orchestration, run artifacts/resume
src/localsql/train/       Phase 4: train config, SFT data/completion-only
                           masking, QLoRA backend
scripts/                  inspect_bird.py, prepare_bird.py,
                           setup_bird_minidev.py, evaluate_bird_minidev.py,
                           run_baseline.py, run_qlora_smoke.py
tests/                    Unit + fixture-based tests
docs/                     ARCHITECTURE.md, DATA_CONTRACT.md, EVALUATION.md,
                           BASELINE.md, TRAINING.md
data/                     raw/ processed/ benchmarks/ runs/ (gitignored)
```
