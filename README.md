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
the training-data pipeline, and `docs/EVALUATION.md` for the benchmark.

## Current phase: Phase 2 -- data foundation + external evaluation

This repository implements two things so far:

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

No model training, inference, or application code exists yet.

## Setup

Requires Python 3.11 and [`uv`](https://docs.astral.sh/uv/).

```powershell
uv sync
uv sync --group eval   # only needed to run the official BIRD evaluator
```

## Commands

```powershell
# Phase 1: inspect / prepare BIRD training data
uv run python scripts/inspect_bird.py
uv run python scripts/prepare_bird.py

# Phase 2: set up / evaluate against BIRD Mini-Dev
uv run python scripts/setup_bird_minidev.py
uv run python scripts/evaluate_bird_minidev.py --predictions path\to\predictions.jsonl

# Run tests (no network required)
uv run pytest -q
```

## Project layout

```text
configs/data.yaml         Phase 1 pipeline constants
configs/benchmark.yaml    Phase 2 benchmark constants
src/localsql/data/        Phase 1: typed models, loaders, serializer,
                           prompt builder, splitter, validator
src/localsql/benchmark/   Phase 2: Mini-Dev loader, manifest builder,
                           prediction contract, diagnostics, official
                           evaluator adapter, report assembly
scripts/                  inspect_bird.py, prepare_bird.py,
                           setup_bird_minidev.py, evaluate_bird_minidev.py
tests/                    Unit + fixture-based tests
docs/                     ARCHITECTURE.md, DATA_CONTRACT.md, EVALUATION.md
data/                     raw/ processed/ benchmarks/ reports/ (gitignored)
```
