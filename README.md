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

See `docs/ARCHITECTURE.md` for the full plan and `docs/DATA_CONTRACT.md` for
exactly what this phase built.

## Current phase: Phase 1 -- data foundation

This repository currently implements **only** the reproducible
training-data pipeline:

- Loads the real `birdsql/bird23-train-filtered` dataset (6,601 rows / 69
  databases -- see `data/reports/bird_inspection_report.json` after running
  the inspection script).
- Sources official BIRD schema metadata (tables, columns, types, primary
  keys, foreign keys) and merges in column descriptions.
- Deterministically serializes each database schema into a compact text
  form and builds the single canonical prompt/completion pair that every
  later phase (training, evaluation, inference) will reuse.
- Splits by `db_id` (90/10 by default) so validation measures
  generalization to unseen schemas, with zero leakage enforced in code.
- Applies deterministic, seedable dropout of BIRD's `evidence` field so the
  model isn't trained to depend on oracle context a real user may not give.
- Validates every row explicitly -- nothing is silently dropped.

No model training, inference, evaluation harness, or application code
exists yet.

## Setup

Requires Python 3.11 and [`uv`](https://docs.astral.sh/uv/).

```powershell
uv sync
```

## Commands

```powershell
# Inspect the real BIRD dataset (rerunnable; writes a JSON report)
uv run python scripts/inspect_bird.py

# Run the full data preparation pipeline
uv run python scripts/prepare_bird.py

# Run tests (no network required)
uv run pytest -q
```

## Project layout

```text
configs/data.yaml        Pipeline constants (seed, split fraction, sources)
src/localsql/data/       Typed models, loaders, serializer, prompt builder,
                          splitter, validator
scripts/                 inspect_bird.py, prepare_bird.py
tests/                   Unit + fixture-based tests
docs/                    ARCHITECTURE.md, DATA_CONTRACT.md
data/                    raw/ interim/ processed/ reports/ (gitignored)
```
