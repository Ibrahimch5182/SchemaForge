# CLAUDE.md

Guidance for future Claude sessions working in this repository.

## What LocalSQL is

A capstone project to fine-tune a compact open-weight LLM (QLoRA) that
converts `question + schema + optional business context` into one
executable read-only SQL query. Target model: `Qwen/Qwen3-4B-Instruct-2507`
(smaller Qwen3-family fallback if compute requires it). Training data:
`birdsql/bird23-train-filtered`. Evaluation: BIRD Mini-Dev.

## Current phase

**Phase 1 only: project foundation + reproducible training-data pipeline.**
No model training, no inference, no application code exists yet. Do not
implement later phases unless explicitly asked.

## Architecture (see `docs/ARCHITECTURE.md`, `docs/DATA_CONTRACT.md`)

```
BIRD --> prepare --> baseline --> QLoRA --> evaluate --> quantize   (future)
User question --> schema introspection --> fine-tuned model
  --> SQL safety validation --> read-only execution --> result       (future)
```

## Important invariants

1. BIRD Mini-Dev is evaluation-only -- never load it as training data.
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
7. Do not add dependencies casually. Phase 1 stays on lightweight
   data-engineering deps (`datasets`-free -- see below -- `huggingface_hub`,
   `pydantic`, `sqlglot`, `pyyaml`, `pytest`). No `torch`/`transformers`/
   `trl`/`peft`/`bitsandbytes` until a training phase is authorized.
8. Do not implement future phases early (no FastAPI, no frontend, no
   agents, no model download/training).

## Environment note

This machine's Windows Application Control policy blocks `pyarrow`'s and
`pandas`'s compiled extensions from loading (`numpy` is fine). The `datasets`
library therefore cannot be imported here. The pipeline avoids it entirely:
raw BIRD JSONL and schema JSON are downloaded via `huggingface_hub` /
`fsspec` and parsed with the standard library `json` module. Keep it that
way unless this constraint is re-verified as no longer applicable.

## Relevant commands

```powershell
# Environment setup (Python 3.11 via uv)
uv sync

# Inspect the real BIRD dataset (writes data/reports/bird_inspection_report.json)
uv run python scripts/inspect_bird.py

# Full data preparation pipeline (writes data/processed/*.jsonl + validation report)
uv run python scripts/prepare_bird.py

# Tests (no network required -- uses tests/data/sample_* fixtures)
uv run pytest -q
```

## Where decisions/docs live

- `docs/DATA_CONTRACT.md` -- raw sources, internal schema, prompt format,
  split methodology, generated artifacts.
- `docs/ARCHITECTURE.md` -- full future architecture (not yet built).
- `configs/data.yaml` -- pipeline constants (seed, dialect, split fraction,
  evidence keep-probability, source identifiers). Don't scatter magic
  numbers into Python.
