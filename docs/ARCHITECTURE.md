# LocalSQL Architecture (High Level)

This document describes the intended end-to-end architecture across all
phases. **Only Phase 1 (data foundation) and Phase 2 (external evaluation)
are implemented today.** Everything else below is a plan, not code.

## Offline ML pipeline (future phases)

```text
BIRD (filtered train) --> prepare --> baseline eval --> QLoRA fine-tune --> evaluate (BIRD Mini-Dev) --> quantize
```

- **prepare** (Phase 1, implemented): raw BIRD rows + official schema
  metadata -> canonical prompt/completion examples, split by `db_id`.
- **baseline**: run an off-the-shelf instruct model against the canonical
  prompt contract to establish a pre-fine-tuning reference point.
- **QLoRA fine-tune**: 4-bit QLoRA supervised fine-tuning of
  `Qwen/Qwen3-4B-Instruct-2507` (or a smaller Qwen3-family fallback if
  compute requires it) using Hugging Face Transformers + TRL + PEFT +
  bitsandbytes, on the `train.jsonl` produced here.
- **evaluate** (Phase 2, implemented up to the scoring boundary): score
  future model predictions against BIRD Mini-Dev (original 500 SELECT-only
  SQLite, official EX/Soft-F1 evaluator) via the gold-free generation
  manifest / prediction / grading contracts. Mini-Dev is held out and never
  used as training data. Model inference itself is not implemented -- this
  phase only scores a prediction file someone else (or a later phase)
  produces.
- **quantize**: prepare the fine-tuned model for efficient local inference.

## Production application (future phases)

```text
User question
  --> schema introspection (connect to target DB, read its real schema)
  --> schema selection (serialize with the same canonical serializer)
  --> fine-tuned model (same prompt contract as training)
  --> SQL safety validation (deterministic, e.g. AST-based read-only checks)
  --> read-only execution (DB permissions enforce read-only, defense in depth)
  --> result
```

No frontend, backend API, agent framework, or orchestration layer exists
yet. The model output contract (SQL only, no chain-of-thought) is designed
specifically so it can later be validated and executed deterministically
without needing an agent loop.

## What Phase 1 actually built

- Reproducible loading of `birdsql/bird23-train-filtered`.
- Official BIRD schema metadata (`train_tables.json`), sourced legitimately
  and merged with column descriptions.
- Typed internal models, a deterministic schema serializer, and the single
  canonical prompt/completion builder that every later phase will reuse.
- Deterministic database-level train/validation split and evidence dropout.
- Explicit validation and rejection reporting -- no silent data loss.

## What Phase 2 actually built

- Reproducible setup of the locked BIRD Mini-Dev variant (original 500
  SELECT-only SQLite examples, 11 databases) -- see `docs/EVALUATION.md`.
- Gold-free generation manifest + isolated grading reference, reusing
  Phase 1's schema serializer and prompt builder unmodified.
- Canonical prediction contract with explicit validation (duplicates,
  missing, unknown ids, `db_id` mismatch).
- The unmodified, pinned-commit official EX/Soft-F1 evaluator, integrated
  via a thin adapter; LocalSQL-only diagnostics (parse/execution) kept
  clearly separate from official correctness. R-VES deferred.

Implementation of the training loop, model inference, application
backend/frontend, or any agent/RAG/orchestration layer is explicitly out of
scope until a later phase is authorized.
