# LocalSQL Architecture (High Level)

This document describes the intended end-to-end architecture across all
phases. **Only Phase 1 (the data foundation) is implemented today.**
Everything else below is a plan, not code.

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
- **evaluate**: score against BIRD Mini-Dev, which is held out and never
  used as training data.
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

Implementation of the training loop, evaluation harness, application
backend/frontend, or any agent/RAG/orchestration layer is explicitly out of
scope until a later phase is authorized.
