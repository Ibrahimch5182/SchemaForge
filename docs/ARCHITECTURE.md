# LocalSQL Architecture (High Level)

This document describes the intended end-to-end architecture across all
phases. **Phase 1 (data foundation) and Phase 2 (external evaluation) are
complete. Phase 3 (baseline inference infrastructure) is implemented but
the real GPU baseline run has not happened yet.** Everything else below is
a plan, not code. See `PROJECT.md` for the full chronological journal.

## Offline ML pipeline (future phases)

```text
BIRD (filtered train) --> prepare --> baseline eval --> QLoRA fine-tune --> evaluate (BIRD Mini-Dev) --> quantize
```

- **prepare** (Phase 1, implemented): raw BIRD rows + official schema
  metadata -> canonical prompt/completion examples, split by `db_id`.
- **baseline** (Phase 3, infrastructure implemented): run the untouched
  `Qwen/Qwen3-4B-Instruct-2507` (4-bit NF4, matching the future QLoRA base
  representation) against the canonical prompt contract to establish a
  pre-fine-tuning reference point. The real GPU run happens on a
  cloud/Kaggle CUDA machine, performed by the user -- not yet run.
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

## What Phase 3 actually built (infrastructure only -- no run yet)

- One Qwen3-4B-Instruct-2507 4-bit NF4 backend (`src/localsql/model/`),
  lazily importing torch/transformers/bitsandbytes so the rest of the
  repository stays installable/testable without them.
- A resumable, crash-safe runner (`scripts/run_baseline.py`) that consumes
  only the Phase 2 gold-free generation manifest and writes predictions
  conforming directly to the Phase 2 prediction contract.
- Reuse (not replacement) of the Phase 1 canonical prompt, wrapped only by
  Qwen's own chat template; deterministic decoding; whitespace-only output
  normalization; full run provenance recording; a `--dry-run` mode and a
  `--token-profile` mode.
- See `docs/BASELINE.md` for the full rationale and workflow.

Implementation of the training loop (QLoRA/TRL/PEFT), application
backend/frontend, or any agent/RAG/orchestration layer is explicitly out of
scope until a later phase is authorized.
