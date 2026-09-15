# LocalSQL Architecture (High Level)

This document describes the intended end-to-end architecture across all
phases. **Phases 1-3 are complete (Phase 3's real Kaggle baseline: official
EX 43.6, Soft-F1 47.6975). Phase 4 (QLoRA smoke-test infrastructure) is
implemented but no real Kaggle GPU training run has happened yet.**
Everything else below is a plan, not code. See `PROJECT.md` for the full
chronological journal.

## Offline ML pipeline (future phases)

```text
BIRD (filtered train) --> prepare --> baseline eval --> QLoRA fine-tune --> evaluate (BIRD Mini-Dev) --> quantize
```

- **prepare** (Phase 1, implemented): raw BIRD rows + official schema
  metadata -> canonical prompt/completion examples, split by `db_id`.
- **baseline** (Phase 3, complete): the untouched `Qwen/Qwen3-4B-Instruct-2507`
  (4-bit NF4, matching the QLoRA base representation) against the canonical
  prompt contract. Real Kaggle result: official EX 43.6, Soft-F1 47.6975,
  500/500 generated.
- **QLoRA fine-tune** (Phase 4, smoke-test infrastructure implemented, no
  real run yet): 4-bit QLoRA supervised fine-tuning of
  `Qwen/Qwen3-4B-Instruct-2507` using Hugging Face Transformers + PEFT +
  bitsandbytes (TRL installed, not yet the training-loop driver -- see
  `docs/TRAINING.md`) on Phase 1's `train.jsonl`, reused verbatim. A
  smoke test (bounded steps/subset) validates the mechanics first; a
  full-length run is a later, separate decision.
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
- See `docs/BASELINE.md` for the full rationale, workflow, and real results.

## What Phase 4 actually built (smoke-test infrastructure only -- no run yet)

- QLoRA config (`src/localsql/train/config.py`), SFT data loading +
  explicit, unit-tested completion-only label masking
  (`src/localsql/train/sft_data.py`, reusing Phase 1's `train.jsonl`
  verbatim), and a QLoRA backend (`src/localsql/train/qlora_backend.py`,
  lazy torch/transformers/peft/bitsandbytes imports, plain HF `Trainer`
  fed our own pre-masked labels, adapter save/reload, NaN/Inf loss
  detection).
- `scripts/run_qlora_smoke.py`: `--dry-run`, `--token-profile` (real
  training-prompt + full-SFT-sequence token stats, never auto-applied to
  `max_seq_length`), bounded smoke training (`--max-train-examples`,
  `--max-steps`), and `--verify-adapter` (plumbing check only, not an
  accuracy evaluation; BIRD Mini-Dev untouched).
- See `docs/TRAINING.md` for the full rationale and workflow.

Implementation of a full-length training run, application backend/frontend,
or any agent/RAG/orchestration layer is explicitly out of scope until a
later phase is authorized.
