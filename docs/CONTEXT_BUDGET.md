# LocalSQL Training Context Policy (Phase 5A/5B)

**Status: Phase 5B IN PROGRESS.** This document records the Phase 5A
analysis and candidate preprocessing policy. As of Phase 5B,
`max_seq_length=4096` for the candidate (compacted) dataset is
**real-tokenizer-confirmed** (zero over-4096 examples across all 6,067
train / 534 validation candidate examples, tokenizer revision
`cdbee75f17c01a7cc42f958dc650907174af0554`) -- see `PROJECT.md` (Phase
5B) for the exact numbers and for the still-outstanding GPU memory/
throughput/resume certification gates. No full training has occurred.
BIRD Mini-Dev generation/evaluation is untouched.

## Correction

An earlier version of this document's linked table in `PROJECT.md`
reported `count > 4096 = 1,626` for the canonical representation without
clearly labeling it as this phase's own local estimate. **The real number,
from the actual Kaggle Phase 4 tokenizer profile, is 1,398** (full SFT) /
1,397 (prompt-only); `>8192` is 539 (both). See `PROJECT.md` (Phase 5A,
section B) for the full correction and root-cause explanation. Every
number in this document and `PROJECT.md` is now explicitly labeled
"(real)" or "(est.)".

## The question

Phase 4's real Kaggle token profile showed 1,398/6,067 (23%) training
examples exceed 4096 tokens (539, 8.9%, exceed 8192), concentrated in 9
databases dominated by `works_cycles` (65 tables/455 columns) and
`hockey` (22 tables/300 columns). Is the long tail caused by verbose
formatting that's safe to drop, or by genuinely large schemas needing
real selection?

## Finding: mostly formatting

Removing AI-generated column descriptions (keeping every table, column,
data type, primary key, and foreign-key relationship intact) cuts schema
text by 86.6%-90.3% across all 9 flagged databases. Estimated: under this
compact representation, zero training examples exceed 4096 or 8192 tokens.

## Selected candidate policy: adaptive per-database full-schema compaction

**A database is compacted (whole database, no question-conditioning, no
gold SQL, every table/column/PK/FK preserved) if the real Phase 4 profile
showed at least one of its examples over 4096 tokens; every other
database is left completely unchanged.** Verified programmatically against
the real `db_id` field of `data/processed/train.jsonl`:

- **Train: 9/62 databases, 1,502/6,067 examples compacted** (`works_cycles`
  383, `hockey` 156, `movie_3` 223, `mondial_geo` 211, `synthea` 141,
  `professional_basketball` 113, `donor` 88, `superstore` 82,
  `world_development_indicators` 105); 53/62 databases, 4,565/6,067
  examples unchanged.
- **Validation: 2/7 databases, 173/534 examples compacted**
  (`college_completion` 45, `legislator` 128 -- decided from the local
  estimator only, since Phase 4 never profiled validation with the real
  tokenizer; canonical length only, never gold correctness); 5/7
  databases, 361/534 examples unchanged.

Build it: `uv run python scripts/build_phase5_candidate.py` -> writes
`data/processed_phase5_candidate/{train,validation}.jsonl` + `policy.json`
(gitignored, reproducible). Asserts: exact counts (6,067/534), db
isolation unchanged, every `completion` byte-identical, every
`example_id` retained exactly once, structural preservation (every table/
column/PK/FK) for every compacted database, and original
`data/processed/*.jsonl` files verified byte-unchanged before/after.

## Question-conditioned schema budgeter: evaluated, NOT selected

`localsql.schema_context.relevance.select_schema_within_budget` (deterministic,
inference-time-safe by construction -- no gold parameter exists) was
evaluated in this phase as an alternative for `works_cycles` specifically
(section E/F in `PROJECT.md`): applied to all 383 examples, it fit
everything under a 3584 candidate with ~98% gold-table/column recall.
**It is not used in the candidate dataset above** -- 98.17% retention is
not 100%, and compact-full-schema alone already solves the 4096/8192
problem without any selection complexity or information-loss risk. The
code remains in the repository as documented research tooling, evaluated
but not chosen; it is not deleted and nothing about its results is
retracted.

## No embeddings, no RAG, no LLM call

Pure deterministic term-matching + a foreign-key graph closure (the
evaluated-but-unused selector) or a static per-database rule (the selected
candidate policy). Nothing here is wired into Phase 1-4 training/
evaluation artifacts.

## Important caveat: candidate numbers are estimates for validation, real for train membership

Train database *membership* in the compact set is real-Kaggle-data-derived.
Validation database membership, and ALL schema/token-length numbers not
directly copied from the Phase 4 Kaggle report, are character-count
estimates (`localsql.schema_context.token_estimate`, calibrated against
real Phase 4 numbers -- see `PROJECT.md` for methodology and error
margins) -- not real tokenizer output.

## Training budget vs. product context capacity

A T4 training-sequence-length budget is **not** a statement about what the
eventual LocalSQL product can serve to real users. See `PROJECT.md`
(Phase 5A) for the one real T4 memory data point available so far
(11,550.2 MB peak at a 3,593-token max in the Phase 4 smoke test) and the
three gates that must pass before 4096 is accepted (real-tokenizer
confirmation of zero over-4096 candidate examples, then a longest-example
GPU memory certification -- neither performed in this phase).

## Commands

```powershell
# Build the candidate dataset (offline, no model/CUDA)
uv run python scripts/build_phase5_candidate.py

# Real-tokenizer profiling of the candidate (Kaggle -- already run; see PROJECT.md Phase 5B)
uv sync --group model --group train
uv run python scripts/run_qlora_smoke.py --run-id phase5-candidate-profile \
  --input data/processed_phase5_candidate/train.jsonl --token-profile

# Evaluated-but-not-selected alternative (question-conditioned selector), kept for the record
uv run python scripts/analyze_schema_context.py --split train --budget-tokens 3584 --write-derived

# Phase 5B: build the GPU/throughput certification sets from the candidate
# + the real longest-examples manifest above (offline, no model/CUDA)
uv run python scripts/build_certification_sets.py
```

See `PROJECT.md` (Phase 5A for the original analysis/per-database tables/
correction; Phase 5B for the real-tokenizer confirmation and the
still-outstanding GPU memory/throughput/resume certification gates before
any full training run).
