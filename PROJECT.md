# LocalSQL Engineering Journal

This is the chronological record of what was built, why, and what was
learned -- written so it still makes sense months from now. New phases are
appended to this file; it is not recreated per phase.

## Project Objective

LocalSQL asks a specific question: can a compact, open-weight LLM
(`Qwen/Qwen3-4B-Instruct-2507`), fine-tuned with 4-bit QLoRA on a filtered
slice of the BIRD training set, generate correct SQL for **database schemas
it never saw during training** -- and does supplying optional business
context (vs. schema alone) meaningfully change accuracy? The end goal is a
Text-to-SQL system that takes `question + schema + optional business
context` and returns one executable, read-only SQL query.

## Locked Architecture and Methodology

These decisions are treated as fixed unless explicitly revisited:

- Base model: `Qwen/Qwen3-4B-Instruct-2507`, untouched until a fine-tuning
  phase is authorized.
- Fine-tuning method (future): 4-bit QLoRA supervised fine-tuning.
- Training data: `birdsql/bird23-train-filtered`.
- Internal train/validation split: by `db_id`, never by individual example.
- External evaluation: BIRD Mini-Dev, original 500 SELECT-only SQLite
  examples, 11 databases -- never used as training data.
- Dialect: SQLite first; PostgreSQL later.
- Model output contract: SQL only, no chain-of-thought, no markdown.
- Excluded for now: agents, LangGraph, RAG/vector DB, MCP, Groq, FastAPI,
  frontend, vLLM/llama.cpp/GGUF, AWS deployment.

---

## Phase 1 -- Data Foundation

### Goal

Build a reproducible pipeline that turns the raw BIRD training data into
model-ready training examples, with a train/validation split that actually
measures generalization to unseen database schemas.

### Why This Phase Was Necessary

Fine-tuning on a mis-split or under-validated dataset produces a model that
looks good on paper and fails on real schemas. Before touching a model, the
project needed: examples that pair a question with a schema the model can
actually see, a split that doesn't leak schema knowledge between train and
validation, and evidence that the input data is actually what it claims to
be (real BIRD questions with real official schema metadata, not something
inferred or approximated).

### What Was Built

- A loader for `birdsql/bird23-train-filtered` (HF dataset: `db_id`,
  `question`, `evidence`, `SQL`).
- A loader for the *official* BIRD schema metadata (`train_tables.json`,
  Spider format: tables, columns, types, primary keys, foreign keys),
  sourced from the official `train.zip` archive -- not inferred from the
  gold SQL, not fabricated. The archive is ~8.9 GB (it bundles per-database
  SQLite files with real row data); only the ~726 KB schema file is
  extracted via HTTP range requests against the remote zip's central
  directory, so the multi-GB archive itself is never downloaded.
- Typed internal models (`ColumnSchema`, `TableSchema`, `DatabaseSchema`,
  `RawBirdExample`, `PreparedTextToSQLExample`).
- A deterministic schema serializer: the same `DatabaseSchema` always
  produces byte-identical text (table/column order taken directly from the
  source, PK/FK/description annotations included only when present).
- A single canonical prompt/completion builder, reused unmodified by every
  later phase (training, Phase 2 evaluation, Phase 3 baseline inference).
- Deterministic, hash-based dropout of BIRD's `evidence` field (default
  50% keep probability) so training doesn't make the model dependent on
  oracle context a real user may not supply.
- A database-level train/validation splitter with an explicit leakage
  assertion.
- Explicit validation with reason codes -- no row is ever silently dropped.

### Important Results

- **6,601 accepted examples.** This is the count of question/schema/SQL
  triples that passed validation (non-empty question, non-empty SQL,
  parseable as SQL via `sqlglot`, schema available) out of the filtered
  BIRD training set -- i.e., 6,601 actual training samples, not a
  theoretical dataset size.
- **69 unique database IDs**, with legitimate official schema coverage for
  all 69 (0 rows rejected for `MISSING_SCHEMA`).
- **62 training databases / 7 validation databases**, giving **6,067
  training examples / 534 validation examples**. The split is applied to
  *databases*, not examples, then examples inherit their database's split
  assignment -- so "6,067 vs. 534" is simply "however many of the 6,601
  examples happen to belong to the 62-vs-7 database split," not a
  separately chosen ratio. With ~90% of 69 databases going to training
  (`train_db_fraction: 0.90`, seed 42), 62/69 landed in training and 7/69
  in validation; those 7 databases happened to contribute 534 examples.
- **Zero database overlap** between train and validation, asserted in code
  (`check_split_leakage`) and re-verified in the validation report.
- Splitting by `db_id` (instead of shuffling individual examples) is the
  single most important methodological choice in this phase: if the same
  database appeared in both train and validation, the model could
  memorize that database's schema/columns instead of learning to
  generalize, and validation accuracy would silently overstate real-world
  performance on schemas it has never seen. Splitting by database is the
  only way validation numbers mean anything for this project's actual
  research question.
- Business-context dropout is a hash of `(seed, db_id, row_index)`, not a
  sequential RNG draw -- so the keep/drop decision for any given example is
  independent of processing order and identical on every rerun.
- The whole pipeline is deterministic end-to-end: two full runs of
  `scripts/prepare_bird.py` produce byte-identical `train.jsonl` /
  `validation.jsonl` (verified via checksum), which matters because the
  same data must be reproducible months later without re-deriving random
  decisions.

### Decisions and Reasoning

- Schema is sourced from the official BIRD release, never inferred from
  gold SQL and never fabricated -- inferring schema from SQL would silently
  bake answer-leakage into the "input" side of training examples.
- The prompt/completion contract (SQL-only, no markdown, no chain-of-
  thought) was fixed once so training, evaluation, and inference never
  drift onto incompatible formats.

### Issues / Technical Debt

- This development machine's Windows Application Control policy blocks
  `pyarrow`'s and `pandas`'s compiled extensions from loading, so the
  Hugging Face `datasets` library (which depends on pyarrow) cannot be
  used here. Worked around by downloading raw JSON/JSONL via
  `huggingface_hub` + `fsspec` and parsing with the standard library
  `json` module -- this constraint shaped every subsequent phase's data
  loading too.

### What We Learned

A 90/10 database split on a modest number of databases (69) produces an
uneven validation set size in examples (534, not exactly 10% of 6,601)
because databases differ in how many questions they contribute -- this is
expected and correct given the goal is schema generalization, not an
exact 90/10 example ratio.

### Outcome

**PASSED.** 31 tests passing at the time, all using synthetic fixtures or a
small committed real-data sample (no network required for tests).

### What Came Next

Phase 2: build a trustworthy way to measure how any future model performs
on real held-out questions, without ever leaking the model training data
or the evaluation answers into each other.

---

## Phase 2 -- Evaluation Foundation

### Goal

Set up the BIRD Mini-Dev benchmark as a scientifically clean, reproducible
external evaluation system: predictions must be generated without any
access to the correct answers, and scoring must use the same official
evaluator the wider BIRD community uses.

### Why This Phase Was Necessary

An evaluation system that a model (or a bug) could see the answers through
is worse than no evaluation at all -- it would produce numbers that look
like generalization but are actually leakage. Before any model generates a
single prediction, the project needed a hard boundary between "what a model
is allowed to see" and "what only the grader is allowed to see," plus
confidence that the official scoring code (not a homemade approximation)
is what actually runs.

### What the 500 Mini-Dev Questions Represent

BIRD Mini-Dev is a curated 500-question development benchmark spanning 11
real-world databases (formula_1, financial, card_games, and others), each
question paired with expert-annotated gold SQL and a difficulty label
(simple/moderate/challenging). The upstream project has since added a
"Mini-Dev V2"/LiveSQLBench variant with 270 additional CRUD (not
SELECT-only) examples across 18 new databases -- LocalSQL explicitly
excludes that variant and uses only the original 500 SELECT-only SQLite
set, because that is the variant this project's SQL-only, read-only
methodology is designed around.

### What Was Built

- A setup script that fetches the official Mini-Dev questions (Hugging
  Face `birdsql/bird_mini_dev`), the official schema/database/gold-SQL
  archive (`minidev.zip`, ~800 MB total; only the ~346 MB of entries
  actually needed -- schema, gold SQL, 11 SQLite databases, column
  descriptions -- are extracted via HTTP range requests), and vendors the
  official evaluator at a pinned commit.
- **Gold isolation**: two separate, positionally-joined artifacts. The
  *generation manifest* (`generation/manifest.jsonl`) contains only
  `example_id`, `db_id`, `question`, `business_context`, serialized
  schema, and the canonical prompt -- its Pydantic model forbids any
  extra field, so a gold SQL column literally cannot be added by accident.
  The *grading reference* (`grading/reference.jsonl`) is the only artifact
  that carries gold SQL, and only evaluation code reads it. Both are
  joined by a stable `example_id` (`bird-mini-dev-sqlite-0000` ..
  `-0499`), independent of any future re-ordering.
- A canonical prediction contract (`example_id`, `db_id`, `predicted_sql`
  plus optional metadata) with explicit validation for duplicates, unknown
  IDs, `db_id` mismatches, and missing predictions.
- A thin adapter around the *unmodified*, pinned-commit official BIRD
  evaluator (`evaluation_ex.py` / `evaluation_f1.py`), imported directly
  (not subprocessed) so per-example results are available for by-database
  and by-difficulty breakdowns, not just an aggregate score.
- LocalSQL-only diagnostics (SQL parse rate via `sqlglot`, SQLite execution
  success/timeout) kept explicitly separate from official correctness --
  a query can parse and execute successfully while still being wrong; only
  the official evaluator determines correctness.
- An oracle-sanity mode that feeds gold SQL back through the grader as
  "predictions," solely to prove the evaluator plumbing works -- never used
  as, or reported as, a model result.

### Execution Accuracy (EX) and Soft-F1

**EX** (primary metric): 1 if the predicted SQL's result set exactly
matches the gold SQL's result set when both are executed against the real
database, 0 otherwise. **Soft-F1** (secondary): a partial-credit row/column
overlap score between predicted and gold result sets, useful when a query
gets some but not all of the right rows/columns. Both come from the
unmodified official evaluator; LocalSQL never redefines or approximates
either.

### The Initial 96% Oracle Result and What It Revealed

The first full oracle-sanity run (gold SQL fed back as "predictions," which
should score at or near 100% if the plumbing is correct) scored **96% EX
and 96% Soft-F1** -- 20 of 500 examples came back "incorrect" despite being
gold-vs-gold. Rather than accept a lower number or dismiss it, this was
treated as a real bug signal requiring root-cause investigation, since the
alternative was shipping an evaluation system nobody could trust.

**Investigation and findings** (adding a diagnostic to capture the official
evaluator's exact per-example results, then cross-checking every input
file the evaluator actually consumed):

- **18 of the 20 failures** were a genuine data-source mismatch. LocalSQL's
  original grading reference sourced `sql` from the Hugging Face
  `bird_mini_dev` dataset's `SQL` field, while the official evaluator
  always scores against the static `minidev.zip` archive's own
  `mini_dev_sqlite_gold.sql`. Those two upstream sources disagree at 18
  positions: 2 rows have a genuinely different/corrected gold query for
  the same question (one of them fixes a real bug -- a missing
  `CAST(...AS REAL)` that made the older query sort lexicographically
  instead of numerically), and 16 rows (the tail of the `financial`
  database's block) come from the archive containing duplicate questions
  that the HF dataset does not. The archive's own question file and its
  own gold SQL agree with each other on all 500 rows, so the divergence is
  specifically "HF vs. archive," not an internal inconsistency in either
  source.
- **2 of the 20 failures** (`bird-mini-dev-sqlite-0340`,
  `bird-mini-dev-sqlite-0393`) were not a source mismatch at all --
  predicted and gold SQL were byte-identical -- but both independently
  exceeded the 30-second execution timeout. Re-executed outside any
  multiprocessing/evaluator machinery to rule out resource contention:
  `-0393` completes in ~83.5 seconds given a larger timeout; `-0340` still
  exceeds 120 seconds. These are genuinely slow queries on unindexed
  BIRD SQLite databases, not an artifact of the evaluator's own
  parallelism (which was already running with a single worker).

### Decision: Archive Gold Is Canonical

LocalSQL's grading reference (`GradingExample.sql`) was changed to source
SQL **only** from the archive's `mini_dev_sqlite_gold.sql` -- the exact
file the official evaluator scores against -- never from HF's `SQL` field.
Reasoning: whatever the official evaluator actually reads is definitionally
what "correct" means for this benchmark; sourcing grading truth from a
different (even if arguably improved) source guarantees disagreement with
official scoring on any row where the two diverge. HF's `SQL` field is
retained as diagnostic-only metadata, and the divergence-detection check
(`analyze_source_consistency`) was kept as a permanent part of the setup
script so any future upstream divergence is surfaced immediately rather
than silently reappearing.

### Two Slow Gold Queries, Retained

`bird-mini-dev-sqlite-0340` and `-0393` remain in the 500-example
denominator, uncorrected and un-special-cased -- they are not removed,
not given a longer timeout just for themselves, and not excluded from
reporting. This is a real property of the benchmark on real hardware, not
a LocalSQL defect, and hiding it would misrepresent what "100% EX" can
even mean here.

### Why the 30-Second Timeout Was Retained

Raising the timeout would "fix" the score without fixing anything real --
it doesn't change whether a future fine-tuned model's predictions are
correct, and it would make Phase 3+ comparisons dependent on a timeout
tuned to make a demo number look better. Instead, the ~99.6% ceiling is
documented as a known, explained property of the environment, and any
future full-oracle run scoring materially below 498/500 (99.6%) indicates
a real regression, not benchmark noise.

### Final Result

After sourcing grading SQL from the archive: **99.6% EX and 99.6% Soft-F1**
on the full 500-example oracle-sanity run (498/500), with the 2 remaining
"failures" fully explained by the timeout limitation above -- not by
source mismatch, not by an alignment bug, not by adapter behavior.

### Decisions and Reasoning

- Official evaluator files are vendored unmodified at a pinned commit and
  imported directly (not copy-pasted or edited), with recorded sha256
  checksums, so upstream scoring semantics are never silently drifted or
  reimplemented.
- A ~96%-instead-of-100% oracle result was treated as a blocking
  correctness question, not an acceptable rounding error -- because an
  evaluation system nobody has stress-tested against its own gold data is
  not trustworthy for judging a future model.

### Issues / Technical Debt

- `evaluation_utils.py` (vendored, unmodified) unconditionally imports
  `psycopg2` and `pymysql` even though LocalSQL only evaluates SQLite;
  both are installed in an optional `eval` dependency group purely so the
  file can be imported.
- The official evaluator's own `compute_acc_by_diff`/`compute_f1_by_diff`
  divide by each difficulty bucket's count, so scoring an arbitrary small
  slice (`--limit`) that happens to omit an entire difficulty tier raises
  a `ZeroDivisionError` inside the unmodified upstream code. LocalSQL's
  adapter catches this and marks EX/Soft-F1 `UNAVAILABLE` with the reason
  rather than crashing or fabricating a score -- callers should pick
  `--limit` values that span all three difficulty tiers to avoid it.

### What We Learned

An evaluation harness's own self-test (oracle sanity) is only meaningful if
someone actually looks hard at *why* it isn't 100%, instead of accepting
"close enough." The 96% result would have been easy to rationalize away
("some queries are just hard to match exactly"); the actual cause (a
silent, upstream two-source divergence) would have quietly under-scored
every future model's oracle ceiling by design, not by model weakness.

### Outcome

**PASSED.** 61 tests passing (Phase 1 + Phase 2), full oracle-sanity result
of 99.6% EX / 99.6% Soft-F1 on the real 500-example benchmark, root cause
of the remaining 0.4% fully explained and documented rather than patched
away.

### What Came Next

Phase 3: before any fine-tuning, establish how the untouched base model
performs on this now-trustworthy benchmark, so fine-tuning has something
real to be measured against.

---

## Phase 3 -- Baseline Inference

**Status: IN PROGRESS -- baseline infrastructure implemented; full GPU
baseline not yet run.**

### Goal

Implement reproducible inference infrastructure for the untouched
`Qwen/Qwen3-4B-Instruct-2507` base model against the Phase 2 gold-free
Mini-Dev manifest, so a human (on a cloud/Kaggle CUDA machine) can produce
a real baseline `predictions.jsonl` and score it with the existing Phase 2
evaluator -- establishing the pre-fine-tuning reference point that a future
QLoRA-tuned model will be compared against.

### Why This Phase Was Necessary

Without a baseline, "the fine-tuned model gets 40% EX" means nothing --
40% could be a huge improvement over an untuned model that barely produces
valid SQL, or barely any improvement at all. The baseline also has to be
run under the *same* 4-bit representation the future QLoRA model will use,
or the comparison would be confounded by a precision change rather than
isolating the effect of fine-tuning.

### What Was Built

- `configs/model.yaml`: centralized model/runtime/generation/prompt
  constants (locked model ID, 4-bit NF4 settings, deterministic decoding,
  context mode, token-profile thresholds).
- `src/localsql/model/`: a config loader (no ML dependencies, always
  importable), a single Qwen backend (lazy torch/transformers/bitsandbytes
  imports, CUDA validation, revision resolution, 4-bit load, deterministic
  generation, decode-new-tokens-only, GPU memory tracking), a generation/
  envelope module (wraps the *existing* canonical LocalSQL prompt as one
  chat user message, applies only the Qwen tokenizer's native chat
  template -- no second Text-to-SQL prompt), and a run-artifacts module
  (resume-safe run directories, provenance recording, no ML dependencies).
- `scripts/run_baseline.py`: reads only the Phase 2 gold-free generation
  manifest (never grading/gold files -- enforced by the manifest's
  Pydantic model rejecting any extra field, and tested directly against
  the script). Supports `--dry-run` (validates manifest/config/resume
  logic with no model or CUDA), `--token-profile` (tokenizer-only prompt
  length statistics), `--limit` (cloud smoke testing), and safe resume
  (a killed session can restart with the same `--run-id` without
  duplicating or losing completed predictions; a *different* configuration
  under the same `--run-id` is refused).
- Output normalization is intentionally minimal: `predicted_sql =
  raw_completion.strip()` -- nothing else. A baseline that produces
  markdown-wrapped or prose-wrapped SQL is a real, visible result.
- `docs/BASELINE.md` documents the full rationale and cloud workflow.

### Decisions and Reasoning

- 4-bit NF4 is the *primary* baseline runtime (not full precision) because
  the future QLoRA model will be trained and run on the same 4-bit base,
  so this isolates the effect of fine-tuning from a precision change.
- The real GPU run is explicitly a human task on a cloud/Kaggle CUDA
  machine -- this development machine (16 GB RAM, 4 GB VRAM) cannot run a
  4-bit 4B-parameter model, and the infrastructure was built and tested
  entirely via `--dry-run` plus fake-backend unit tests, without
  downloading model weights or a tokenizer locally.
- The Hugging Face model commit SHA is resolved and recorded at run time
  rather than silently trusting mutable `main`, so a run's provenance
  identifies exactly which model files produced it.

### Issues / Technical Debt

- The real baseline run, its accuracy numbers, and any token-profile
  results do not exist yet -- this section will be updated once the user
  supplies them.
- The `without_business_context` ablation is architected (
  `resolve_generation_example`) but intentionally not run in this phase,
  to avoid a second 500-example GPU pass before it's needed.

### What We Learned

(To be completed once the real baseline run's results are available.)

### Outcome

**IN PROGRESS.** Infrastructure implemented and tested (91 tests passing,
Phases 1-3 combined, all without network/CUDA/model downloads); no model
has been downloaded or run on this machine; the GPU baseline itself is
pending the user's cloud run.

### What Comes Next

Once the user supplies real baseline `predictions.jsonl` / evaluation
results, this section will be updated with the actual numbers, and only
then would a QLoRA fine-tuning phase be considered.
