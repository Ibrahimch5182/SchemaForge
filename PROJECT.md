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

**Status: COMPLETE -- real Kaggle GPU baseline run (500/500) and official
BIRD Mini-Dev evaluation obtained.**

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

### Two BatchEncoding Bugs Found on Real Kaggle Hardware

Both root causes are the same underlying surprise: the real Qwen tokenizer's
`apply_chat_template(..., tokenize=True, add_generation_prompt=True,
return_tensors="pt")` returns a `transformers.BatchEncoding` -- a dict-like
object with `input_ids` and `attention_mask` keys -- not a bare tensor. This
machine's fake-tokenizer unit tests didn't reproduce that shape until fixed,
which is exactly why the real Kaggle run was still worth doing even after
"passing" local tests.

- **Bug #1 -- token-profiler undercount.** `--token-profile` used
  `len(encoded)` on the `apply_chat_template(...)` result. `len()` on a
  `BatchEncoding` counts its 2 keys, not tokens, so every prompt was
  reported as 2 tokens regardless of actual length. Fixed with an explicit
  `count_input_tokens()` helper (`src/localsql/model/generation.py`) that
  reads the real token dimension whether given a `BatchEncoding`/dict, a
  tensor `[batch, sequence]`, or a plain list. Verified against the real
  corrected profile below.
- **Bug #2 -- generation crash.** `generate_one()` treated the same
  `apply_chat_template(..., return_tensors="pt")` result as a bare tensor:
  `input_ids.shape[-1]` and `model.generate(input_ids, ...)`. Against the
  real tokenizer this raised `AttributeError` (a `BatchEncoding` has no
  `.shape`), failing all 5 examples of the first Kaggle smoke run. Fixed
  by extracting `encoded["input_ids"]` for the token count and unpacking
  the encoding by keyword into `model.generate(**encoded, ...)` (mirrors
  the exact correction validated on Kaggle: smoke-v2 generated 5/5, the
  canonical run generated 500/500). A regression test injects a fake
  `torch` module so the real `generate_one()` code path runs and is
  proven to raise the original `AttributeError` without the fix, and to
  pass with it.

**Lesson**: a fake/mocked tokenizer in local tests can hide a real
library's actual return-type behavior. Both fixes replaced ad hoc
attribute access with either an explicit shape-handling helper or exact
dict-style access -- proven end-to-end on real Kaggle hardware, not just
inferred from documentation.

### Canonical Baseline Run (Kaggle)

**Configuration** (unchanged from `configs/model.yaml`, none of these were
altered to produce this result): `Qwen/Qwen3-4B-Instruct-2507`, resolved
model/tokenizer revision `cdbee75f17c01a7cc42f958dc650907174af0554`; 4-bit
NF4, double quantization, float16 compute, single Tesla T4, batch size 1;
greedy decoding (`do_sample=false`), `max_new_tokens=512`, seed 42,
`context_mode=with_business_context`; `predicted_sql =
raw_completion.strip()` only -- no repair, no cleanup, no fallback.

**Generation result**: 500/500 examples generated, 0 generation failures,
0 OOM failures. Output-format diagnostics over all 500: 500 `status=ok`,
0 markdown fences, 500 start with `SELECT`/`WITH` (500 specifically with
`SELECT`), 0 empty predictions, 0 hit the 512-token generation ceiling,
and `raw_completion == predicted_sql` for all 500 (the untuned model never
produced prose/fences that `.strip()`-only normalization would have had to
pass through visibly broken -- a stronger instruction-following result
than assumed going in).

**Runtime**: total 3045.86s; latency median 5576.76 ms, p90 8890.7 ms,
p95 9981.84 ms, max 34239.53 ms; peak GPU memory 4773.4 MB (of 14911.7 MB
reported total on the T4).

**Corrected prompt-token profile** (full 500, post Bug #1 fix): min 203,
median 930.5, p90 2322.7, p95 2368.05, max 2428; 0 prompts above 4096, 0
above 8192 -- no truncation risk, comfortably informs later training
sequence-length choices. **Output-token profile**: min 9, median 53, p90
88.1, p95 106.05, max 416 (max stays under the 512 ceiling).

**Provenance**: Python 3.12.13, torch 2.10.0+cu128, transformers 5.17.0,
bitsandbytes 0.50.2, CUDA runtime 12.8, GPU Tesla T4.

### Official BIRD Mini-Dev Evaluation of the Untuned Baseline

Scored with the same unmodified, pinned-commit official evaluator and the
same Phase 2 methodology used for oracle sanity -- archive
`mini_dev_sqlite_gold.sql` as canonical grading truth, 30-second timeout
unchanged, EX primary / Soft-F1 secondary:

- **Official EX = 43.6**
- **Official Soft-F1 = 47.6975**
- Phase 2's own oracle ceiling on this environment remains **99.6** (the
  two canonical gold-query 30-second timeouts, `bird-mini-dev-sqlite-0340`
  and `-0393`, are a fixed property of the benchmark/hardware, independent
  of any model).

So the untouched base model resolves roughly 44% of Mini-Dev correctly
end-to-end, despite near-perfect SQL-only instruction-following (500/500
well-formed `SELECT`/`WITH` predictions, 0 markdown, 0 empty) -- meaning
the gap to the 99.6% ceiling is almost entirely query *correctness*, not
output-format violations. That is precisely the kind of headroom a
fine-tuning phase would aim to close, and precisely why this number needed
to exist before fine-tuning was considered.

### Evaluator Dependency Note

Running the official evaluator requires `func_timeout`, `pymysql`, and
`psycopg2-binary` at import time (the vendored, unmodified
`evaluation_utils.py` imports all three unconditionally even though only
SQLite is evaluated here). These are already declared in `pyproject.toml`
as the optional `eval` dependency group (added during Phase 2) --
`uv sync --group eval` installs exactly what's needed, so a fresh
environment does not need ad hoc `uv run --with func-timeout --with
psycopg2-binary --with pymysql` flags. Re-verified working in this phase.

### Issues / Technical Debt

- The `without_business_context` ablation remains architected
  (`resolve_generation_example`) but not run -- no second 500-example GPU
  pass was performed in this phase.
- 43.6% EX is a first data point, not yet decomposed by difficulty/database
  -- that breakdown is available in the Phase 2 evaluator's report output
  (`by_difficulty`/`by_database`) if/when deeper baseline error analysis is
  wanted, but doing that analysis was out of scope for this closeout.

### What We Learned

Two BatchEncoding-shape bugs (profiler undercount, generation crash) only
surfaced against the *real* Qwen tokenizer, not the fake tokenizers used in
local unit tests -- confirming the project's own `--dry-run`-first,
GPU-run-second design was the right call: cheap local tests caught
structural issues (gold isolation, resume logic, contract shape), while
the expensive real run caught library-behavior surprises that no amount of
additional mocking would have found without eventually just running the
real thing. Separately: an untuned 4B instruct model can already follow
the SQL-only output contract almost perfectly while still getting the
*query* wrong most of the time -- output-format compliance and
correctness are genuinely different problems, and this baseline cleanly
separates them for the first time in this project.

### Outcome

**COMPLETE.** Both BatchEncoding bugs (token-profiler undercount,
generation crash) found on real Kaggle hardware, fixed, and covered by
regression tests (97 tests passing, Phases 1-3 combined, all without
network/CUDA/model downloads on this machine). Canonical untuned-baseline
result obtained: 500/500 generated, 0 failures, official EX 43.6, official
Soft-F1 47.6975, against the unmodified Phase 2 evaluator and methodology
(archive-canonical gold, 30s timeout, 99.6% oracle ceiling unchanged).

### What Comes Next

A QLoRA fine-tuning phase, if and when explicitly authorized, now has a
real, trustworthy number to be measured against: 43.6% EX / 47.6975
Soft-F1 from the untouched `Qwen/Qwen3-4B-Instruct-2507` base model under
the exact 4-bit NF4 representation fine-tuning would also use.

---

## Phase 4 -- QLoRA Training Smoke Test

**Status: COMPLETE -- real Kaggle QLoRA smoke run succeeded (20/20 steps,
finite loss, adapter saved and reload-verified). Phase 5 context-length
strategy remains an explicit open decision (see below) -- this phase does
not choose it.**

### Goal

Prove the QLoRA training path (4-bit NF4 base, LoRA adapter, explicit
completion-only loss) works correctly end to end -- on a small subset
and/or a bounded number of optimizer steps on Kaggle/T4 -- before ever
attempting a real, full-length fine-tuning run. This is infrastructure
validation, not the fine-tuning experiment itself.

### Why This Phase Was Necessary

Phase 3 already demonstrated that "should work per the docs" and "actually
works against the real library on real hardware" are different things
(two BatchEncoding bugs, both invisible to local mocked tests). Training
has a much larger blast radius than inference if something is silently
wrong -- an incorrectly masked loss, for instance, could train the model to
predict its own prompt tokens instead of SQL, and that failure mode
produces a loss curve that still looks plausible. A smoke test that proves
the mechanics (masking, 4-bit+LoRA load, gradient step, checkpoint save,
adapter reload) are correct on a handful of examples is far cheaper than
discovering a masking bug after a multi-hour full run.

### Research Question This Phase Sets Up

Can QLoRA specialization of Qwen3-4B materially improve Text-to-SQL
accuracy on completely unseen database schemas (the 7 held-out validation
DBs, and ultimately BIRD Mini-Dev) while remaining practical for local
deployment? Phase 4 does not answer this -- it builds the infrastructure a
later full training run would need to answer it, and proves that
infrastructure isn't silently broken.

### What Was Built

- `configs/train.yaml`: centralized QLoRA smoke-test configuration --
  same locked base model and 4-bit NF4 settings as the Phase 3 baseline
  (so a future base-vs-adapter comparison isolates fine-tuning, not a
  precision change); starting LoRA config (r=16, alpha=32, dropout=0.05,
  all 7 attention/MLP target modules); starting optimization config
  (batch size 1, grad accumulation 8, LR 1e-4, warmup 0.05, gradient
  checkpointing, paged AdamW 8-bit, seed 42); `max_seq_length: 4096`
  explicitly marked provisional, not to be treated as justified until
  reviewed against the real training-prompt token profile.
- `src/localsql/train/`: a config loader (no ML dependencies), `sft_data.py`
  (reuses Phase 1's `train.jsonl` verbatim -- prompt/completion/evidence-
  dropout decisions already baked in, never re-split or re-derived here --
  plus an explicit, unit-tested completion-only label-masking function),
  and `qlora_backend.py` (lazy torch/transformers/peft/bitsandbytes
  imports, 4-bit load, LoRA attach, bounded-step training via plain HF
  `Trainer` fed our own pre-masked labels, adapter save/reload, NaN/Inf
  loss detection).
- **Completion-only masking, explicit and tested**: two separate
  `tokenizer.apply_chat_template` calls (user-only with
  `add_generation_prompt=True`, and the full user+assistant conversation)
  establish the exact prompt/completion boundary; everything before that
  boundary gets label `-100`, everything at or after it (the gold SQL)
  stays trainable. Padding also gets `-100`. Tests prove: prompt tokens
  masked, completion tokens trainable and equal to the real token ids, at
  least one trainable label exists, padding masked, and -- structurally,
  since the prompt-length boundary is computed from a call that is never
  given the completion text at all -- gold SQL cannot leak into what gets
  masked as "prompt".
- `scripts/run_qlora_smoke.py`: `--dry-run` (data/config validation, no
  model/CUDA), `--token-profile` (tokenizer-only profiling of the real
  training prompts *and* full prompt+completion SFT sequence length,
  reported but never used to auto-adjust `max_seq_length`), bounded smoke
  training (`--max-train-examples`, `--max-steps`), and
  `--verify-adapter` (reload the saved adapter onto the base model,
  confirm it's active, run one tiny sanity generation -- explicitly not an
  accuracy evaluation, and BIRD Mini-Dev is never touched in this phase).
- Reused, not reimplemented: the same `build_model_inputs` /
  BatchEncoding-safe envelope construction fixed for the Phase 3 baseline
  backend is reused for the adapter-verification sanity generation, so the
  same `AttributeError` class of bug cannot recur there.
- `pyproject.toml`'s new `train` dependency group adds only `peft` and
  `trl` -- deliberately not re-listing `torch`/`transformers`/`accelerate`/
  `bitsandbytes` (already in the `model` group), so installing it never
  risks reinstalling Kaggle's own torch build.
- **Artifact-contract bug fixed**: the real successful smoke run correctly
  embedded `adapter_verification` inside `summary.json`, but the promised
  standalone `adapter_verification.json` was never written -- the
  post-training verification path only built the dict, it never called a
  writer. Fixed with a single shared `write_adapter_verification()`
  helper, now called from both the post-training path and the standalone
  `--verify-adapter` mode (so they cannot drift apart again), with a
  CPU/offline regression test that doesn't require a new GPU run.

### Decisions and Reasoning

- Plain HF `Trainer`, not `trl.SFTTrainer`, drives the actual training
  loop: completion-only masking is fully computed by `sft_data.py` before
  the trainer ever sees an example, so none of `SFTTrainer`'s own dataset
  formatting/masking conveniences are needed, and its API surface varies
  more across versions than `Trainer`'s. TRL itself is still an installed
  dependency (per the task's request) for any later phase that wants it.
- `max_seq_length` is treated as data, not doctrine: 4096 is a starting
  number, explicitly labeled provisional in both the config file and the
  token-profile report, and this phase deliberately does *not* auto-adjust
  it -- that decision needs the real Kaggle tokenizer's profile of the
  actual 6,067 training prompts (and full SFT sequences including the
  completion) reviewed by a person first.
- Examples whose full SFT sequence exceeds `max_seq_length` are skipped
  and counted, never truncated -- truncating from the right could cut off
  the gold SQL completion itself, which would be far worse than simply
  excluding the example from this smoke run.
- Adapter-reload verification is a plumbing check only, explicitly not an
  accuracy evaluation, and BIRD Mini-Dev is not touched -- consistent with
  Phase 2's evaluation boundary.

### Real Kaggle Environment / Provenance

Python 3.12.13, torch 2.10.0+cu128, transformers 5.0.0, accelerate 1.13.0,
bitsandbytes 0.50.2, peft 0.19.1, trl 1.13.0, CUDA 12.8, GPU Tesla T4
(14911.7 MB total). Model `Qwen/Qwen3-4B-Instruct-2507`, resolved
model/tokenizer revision `cdbee75f17c01a7cc42f958dc650907174af0554` (same
revision as the Phase 3 baseline). QLoRA: 4-bit NF4, double quant, float16
compute, LoRA r=16/alpha=32/dropout=0.05 on all 7 target modules, batch
size 1, gradient accumulation 8, LR 1e-4, gradient checkpointing,
`paged_adamw_8bit`, seed 42, completion-only loss.

### Real Token Profile -- All 6,067 Training Examples

| | prompt-only | full SFT (prompt+SQL) |
|---|---|---|
| min | 509 | 549 |
| median | 2,468 | 2,506 |
| p90 | 7,956.4 | 8,013.8 |
| p95 | 27,890 | 27,927 |
| p99 | 27,918.34 | 27,975 |
| max | 27,995 | 28,082 |
| count > 4096 | 1,397 | 1,398 |
| count > 8192 | 539 | 539 |

Real-Qwen masking/boundary validation (the assumption flagged as unverified
after the infrastructure-only implementation): **0 prefix mismatches**
across all 6,067 examples -- confirms `add_generation_prompt=True` really
does produce an exact token-level prefix of the full sequence on the real
tokenizer, so the completion-only mask boundary is correct in practice, not
just in the fake-tokenizer unit tests. Median SQL completion contributes
only 50 tokens to the sequence, max 216 -- **the long sequences are
schema-driven, not SQL-target-driven.**

**Database concentration of long schemas**: 9 of 62 training databases have
at least one sequence exceeding 4096 tokens; only 2 exceed 8192.
`works_cycles` (n=383, median ~27,948, max 28,082, *all* 383 examples
exceed both 4096 and 8192) and `hockey` (n=156, median ~13,792, max 13,870,
all 156 exceed both) account for the bulk of the extreme tail by
themselves. Also over 4096: `movie_3` (n=223, median 4,314), `mondial_geo`
(n=211, median 6,963), `synthea` (n=141, median 4,865),
`professional_basketball` (n=113, median 8,021, none exceed 8192), `donor`
(n=88, median 4,676), `superstore` (n=82, median 4,246); one `movie_3`-
adjacent DB `world_development_indicators` has only 1 example over 4096.
This is a small number of schema-heavy databases dominating the tail, not
a spread-out long-tail across most databases.

### 1-Step QLoRA Preflight (`qlora-smoke-preflight`)

16 examples, 1 optimizer step: real 4-bit Qwen load succeeded; 33,030,144
trainable / 2,238,840,320 total parameters (LoRA on 7 target modules);
loss 1.5232, finite grad norm (7.31); adapter saved; no OOM. First proof
the mechanics work end to end on real hardware.

### 200-Example Canonical Smoke Subset -- Token Profile

Deliberately chosen to fit comfortably under 4096 for the smoke test:
full SFT sequence min 1,355, median 2,766.5, p90 3,551.2, p95 3,559.15,
p99 3,593, **max 3,593** -- 0 examples above 4096 or 8192. **This 3,593-token
ceiling is a property of the 200-example subset chosen for the smoke test,
not of the training set as a whole** -- it says nothing about whether 4096
or 8192 is an adequate limit for the real 6,067-example training set (see
Phase 5 decision below).

### Initial 20-Step Attempt: CUDA OOM (preserved as evidence, not hidden)

The first attempt at 200 examples / 20 steps, without allocator tuning,
failed with CUDA OOM before completing step 1: T4 total 14.56 GiB,
PyTorch allocated 9.50 GiB, reserved-but-unallocated 3.47 GiB, requested
allocation 1.48 GiB -- a classic allocator-fragmentation OOM (plenty of
*reserved* memory, just not contiguous enough for the requested block),
not an actual capacity shortfall.

### Successful Retry: `PYTORCH_ALLOC_CONF=expandable_segments:True`

Setting `PYTORCH_ALLOC_CONF=expandable_segments:True` (PyTorch's
segment-expansion allocator, which avoids the fragmentation pattern above)
and rerunning the identical 200-example / 20-step configuration
(`qlora-smoke-1-alloc-retry`) succeeded: **20/20 optimizer steps
completed**, all logged losses and gradient norms finite, final
`train_loss = 0.40999391712248323`, `train_runtime = 3097.4346s`
(`runtime_seconds = 3098.01` including setup/teardown), **peak GPU memory
11,550.2 MB** (comfortably under the 14,911.7 MB T4 total once the
allocator issue was resolved), 0 examples skipped for exceeding
`max_seq_length=4096` (the 200-example subset was chosen specifically to
avoid that). Adapter saved successfully.

This is an environment/runtime fix, not a hyperparameter change: LoRA
config, batch size, gradient accumulation, learning rate, and all other
training hyperparameters are unchanged from `configs/train.yaml` between
the failed and successful attempts -- only the CUDA allocator's memory
management strategy changed.

### Adapter-Reload Sanity Check

Reload succeeded: `adapter_active = true`, `adapter_names = ["default"]`.
Sample generation (from the saved adapter, on a real training prompt) --
`SELECT T1.director_name FROM movies AS T1 WHERE T1.movie_title = 'Sex,
Drink and Bloodshed'` -- a well-formed, SQL-only completion. **This is a
plumbing sanity check only, not an accuracy evaluation**; BIRD Mini-Dev was
not touched.

### Issues / Technical Debt

- **Fixed this closeout**: `adapter_verification.json` was not written by
  the post-training verification path (only embedded in `summary.json`) --
  see "What Was Built" above.
- **Transformers 5.x `warmup_ratio` deprecation** (observed during the real
  run): recorded as a Phase 5 compatibility cleanup item. Does not change
  or invalidate the completed smoke results above -- `TrainingArguments`
  still accepted it and warmup behaved as configured.
- **Harmless greedy-generation warning** (temperature/top_p/top_k set but
  unused under `do_sample=False`, observed during the adapter-reload
  sanity check): cosmetic only, does not affect determinism or the sample
  generation's correctness. Cleanup item, not a Phase 4 rerun trigger.
- `paged_adamw_8bit` worked as configured on the real stack -- no fallback
  needed.

### What We Learned

The infrastructure-only implementation's two biggest open assumptions --
whether `add_generation_prompt=True` really produces an exact prefix on
the real Qwen tokenizer (0/6,067 mismatches: yes), and whether the
`Trainer`-based training loop would need a Phase-3-style follow-up bugfix
(no: it ran correctly on the first real attempt once the allocator issue
was resolved) -- both held up. The one real failure mode (CUDA OOM) was an
allocator/fragmentation issue, not a code or hyperparameter bug, and
resolved with an environment variable rather than any change to the
locked QLoRA configuration -- a useful reminder that "the run failed" and
"the configuration is wrong" are not the same thing, and the fix belongs
at the layer where the actual problem lives. Separately: the training-data
token-length tail is concentrated in a handful of schema-heavy databases
(`works_cycles`, `hockey` above all), not spread evenly -- a `max_seq_length`
decision for Phase 5 is really a decision about how to handle *those
specific databases*, not a generic "raise the number" question.

### Outcome

**COMPLETE.** Real Kaggle QLoRA smoke test succeeded end to end: 4-bit
load, LoRA attach, completion-only-masked training (20/20 steps, finite
loss throughout, final loss 0.410), adapter save, and adapter-reload
verification, after resolving one allocator-related OOM via
`PYTORCH_ALLOC_CONF=expandable_segments:True` (no hyperparameter changes).
Real training-prompt token profile obtained and reviewed (6,067 examples).
One artifact-contract bug (missing standalone `adapter_verification.json`)
found and fixed with a regression test (118 tests passing, Phases 1-4
combined). No model was downloaded, trained, or run on this development
machine.

### Phase 5 Context-Length Decision -- Explicitly UNRESOLVED

This is the one substantive open question this phase surfaces and
deliberately does not answer:

- **4096 worked for the bounded Phase 4 smoke test** (the 200-example
  subset was chosen to fit under it) -- that is a statement about the
  smoke test, not a statement about the full training set.
- **4096 is NOT accepted for the Phase 5 full experiment**: it excludes
  1,398 of 6,067 training examples (~23%), concentrated in a handful of
  schema-heavy databases (`works_cycles`, `hockey`, and 7 others).
- **8192 is NOT automatically accepted either**: it still excludes 539
  examples (~8.9%), almost entirely `works_cycles` and `hockey`.
- The long-context / schema strategy for Phase 5 (raise `max_seq_length`
  further? truncate/summarize only the worst-offending schemas? exclude
  `works_cycles`/`hockey` from the full training run and note the
  trade-off? some other approach?) is an explicit **pre-training decision
  for Phase 5**, not decided in this closeout.
- Whatever the eventual strategy, it must **never silently truncate gold
  SQL** (the target itself) and must **never silently discard a database
  group** -- any exclusion must be an explicit, reported, reviewable
  decision, matching this project's established validation philosophy
  from Phase 1 onward.

### What Comes Next

A Phase 5 full-length QLoRA training run requires, first, an explicit
human decision on the context-length/schema strategy above -- informed by
the real token profile and database-concentration data now recorded here,
not by an unreviewed default. Only after that decision is made would
full-length training (not a smoke test) begin.

---

## Phase 5A -- Training Context Policy Analysis

**Status: IN PROGRESS -- analysis and deterministic preprocessing design
complete; NO final Phase 5 training-context configuration has been chosen,
and no training has occurred.**

### Goal

Before the canonical full QLoRA experiment, determine whether the
8K-28K-token schema explosion discovered in Phase 4 (23% of training
examples over 4096 tokens, concentrated in a handful of schema-heavy
databases) can be solved by simply trimming verbose schema formatting --
or whether an actual schema-selection mechanism is required -- without
ever dropping an example, truncating gold SQL, or silently discarding a
database.

### Why This Analysis Was Necessary

Phase 4 established that 4096 excludes ~23% of training examples and 8192
still excludes ~9%, concentrated overwhelmingly in two databases
(`works_cycles`, `hockey`). Jumping straight to "just raise
`max_seq_length` to 28K" would be a real, expensive, hardware-constrained
decision made without knowing *why* those sequences are so long. This
phase answers that "why" with real measurements first.

### Methodology Note: Local Token Estimates vs. Real Kaggle Numbers

This analysis runs entirely offline on this Windows machine -- no model or
tokenizer download, no CUDA, matching every prior phase's constraint.
Token counts here are therefore **estimates**
(`localsql.schema_context.token_estimate`): character count divided by a
calibrated chars-per-token ratio (3.771), derived by comparing this
repository's real canonical schema text against the REAL per-database
median token counts already obtained on Kaggle in Phase 4 for the 8
long-context databases with known numbers. Held-out validation against
the real whole-training-set statistics (not used for calibration): predicted
median 2,723 vs. real 2,506 (+8.7%), predicted min 511 vs. real 549
(-6.9%), predicted max 28,054 vs. real 28,082 (-0.1%). Good enough to rank
representations and estimate magnitudes; **not** a substitute for a real
tokenizer run. `scripts/analyze_schema_context.py` is offline-only in this
phase; a real-tokenizer confirmation run (mirroring Phase 3/4's
`--token-profile` pattern) is a natural, cheap follow-up on Kaggle before
finalizing any Phase 5 number.

### A. Why the Long Prompts Are Large

Inspected the official BIRD schema metadata directly for the 9 databases
Phase 4 flagged: `works_cycles` has **65 tables / 455 columns**; `hockey`
has **22 tables / 300 columns**. These are large, genuinely multi-table
relational schemas (Microsoft's AdventureWorks-style Cycles sample DB and
an NHL statistics DB), not a small schema wrapped in unusually verbose
prose. 451 of `works_cycles`' 455 columns (99%) and 298 of `hockey`'s 300
(99%) carry an AI-generated descriptive comment in the canonical
serializer (Phase 1's `-- description` suffix), and those descriptions
average several hundred characters each -- so a large fraction of the
actual character budget is descriptive text, but the underlying *table and
column count* is also just genuinely large. Both factors matter; neither
alone explains the full picture.

### B. Current vs. Compact-Schema Token Distributions

Representation A = Phase 1's canonical serializer (identifiers + types +
PK/FK + descriptions). Representation B = same identifiers, types, PK, FK
-- descriptions removed, nothing else changed (`localsql.schema_context.
compact_serializer`, a new module; Phase 1's canonical serializer itself
is untouched).

> **Correction (applied in the Phase 5A closeout that follows this
> section):** an earlier version of this table showed `count > 4096 =
> 1,626` and `count > 8192 = 652` for representation A and presented them
> without a clear per-cell "(real)"/"(est.)" label, next to genuinely real
> cells that *did* carry that label -- creating exactly the kind of
> ambiguity this document otherwise tries hard to avoid. **Those two
> numbers were never real**: they were this phase's own local
> character-count estimator's output for representation A's threshold
> counts, which -- unlike the min/median/max figures, whose estimate
> tracked the real numbers to within roughly 1-9% -- diverges much more on
> *threshold-crossing counts* specifically, because a handful of examples
> sit close enough to the 4096/8192 boundary that a ~5-6% per-example
> estimation error (the calibration's own stdev) is enough to move them
> across the line in the estimate but not in reality. The real Phase 4
> Kaggle numbers (`kaggle-phase4-evidence/runs/qlora-smoke-1/
> train_token_profile.json`, tokenizer revision
> `cdbee75f17c01a7cc42f958dc650907174af0554`) are **1,398** (full SFT) /
> **1,397** (prompt-only) for `>4096`, and **539** for `>8192` (both
> representations). The table below is corrected; every cell is now
> explicitly labeled.

Whole training set (6,067 examples, full SFT: prompt + completion):

| | A (canonical) | B (compact) |
|---|---|---|
| min | 549 (real) | ~137 (est.) |
| median | 2,506 (real) | ~485 (est.) |
| p90 | 8,013.8 (real) | ~997 (est.) |
| p95 | 27,927 (real) | ~3,625 (est.) |
| p99 | 27,975 (real) | ~3,671 (est.) |
| max | 28,082 (real) | ~3,745 (est.) |
| count > 3584 | ~1,894 (est. -- not measured by Phase 4) | **383 (est.)** |
| count > 4096 | **1,398 (real)** | **0 (est.)** |
| count > 8192 | **539 (real)** | **0 (est.)** |

Every cell above is now explicitly labeled "(real)" (exact Phase 4 Kaggle
number) or "(est.)" (this phase's calibrated estimate -- see methodology
note above). Representation B has no real-tokenizer numbers at all yet for
any statistic; all of its cells are estimates pending a real Kaggle
profiling run (`scripts/run_qlora_smoke.py --token-profile --input ...`,
documented further down).

### C. Per-Long-DB Before/After (schema-only text, calibrated estimate)

| DB | tables/cols | A chars | B chars | reduction | A real median tok | B est. tok |
|---|---|---|---|---|---|---|
| works_cycles | 65/455 | 104,916 | 13,247 | 87.4% | 27,948 | ~3,513 |
| hockey | 22/300 | 52,700 | 5,358 | 89.8% | 13,792 | ~1,421 |
| professional_basketball | 9/157 | 33,713 | 3,256 | 90.3% | 8,021 | ~863 |
| mondial_geo | 34/139 | 27,281 | 2,832 | 89.6% | 6,963 | ~751 |
| synthea | 11/85 | 18,016 | 1,989 | 89.0% | 4,865 | ~527 |
| world_development_indicators | 6/67 | 17,014 | 1,897 | 88.9% | (1 ex. >4096 under A) | ~503 |
| donor | 4/71 | 17,073 | 1,925 | 88.7% | 4,676 | ~510 |
| movie_3 | 16/89 | 15,447 | 2,064 | 86.6% | 4,314 | ~547 |
| superstore | 6/61 | 15,022 | 1,451 | 90.3% | 4,246 | ~385 |

Removing descriptions alone cuts schema text by **86.6%-90.3%** across
every one of these 9 databases -- a remarkably consistent reduction,
because nearly every column in each of them carries a description of
similar verbosity.

### D. Does Compact-Full-Schema Alone Solve the Issue?

**Almost entirely, yes -- for the 4096 candidate, completely.** Under
representation B, **zero** training examples exceed 4096 or 8192 tokens
(estimated). Only `works_cycles` (383 examples, 100% of that DB) remains
over a **3584** candidate -- every other previously-flagged database,
including `hockey`, drops comfortably under 3584 on compact schema alone
(`hockey`'s estimate: ~1,421 tokens, nowhere near any candidate ceiling).
`works_cycles` remains a special case specifically because 65 tables/455
columns of bare identifiers, types, and FK annotations is, by itself,
already ~3,513 estimated tokens before a single word of the question is
added -- description removal cannot shrink raw identifier/relationship
information below what the schema actually contains.

### E. Schema Budgeting Algorithm (Task 3) -- Used Only for `works_cycles`

Implemented in `localsql.schema_context.relevance`, **inference-time-safe
by construction**: `select_schema_within_budget(schema, question,
business_context, budget, ...)` has no parameter for gold SQL, gold
tables, or gold columns -- there is nothing to accidentally pass gold data
into (verified by a test that inspects the function signature directly).

1. Normalize every table/column identifier (snake_case and camelCase
   aware: `CustomerID` and `customer_id` both normalize to
   `{"customer", "id"}`) and the question + business-context text into
   lowercase word-term sets.
2. Score each table: +2.0 if every term of its name appears in the
   question/context; otherwise partial credit for term overlap, plus a
   smaller contribution for how many of its columns' terms also appear.
   Pure function of (table, terms) -- no randomness, no hash-order
   dependence.
3. Visit tables in descending-score order (ties broken by original schema
   position, never by hash) and greedily add each one **plus its full
   foreign-key closure** (every table transitively reachable via FK
   targets, to a fixed point) if the *whole* resulting schema still fits
   the budget -- otherwise skip and keep trying lower-scored tables.
4. Never partially includes a table (whole table or nothing -- no
   identifier is ever cut mid-string).
5. If literally nothing fits (a pathological tiny budget), falls back to
   the single highest-scored table rather than ever returning an empty
   schema.
6. `length_fn` is pluggable (defaults to `len` for this offline analysis;
   a real tokenizer-based counter can be substituted on Kaggle for exact
   enforcement without changing the algorithm).

Applied to all 383 `works_cycles` examples with a schema-only budget tuned
to ~3,300 estimated tokens: **max estimated full-SFT length 3,512.6 -- all
383 now fit under 3584, zero remain over.**

### F. Gold SQL Coverage Diagnostic (Evaluation Only -- Never Fed Back)

`localsql.schema_context.coverage` parses gold SQL via `sqlglot`
(with join-aware column qualification where possible) **strictly after**
selection, in a module `relevance.py` never imports, purely to measure how
well the gold-free selector performed. Over all 383 budgeted
`works_cycles` examples:

- 383/383 (100%) gold SQL parseable -- 0 unparseable/ambiguous.
- **Table recall: 98.59%** (99.01% at a slightly looser budget).
- **Column recall: 97.78%** (whole-table granularity -- a column counts as
  retained iff its owning table was retained, matching the selector's
  all-or-nothing-per-table design).
- **98.17% of examples retained every gold-referenced table**, and
  therefore every gold-referenced column.

A deterministic, embeddings-free, question-term-matching selector recovers
the correct tables for the actual gold query in ~98% of `works_cycles`
cases without ever seeing the gold query.

### G. Validation-Split Results (Same Logic, Unchanged, Untuned)

The exact same analysis applied to the 534-example / 7-DB held-out
validation split (`authors`, `college_completion`, `craftbeer`,
`image_and_language`, `legislator`, `movie`, `retail_complains`) -- **none
of the rules were tuned using validation gold performance; validation was
only ever *measured*, not fitted.**

- Representation A (canonical): min 589 (est.), median 3,398 (est.), max
  7,836 (est.); **173/534 (32.4%) exceed 4096** -- proportionally *worse*
  than train's 23%, though still well under 8192.
- Representation B (compact): min ~170, median ~469, max ~1,082
  (estimated) -- **zero examples exceed 3584, 4096, or 8192.** No
  budgeting needed for validation at either candidate; none of the 7
  held-out databases are among the long-schema-concentration group.

### H. Candidate 3584 vs. 4096 -- T4 Training-Budget Implications

The real Phase 4 smoke evidence: a 200-example subset with full-SFT max
3,593 tokens completed 20/20 steps at **peak GPU allocation 11,550.2 MB**
of 14,911.7 MB total on a T4 (after resolving an allocator-fragmentation
OOM via `PYTORCH_ALLOC_CONF=expandable_segments:True`, no hyperparameter
change). That is one real data point near the 3584 candidate, not a
controlled sweep of both candidates -- **peak memory at an actual 4096
ceiling has not been measured on real hardware.**

**This is explicitly a statement about the *training* sequence-length
budget on the T4 hardware available right now -- it is NOT a statement
about the eventual product's context capacity.** A production system
serving real users is not restricted to whatever length this phase trains
on; it can use a larger model, better hardware, or a smarter runtime
context strategy (e.g. the very budgeter validated here) independent of
what a T4 smoke test could afford. Conflating "what we can afford to train
on a free/cheap T4 right now" with "what LocalSQL can ever serve" would be
a methodological error this document deliberately avoids.

With representation B: **4096 requires zero budgeting for any of the
6,067 train or 534 validation examples** (cleanest, simplest option, but
untested for T4 memory headroom at that exact length). **3584 requires
budgeting only `works_cycles`** (383/6,067 = 6.3% of train, measured at
~98% gold table/column recall) and has one real (if narrowly-scoped) T4
memory data point near it.

### I. Files Changed/Added

New: `src/localsql/schema_context/{__init__,compact_serializer,
token_estimate,relevance,coverage}.py`; `scripts/analyze_schema_context.py`;
`tests/schema_context/{__init__,fixtures,test_compact_serializer,
test_relevance,test_coverage,test_derived_dataset}.py`. Changed:
`.gitignore` (new `data/processed_context_budgeted/` rule, same pattern as
other derived-artifact directories). Generated (gitignored, reproducible):
`data/reports/schema_context_{train,validation}_b{3584,4096}.json`,
`data/processed_context_budgeted/{train,validation}.jsonl` +
`*_provenance.json` (Task 7 derived-dataset prototype -- Phase 1's
`data/processed/{train,validation}.jsonl` verified byte-unchanged: the
provenance file's `original_file_sha256` matches the exact hash already
recorded in Phase 4's `run_config.json`).

### J. Tests

`uv run pytest -q` -- **149 passed** (118 previous + 31 new), run once, all
CPU/offline, no network/CUDA/model. Coverage: deterministic output (repeat
calls produce identical selections), no-gold-parameter enforced by
signature inspection, no completion/target mutation, no example dropping,
PK/FK preservation, FK-closure correctness (including multi-hop chains),
token-budget enforcement without partial-table truncation, never-empty
fallback for pathological budgets, full-schema-under-budget left unchanged,
train/validation db isolation preserved through the derived-dataset writer,
and the gold-coverage diagnostic's parseable/unparseable/full-vs-partial-
retention accounting.

### K. Candidate Policy Finalization (Supersedes "Recommended... Not
Automatically Adopted" Above)

After reviewing the analysis above, the question-conditioned schema
budgeter (section E) was **evaluated and explicitly NOT selected** for the
candidate training dataset. Reasons: its own measured gold-table/column
retention was 98.17%, not 100% (a real, if small, information-loss risk);
compact full-schema alone already eliminates the 4096/8192 problem
entirely; and introducing selection complexity is unjustified when a
strictly simpler policy suffices. **The selector code
(`localsql.schema_context.relevance`) remains in the repository as
documented research tooling -- it is not deleted, and nothing about
sections E/F above is retracted -- it is simply not the chosen policy.**

**Selected candidate policy: adaptive per-database full-schema compaction**
(`localsql.schema_context.db_policy`, `scripts/build_phase5_candidate.py`):

1. **Train**: a database is compacted (compact full-schema serializer,
   every table/column/PK/FK preserved, no question-conditioning, no gold
   SQL) for ALL its examples if the REAL Phase 4 Kaggle tokenizer profile
   showed at least one of its examples exceeding 4096 tokens. Every other
   database is left completely unchanged from Phase 1's canonical
   `data/processed/train.jsonl`.
2. **Validation**: Phase 4 never profiled validation with the real
   tokenizer, so the same per-database rule is applied using the local
   character-count estimator's canonical (representation A) full-SFT
   length only -- never gold correctness, never tuned against validation
   accuracy.

**Verified train counts** (computed directly from `data/processed/
train.jsonl`'s real `db_id` field -- exact counting, no estimation
involved): **9 databases / 1,502 examples compacted**, **53 databases /
4,565 examples unchanged** -- confirms the counts given at the start of
this task exactly. Compacted: `donor` (88), `hockey` (156), `mondial_geo`
(211), `movie_3` (223), `professional_basketball` (113), `superstore`
(82), `synthea` (141), `works_cycles` (383), `world_development_indicators`
(105).

**Verified validation counts** (local-estimate-based, per-database, every
example in the flagged databases estimated over 4096 -- not a borderline
mixed case): **2 databases / 173 examples compacted** (`college_completion`:
45, `legislator`: 128), **5 databases / 361 examples unchanged**
(`authors`, `craftbeer`, `image_and_language`, `movie`,
`retail_complains`).

**Candidate artifacts** (gitignored, reproducible via
`scripts/build_phase5_candidate.py`): `data/processed_phase5_candidate/
{train,validation}.jsonl` (6,067 / 534 rows respectively) and `policy.json`
(full decision record + provenance). Original-file-unchanged verification:
the script re-hashes `data/processed/{train,validation}.jsonl`
immediately before and after writing candidate output, and the recorded
`train_sha256` (`81c29e14e3b...`) matches, digit for digit, the hash
already recorded independently in Phase 4's `run_config.json` -- proof the
original was never touched, from two unrelated points in the project's
history.

**Structural preservation**: every compacted database's serialization is
checked programmatically, per-table and per-column (not just "the schema
text somewhere contains the identifier" -- scoped to that table's own
block, since large schemas like `works_cycles` reuse column names like
`id` across many tables), against the real BIRD schema metadata: every
table present, every column present, every primary-key annotation present
on the correct column's line, every foreign-key annotation present on the
correct column's line. Passed for all 9 databases including
`works_cycles`' 65 tables / 455 columns.

**Gold/completion immutability**: the candidate builder asserts, for every
one of the 6,601 combined train+validation examples, that `completion`
is byte-identical to the original; for unchanged-policy examples, that
`prompt` and `serialized_schema` are also byte-identical; that every
`example_id` from the original appears exactly once in the candidate (no
duplication, no loss); and that the `db_id` set is unchanged (train/
validation isolation preserved). No function anywhere in this pipeline
accepts gold SQL as an input to a decision.

### L. Real-Tokenizer Kaggle Profiling (Generalized)

`scripts/run_qlora_smoke.py --token-profile` now accepts `--input <path>`
for ANY JSONL with `example_id`/`prompt`/`completion` fields -- Phase 1's
`train.jsonl`/`validation.jsonl`, or the new
`data/processed_phase5_candidate/{train,validation}.jsonl` -- not just the
hardcoded training file. Reports full-SFT and prompt-only distributions
(count/min/median/p90/p95/p99/max, counts `>3584`/`>4096`/`>8192`) with the
real Qwen tokenizer, and **asserts the real prompt/full prefix boundary**
per example (the same check Phase 4 validated at 0/6,067 mismatches) --
flagging (not silently ignoring) any example where the assumption
completion-only masking depends on fails to hold. This has NOT been run
yet -- it requires the real tokenizer on Kaggle, which this phase does not
download or run locally.

```powershell
uv sync --group model --group train
uv run python scripts/run_qlora_smoke.py --run-id phase5-candidate-profile \
  --input data/processed_phase5_candidate/train.jsonl --token-profile
uv run python scripts/run_qlora_smoke.py --run-id phase5-candidate-profile \
  --input data/processed_phase5_candidate/validation.jsonl --token-profile
```

### M. Longest-Example Certification Manifest Support

The same `--token-profile` command now also writes a deterministic
`<profile_name>_longest_examples.json` manifest -- the top N (default 50,
`--top-n-manifest`) examples ranked by real full-SFT token count, ties
broken by `example_id` -- for a **later, separate** GPU memory
certification run against the actual longest candidate examples. **That
certification run is not performed in this phase.**

### Phase 5 Candidate Policy -- max_seq_length = 4096 (Real-Tokenizer Confirmed)

Adaptive per-database full-schema compaction (section K) + candidate
`max_seq_length = 4096`. As of Phase 5B, gate 1-2 below are **PASSED**
with real Kaggle-tokenizer evidence (tokenizer revision
`cdbee75f17c01a7cc42f958dc650907174af0554`); gate 3 (GPU memory
certification) is a separate, not-yet-run empirical step -- see
"Phase 5B" below.

1. The candidate dataset (`data/processed_phase5_candidate/*.jsonl`) was
   profiled with the real resolved Qwen tokenizer (section L's command).
   **PASSED.**
2. That profiling showed **zero** candidate examples exceeding 4096
   tokens for real: TRAIN (6,067 examples) full-SFT max = 4,005 tokens,
   `>3584` = 331, `>4096` = 0, `>8192` = 0, prefix mismatches = 0.
   VALIDATION (534 examples) full-SFT max = 3,581 tokens, `>3584` = 0,
   `>4096` = 0, `>8192` = 0, prefix mismatches = 0. **PASSED** -- real,
   not estimated.
3. A longest-example T4 memory certification (section M's manifest) --
   **NOT YET RUN**. See "Phase 5B" below.

The 3584 candidate + question-conditioned selector analysis (sections
D-H) remains documented as an **evaluated alternative**, not the selected
candidate -- kept for the record, not deleted, not acted on further.

**Hardware/wall-clock reality check**: Phase 4's real smoke throughput
(200 examples, 20 steps, 3,098 seconds) extrapolates to roughly 4+ hours
just to complete a single epoch's worth of *optimizer steps* over a subset
this size on a single T4 -- full canonical training (6,067 examples,
multiple epochs, much longer average sequence length even after
compaction) is likely **wall-clock impractical on a single free/cheap T4**.
A representative-sample throughput measurement (Phase 5B, not yet run)
will replace this extrapolation with a real estimate before any hardware
decision is made.

## Phase 5B -- GPU Certification, Throughput Benchmark, and Crash-Safe
Resume Infrastructure (IN PROGRESS)

### Goal

Build the minimum infrastructure to answer, empirically and before any
full canonical training run: (1) can the longest real candidate examples
(~4,005 tokens) train safely on one T4 under the exact intended QLoRA
config, (2) what is a realistic full-training wall-clock estimate from a
representative throughput sample, and (3) can training be stopped and
resumed from a checkpoint without restarting. **No full canonical
training, no BIRD Mini-Dev evaluation, and no Phase 6 work occur in this
phase.**

### Certification Gate Status

- **A. Real tokenizer context certification -- PASSED.** See numbers
  above; sourced from `kaggle-phase5-tokenizer-evidence/` (read-only,
  unmodified).
- **B. GPU memory certification -- NOT YET RUN.** Requires a real Kaggle
  T4 GPU. Command below.
- **C. Throughput certification -- NOT YET RUN.** Requires a real Kaggle
  T4 GPU. Command below.
- **D. Stop/resume certification -- NOT YET RUN.** Requires two separate
  Kaggle process invocations (RUN A, RUN B). Commands below.
- **E. Canonical full training -- NOT STARTED.** Blocked on B-D above
  and an explicit hardware/session-strategy decision.

No GPU certification success is claimed here until Kaggle actually
produces it -- this section records infrastructure only.

### Reliability Correction Pass (superseding the initial Phase 5B build)

The initial Phase 5B implementation below was accepted except for two
issues corrected before commit:

1. **Estimator-based throughput sampling was replaced.** The original
   `throughput_sample_64` used 8 real long-tail examples plus examples
   ranked by the local character-count estimator
   (`localsql.schema_context.token_estimate`). This is **not accepted**
   for the canonical benchmark -- the estimator has known material
   threshold-count error versus the real Qwen tokenizer, and the
   throughput benchmark drives a hardware/runtime decision. It is
   replaced by a **real-token-manifest-only** pipeline: `--token-profile`
   now also writes a per-example real token-length manifest
   (`<profile>_token_lengths.jsonl` -- `example_id`, `db_id`,
   `representation`, `prompt_only_token_count`, `full_sft_token_count`,
   `completion_token_count`, `prefix_boundary_match`, all from the same
   resolved-tokenizer/chat-template path already used for the approved
   Phase 5B profile), and the canonical throughput sample is now 64
   deterministic, approximately-equally-spaced ranks (round-half-up rule)
   over ALL profiled examples sorted by `(full_sft_token_count,
   example_id)` -- no estimator, no gold SQL, no random sampling, no
   hand-picked substitutions. A representation gap in the selected sample
   is a **BLOCKER** (`ThroughputSampleRepresentationError`), never
   silently patched. The stale estimator-built `throughput_sample_64.*`
   files were deleted; regenerating them now requires the real manifest
   (see "Exact Kaggle Commands" below) -- `build_certification_sets.py`
   BLOCKERs with the exact next command if it's missing.
2. **Durable checkpoint export was hardened to fail closed.**
   `validate_checkpoint_completeness` now requires every HF Trainer/PEFT
   resumable-state category -- model/LoRA adapter state, optimizer state,
   LR scheduler state, `trainer_state`/global step, RNG state, training
   arguments -- to actually be present (tolerating filename variants like
   `adapter_model.safetensors`/`.bin`, `optimizer.pt`/`.bin`) before
   `build_export` creates ANY export directory; a missing category raises
   `IncompleteCheckpointError` and writes nothing. The export also now
   packages the **exact training-data JSONL** a run used
   (`training_data.jsonl`, byte-for-byte, SHA256-verified against
   `run_config.json`'s `train_file_sha256` via
   `verify_training_data_snapshot` -- `TrainingDataMismatchError`,
   fail-closed, on any mismatch), plus environment/source provenance
   (Python/torch/transformers/peft/bitsandbytes/accelerate/trl versions,
   CUDA/GPU info, resolved model/tokenizer revision, and a source-code
   revision resolved via `localsql.train.provenance.resolve_source_revision`
   -- explicit `--source-revision` > a `SOURCE_REVISION` file at the repo
   root > best-effort `git rev-parse HEAD` -- since `git archive`/Kaggle
   upload strips `.git` and would otherwise silently lose the commit).

### What Was Built

1. **Longest-16 certification set** (`scripts/build_certification_sets.py`,
   `src/localsql/train/certification.py`): exact copies of the 16
   real-tokenizer-longest candidate training examples (per the Kaggle
   `train_longest_examples.json` manifest, rank order already
   deterministic), for worst-case memory certification. Real length range:
   3,939.0-4,005.0 tokens. No re-ranking, no mutation, no truncation.
   SHA256-tracked against both the source candidate file and the source
   manifest. Methodology unchanged by the reliability correction.
2. **Real per-example token-length manifest + canonical throughput
   sample** (`--token-profile`'s `write_token_length_manifest`,
   `localsql.train.certification.build_real_token_throughput_sample`/
   `select_equally_spaced_by_real_rank`): see "Reliability Correction
   Pass" above. Not yet regenerated against the real candidate-dataset
   manifest -- that Kaggle run has not happened yet (see "What Comes
   Next").
3. **Checkpointing** (`src/localsql/train/qlora_backend.py`,
   `scripts/run_qlora_smoke.py`): `Trainer`'s own full-state periodic
   checkpointing (`save_steps`/`save_total_limit`), exposed as explicit
   CLI flags, plus explicit (never auto-discovered) `--resume-from-checkpoint`.
   `run_config.json`/`summary.json` record `checkpoint_dir`, `save_steps`,
   `save_total_limit`, `resume_from_checkpoint`, `starting_global_step`,
   the final `global_step`, and (reliability correction) `source_revision`/
   `source_revision_origin` and `accelerate_version`/`trl_version` in
   `summary.json`'s provenance block, so a resume/export is always
   independently verifiable from the run artifacts alone.
4. **Durable checkpoint export, now fail-closed**
   (`scripts/export_checkpoint.py`,
   `src/localsql/train/checkpoint_export.py`): see "Reliability
   Correction Pass" above -- validates checkpoint completeness and
   packages the exact training-data snapshot + full provenance before
   writing anything, still explicitly excluding full base-model weights
   (allow-list + size-cap safety net).

### Blocking Integration Bug -- `--input` Never Reached Training (Kaggle-Discovered, Fixed)

The first real Kaggle attempt at the memory-certification command failed
before model loading:

```
BLOCKER: training file not found: /kaggle/working/localsql/data/processed/train.jsonl
```

**Root cause**: `main()`'s actual training path called `load_examples(cfg,
REPO_ROOT, args.max_train_examples)` directly -- unconditionally
resolving `configs/train.yaml`'s `data.train_file`
(`data/processed/train.jsonl`) and never even looking at `args.input`.
Only the `--token-profile` branch had ever been wired to honor `--input`;
the actual training/certification path was silently stuck on the Phase 1
default the entire time. **Fix**: both branches now call a single new
`resolve_train_examples(cfg, repo_root, explicit_input, limit)` --
`--input`, when given, is validated (BLOCKER on that exact path if
missing) and loaded BEFORE `cfg.data.train_file` is ever touched, for
BOTH `--token-profile` and real training; without `--input`, behavior is
byte-for-byte unchanged. The resolved path flows unchanged into
`run_smoke_training`'s `train_path` parameter, so `run_config.json`'s
`train_file`/`train_file_sha256` -- and therefore
`scripts/export_checkpoint.py`'s training-data snapshot, which reads
those fields -- automatically reflect the override with no further
changes needed. `configs/train.yaml` was not touched, and no
certification JSONL was copied/renamed into `data/processed/train.jsonl`.
See `tests/train/test_runner_contracts.py` (override bypasses default,
missing-input BLOCKERs on that path, no-input behavior preserved,
structural guard against a future edit re-splitting the two branches) and
`tests/train/test_checkpoint_export.py`'s end-to-end
`test_export_checkpoint_script_snapshots_the_override_train_file_from_run_config`.

### Exact Kaggle Commands (not yet executed)

GPU memory certification (16 longest examples, exact intended config;
`configs/train.yaml` already carries batch=1/grad-accum=8/LoRA
r16-alpha32-dropout0.05/max_seq_length=4096/gradient checkpointing):

```
export PYTORCH_ALLOC_CONF=expandable_segments:True
uv run python scripts/run_qlora_smoke.py --run-id phase5b-mem-cert \
    --input data/certification/longest_16.jsonl --max-steps 2
```

(16 examples, batch 1, grad-accum 8 -> 16 microbatches / 2 optimizer
steps, matching the task's expectation.)

Throughput benchmark -- now a TWO-STEP process (reliability correction):
first produce the real per-example token-length manifest, then build the
canonical sample from it, then run the sample:

```
# Step 1: real per-example token-length manifest over the full candidate
# training set (writes data/runs/phase5-candidate-profile/train_token_lengths.jsonl;
# CPU-only for the manifest math, but needs the real tokenizer -- run
# alongside the model/train groups on Kaggle).
uv run python scripts/run_qlora_smoke.py --run-id phase5-candidate-profile \
    --input data/processed_phase5_candidate/train.jsonl --token-profile

# Step 2 (offline, no CUDA needed -- copy train_token_lengths.jsonl to
# kaggle-phase5-tokenizer-evidence/phase5-candidate-train-profile/ first,
# or pass --token-lengths-manifest pointing at it directly):
uv run python scripts/build_certification_sets.py

# Step 3: the actual throughput benchmark run.
export PYTORCH_ALLOC_CONF=expandable_segments:True
uv run python scripts/run_qlora_smoke.py --run-id phase5b-throughput \
    --input data/certification/throughput_sample_64.jsonl --max-steps 8 \
    --save-steps 4 --save-total-limit 1
```

Stop/resume certification -- two SEPARATE process invocations, two
distinct `--run-id`s (this runner refuses to overwrite a run-id that
already has a `summary.json`, so RUN B must be a new run-id resuming from
RUN A's checkpoint, not the same run-id):

```
# RUN A -- train to a known step, save a checkpoint, exit.
uv run python scripts/run_qlora_smoke.py --run-id phase5b-resume-a \
    --input data/certification/longest_16.jsonl --max-steps 2 \
    --save-steps 1 --save-total-limit 2

# RUN B -- a NEW process, explicit resume from RUN A's checkpoint,
# continuing to a HIGHER global_step.
uv run python scripts/run_qlora_smoke.py --run-id phase5b-resume-b \
    --input data/certification/longest_16.jsonl --max-steps 4 \
    --save-steps 1 --save-total-limit 2 \
    --resume-from-checkpoint data/runs/phase5b-resume-a/checkpoint/checkpoint-2
```

Verification (from RUN B's `summary.json`): `starting_global_step > 0`,
final `global_step > starting_global_step`, `resumed_from_checkpoint`
records RUN A's checkpoint path, and `adapter_verification` shows a valid
reloaded adapter -- proving optimizer/scheduler/model state was actually
restored, not re-trained from scratch inside one continuous process.

Durable export, after any of the above:

```
uv run python scripts/export_checkpoint.py --run-id phase5b-resume-b
```

### Full-Run Session Safety Design (Documentation Only -- Not Executed)

This is a design for later multi-session Kaggle training, not something
run in this phase:

- Keep each Kaggle session comfortably below the platform's max session
  length (leave real margin, not just under the hard cap), so a session
  ends on a deliberate checkpoint/export rather than a forced kill
  mid-step.
- Checkpoint frequently (`--save-steps` set to a cadence measured in
  minutes, not epochs, once Phase 5B's throughput numbers are real) so
  the worst-case lost work from an unplanned interruption is small.
- Before a planned session end, run `scripts/export_checkpoint.py` to
  produce a durable, portable package -- never rely on
  `/kaggle/working` (or any single ephemeral VM path) surviving past the
  session.
- The next session resumes explicitly from that exported checkpoint
  (`--resume-from-checkpoint`, a new `--run-id`) -- never an implicit or
  auto-discovered resume.
- If a weekly compute quota is exhausted mid-training, preserve the
  latest durable export and wait for quota renewal, or move to different
  already-available compute -- **not** a workaround for the quota itself
  (e.g. rotating accounts). This project uses one Kaggle account's
  legitimate free GPU quota only.
- No final hardware choice is made here -- that decision comes only after
  Phase 5B's real throughput measurement (gate C above) is in hand.

### Tests

CPU/offline only (no CUDA, no model download):
`tests/train/test_certification.py`,
`tests/train/test_checkpoint_export.py`,
`tests/train/test_provenance.py`, plus new cases in
`tests/train/test_runner_contracts.py`. Cover deterministic
longest-16/real-token-manifest throughput-sample selection (including the
round-half-up rank formula, exactly-64-unique on a 6,067-scale fixture,
representation-gap fail-closed behavior, and no estimator dependency),
byte-identical record copies, checkpoint CLI/config wiring, explicit
(never auto-discovered) resume path, run metadata capturing the parent
checkpoint, source-revision resolution priority, and the durable export's
required-state validation (one rejection test per missing category),
training-data-snapshot verification, and manifest/provenance/allowlist/
exclusion logic. Full suite (248 tests): `uv run pytest -q`.

### What Comes Next

Run the real-tokenizer profiling command against the full candidate
training set (produces the real per-example token-length manifest this
correction pass requires), rebuild the canonical throughput sample from
it, then run the three Kaggle commands above (B, C, D) on real T4
hardware, review the real memory/throughput/resume evidence, and only
then decide hardware scaling and a full-training session strategy. Full
canonical training and Phase 6 remain explicitly out of scope until that
decision is made.

## Phase 5C -- Canonical Resumable Full-Training Runner

**No hyperparameters, dataset policy, split, base model, tokenizer/
revision policy, quantization, checkpoint format, or durable export
format changed. No training was run.** This phase adds the infrastructure
to actually RUN the canonical 2-epoch experiment across one or more
Kaggle sessions once B-D above are satisfied.

**Critical requirement**: a multi-session run must be mathematically
equivalent to one uninterrupted `num_train_epochs=2` run. Implemented by
never touching `TrainingArguments.max_steps`/`num_train_epochs` between
sessions (`num_train_epochs` is always passed, `max_steps` is never used
for this path) and adding a session-boundary mechanism
(`--stop-after-global-step`) via a `TrainerCallback` that only sets
`control.should_training_stop`/`should_save` -- it never touches the
scheduler's total-step horizon.

**What was built**:
- `QLoraBackend.train_smoke` (`src/localsql/train/qlora_backend.py`)
  extended with `num_train_epochs` (mutually exclusive with `max_steps` --
  passing both would let `TrainingArguments.max_steps` silently override
  `num_train_epochs`) and `stop_after_global_step`, delegating its
  stop/force-save decision to `StopAfterGlobalStepState`
  (`src/localsql/train/full_training.py`, CPU/offline-testable
  independent of transformers/CUDA).
- `src/localsql/train/full_training.py`: `compute_optimizer_step_schedule`
  (mirrors `Trainer`'s own step-counting formula --
  `ceil(ceil(N/batch)/grad_accum) * epochs`), `validate_resume_compatibility`
  / `ResumeCompatibilityError` (compares canonical settings between a
  resume request and the checkpoint's source `run_config.json`, only
  flagging fields present-and-different on both sides), and
  `StopAfterGlobalStepState`.
- `scripts/run_qlora_full_training.py`: a distinctly named canonical
  full-training runner (not another smoke test), defaulting `--input` to
  `data/processed_phase5_candidate/train.jsonl` (the canonical 6,067-example
  set; `validation.jsonl` is recorded for provenance only -- no
  generation- or loss-based validation subsystem is introduced here).
  Reuses `QLoraBackend` for all training logic. `run_config.json`/
  `summary.json` use the SAME field names `scripts/export_checkpoint.py`
  already reads (`train_file`, `train_file_sha256`, `resolved_revision`,
  `max_seq_length`, `save_steps`, `save_total_limit`,
  `resume_from_checkpoint`, `source_revision`/`source_revision_origin`,
  and a `provenance` block matching Phase 5B's), so full-training runs
  are exportable via the existing, unmodified `export_checkpoint.py`.
  Additionally records `canonical_num_train_epochs`,
  `canonical_total_optimizer_steps`, `expected_steps_per_epoch`,
  `requested_stop_after_global_step`, `starting_global_step`,
  `final_global_step`, and `session_end_reason`
  (`"planned_boundary"` / `"canonical_completion"`).
- Resume-compatibility validation runs BEFORE any GPU work: if
  `--resume-from-checkpoint` is given and the checkpoint's parent run's
  `run_config.json` is found, `model_id`/`max_seq_length`/`quantization`/
  `lora`/`canonical_num_train_epochs`/`canonical_total_optimizer_steps`/
  `train_file_sha256` must match or the run BLOCKERs and exits -- never
  silently continues under different settings. Missing source
  `run_config.json` (nothing to validate against) is a proceed-with-note,
  not a BLOCKER.

**Expected canonical schedule (real dataset, computed from the actual
implementation)**: 6,067 training examples, batch=1, grad-accum=8 ->
**759 optimizer steps/epoch**, **1,518 total optimizer steps for 2
epochs** (verified live via `--dry-run` against the real
`data/processed_phase5_candidate/train.jsonl`).

**Tests**: `tests/train/test_full_training.py` (schedule math incl. the
real 759/1,518 numbers, `StopAfterGlobalStepState` boundary/force-save
decision, `validate_resume_compatibility` precedence/field coverage),
`tests/train/test_full_training_runner_contracts.py` (BLOCKER paths for
overwrite/missing-resume-path/out-of-range or non-positive
`--stop-after-global-step`/canonical-setting mismatch, proceed-with-note
on a missing source `run_config.json`, structural guards that
`max_steps` is never passed to the backend and that
train-file-SHA256/provenance fields are recorded). CPU/offline, no
CUDA/transformers needed (the mutual-exclusivity check in
`train_smoke` was deliberately placed before its heavy imports so it
stays testable on a machine without the `model`/`train` dependency
groups installed at all). Full suite (303 tests): `uv run pytest -q`.
No regressions in Phase 4/5B smoke/certification/export tests.

**Exact commands** (not yet run):

```
# Session 1
uv run python scripts/run_qlora_full_training.py --run-id phase5c-session-1 \
    --save-steps 100 --save-total-limit 3 --stop-after-global-step 400 \
    --source-revision <commit>

# Session 2 (resume, same fixed 2-epoch horizon)
uv run python scripts/run_qlora_full_training.py --run-id phase5c-session-2 \
    --save-steps 100 --save-total-limit 3 --stop-after-global-step 800 \
    --resume-from-checkpoint data/runs/phase5c-session-1/checkpoint/checkpoint-400 \
    --source-revision <commit>
```

**What Comes Next**: Phase 5B's B/C/D certification gates must pass on
real Kaggle hardware first; only then does an actual canonical training
session start, using the commands above.

## Phase 5D -- Fine-Tuned Evaluation Runner

**Canonical training is COMPLETE**: the Phase 5C canonical full-training
run finished at **1518/1518 optimizer steps (2 epochs)** on training data
sha256 `e23a97ea746cef24b17f6bea8dc8440ab96313798837033ec76af9ca79830196`
(source revision `e173c235dcfee41868010de68db471616c92db36`), producing
adapter checkpoint `checkpoint-1518` (`adapter_model.safetensors` +
`adapter_config.json` + full resumable checkpoint state), exported and
durable per Phase 5B's export infrastructure.

**What was built** (evaluation-path plumbing only -- no training code
touched): `QwenBackend.load()` (`src/localsql/model/qwen_backend.py`)
gained an optional `adapter_path` parameter, default `None`, in which case
behavior is unchanged from the Phase 3 baseline path (same base-model
kwargs, no `peft` import at all). When `adapter_path` is supplied, the
identical NF4 base model is loaded first, then wrapped via
`PeftModel.from_pretrained(..., is_trainable=False)` (never
`merge_and_unload()`), mirroring the pattern already used for Phase 4's
`load_adapter_for_verification`. A new `validate_adapter_path()` fails
closed (missing directory / missing `adapter_config.json` or
`adapter_model.safetensors`) before any torch/transformers/CUDA work, so a
bad `--adapter` path is rejected the same way whether or not the optional
"model"/"train" dependency groups are even installed. `generate_one()` is
untouched -- one generation implementation, reused as-is.

`scripts/run_finetuned.py` is a new, separate runner (never modifies
`run_baseline.py`), structurally reusing the Phase 3 machinery unchanged:
the same `GenerationExample`, `resolve_generation_example()`,
`generate_one_example()`, `RunDirectory`, Phase 2 `predictions.jsonl`
contract, `model.yaml` generation config, and unmodified
`normalize_predicted_sql()` (whitespace-trim only). CLI:
`--manifest --run-id --adapter [--model-config] [--limit] [--context-mode]
[--dry-run]`. Fail-closed adapter validation runs before the manifest is
even read. Per-example resume works identically to `run_baseline.py`
(same `RunDirectory`/`generations.jsonl`/`rewrite_predictions` machinery),
so a Kaggle session interruption mid-run is restartable under the same
`--run-id`. Resume-safety against accidentally mixing a baseline and an
adapter run under the same `--run-id`, or two different adapters, is
achieved without touching the shared `RunConfig` schema: the run's
`model_id` is tagged `"<base_model_id>+lora:<resolved adapter path>"`
(`adapter_model_id()`), which both appears in `predictions.jsonl` and
participates in `RunConfig.matches()`'s existing identity check. Provenance
recorded in the run's `summary.json` (no shared schema changes): base
model id, resolved base revision, adapter path, adapter weights SHA256,
`adapter_active`, context mode, manifest SHA256, generation config,
quantization config -- so this run can never be mistaken for the Phase 3
baseline result.

**Tests** (`tests/model/test_qwen_backend.py`,
`tests/model/test_run_finetuned_contracts.py`): baseline `QwenBackend`
behavior unchanged when `adapter_path=None` (asserted via a fake
torch/transformers/bitsandbytes/huggingface_hub stack injected into
`sys.modules`, with `peft` deliberately absent to prove it's never
imported on that path); missing/incomplete adapter directories fail
closed with a clear `AdapterValidationError` before any model load;
adapter wrapping never calls `merge_and_unload()`; the fine-tuned runner
consumes the same gold-free manifest contract and rejects a gold-bearing
row exactly like `run_baseline.py`; generation is proven to go through
the shared `generate_one_example()` (no second implementation);
`--dry-run` requires no CUDA/model deps; provenance (`model_id` tagging)
is proven to differ between a baseline-shaped and an adapter-shaped
`RunConfig`. Full suite: 318 tests, all green, no regressions.

**[Historical -- as of the end of Phase 5D; superseded by Phase 6 below] Real BIRD Mini-Dev fine-tuned result: NOT YET RUN.** The Phase 3
baseline (untouched, still the only real number on record) remains
**official EX 43.6%, Soft-F1 47.6975%** (500/500 generated, 0 failures).
Do NOT claim improvement -- or any result at all -- for checkpoint-1518
until `scripts/run_finetuned.py` is actually executed on real Kaggle GPU
hardware against the full 500-example manifest with
`--context-mode with_business_context`, and the resulting
`predictions.jsonl` is scored with the existing, unmodified
`scripts/evaluate_bird_minidev.py`.

**What Comes Next**: run `scripts/run_finetuned.py` on Kaggle against
`checkpoint-1518` with `--context-mode with_business_context` over the
full 500-example Mini-Dev manifest, then score with
`scripts/evaluate_bird_minidev.py` -- this is the final Phase 5 gate.

## Phases 6-12 -- Summary Addendum (added at Phase 12)

The detailed per-phase write-ups for Phases 6-11 live in their own documents rather than in
this journal; this addendum closes the gap so the journal is not left stating that the
fine-tuned result is "not yet run".

- **Phase 6 -- held-out and external evaluation (frozen results).** Base vs `checkpoint-1518`:
  seen-training normalized SQL exact match 4.20% -> 37.40% (+33.20 pp, n=500; specialization,
  not generalization); schema-held-out execution accuracy 39.14% -> 44.57% (+5.43 pp, n=534,
  7 unseen DBs, SchemaForge SQLite comparator, NOT the official evaluator); untouched BIRD
  Mini-Dev EX 43.6% -> 44.8% (+1.2 pp, n=500, official evaluator), Soft-F1 47.6975 -> 47.1816
  (no improvement), parse success 0.942 -> 0.984, execution success 0.82 -> 0.87. Full tables,
  paired counts and caveats: `docs/RESULTS.md`. Raw run outputs are local uncommitted archives.
- **Phase 7** quantization/local inference: `docs/PHASE7.md`. **Phase 8** backend:
  `docs/PHASE8.md`. **Phase 9** frontend: `docs/PHASE9.md`. **Phase 10** reliability:
  `docs/PHASE10.md`. **Phase 11** persistent serving + real AWS/Vercel proof:
  `docs/PHASE11.md`, `docs/evidence/phase11-production-proof.md`.
- **Phase 12 -- final release and portfolio certification** (documentation, repository audit,
  final checks; no retraining, re-evaluation, model or methodology change): `README.md`,
  `docs/RESULTS.md`, `docs/ARCHITECTURE.md`, `docs/LIMITATIONS.md`, `docs/QUICKSTART.md`,
  `docs/FINAL_CERTIFICATION.md`, `docs/PORTFOLIO.md`. Final checks: backend 675 passed / 1
  skipped; frontend typecheck, lint and build clean; frontend tests 114/115 with the one
  failure being the accepted landing benchmark-display assertion (a presentation change, not
  scientific evidence).
