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
