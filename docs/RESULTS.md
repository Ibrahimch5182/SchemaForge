# SchemaForge — Final Results

This is the single authoritative results page. The three evaluations below use **different
data, different metrics and different evaluators**. They must be read separately and are
never averaged or combined into one score.

> **One-sentence summary.** QLoRA specialization of Qwen3-4B improved normalized SQL exact
> match 4.2 → 37.4% (+33.2 pp) on a fixed seen-training sample, schema-held-out execution
> accuracy 39.1 → 44.6% (+5.4 pp) on 534 examples from held-out DB schemas, and external
> untouched BIRD MiniDev EX 43.6 → 44.8% (+1.2 pp).

| | A. Seen-training specialization | B. Schema-held-out generalization | C. External BIRD MiniDev |
|---|---|---|---|
| Question answered | Did the adapter learn the task on data it trained on? | Does it transfer to database schemas it never saw? | Does it transfer to an independent, untouched benchmark? |
| Metric | Normalized SQL exact match | Execution accuracy (EX) | Execution accuracy (EX) + Soft-F1 |
| Sample | 500 (fixed sample of training examples) | 534 (7 held-out databases, zero DB overlap with training) | 500 (11 databases, SELECT-only SQLite) |
| Base | 4.20% | 39.14% | 43.6% |
| Fine-tuned | 37.40% | 44.57% | 44.8% |
| Δ | **+33.20 pp** | **+5.43 pp** | **+1.2 pp** |
| Evaluator | SchemaForge normalized-SQL comparison | SchemaForge SQLite execution comparator (**not** the official BIRD evaluator) | Official BIRD Mini-Dev evaluator (vendored, unmodified, pinned commit) |
| Evidential weight | Specialization only — **not** generalization | Strongest internal generalization evidence | Independent confirmation; real but modest |

"Base" is the untouched `Qwen/Qwen3-4B-Instruct-2507` (4-bit NF4); "Fine-tuned" is the same
model plus the `checkpoint-1518` LoRA adapter, evaluated with the identical canonical prompt
and deterministic decoding. See [Provenance](#provenance-and-what-was-evaluated).

---

## A. Seen-training specialization

| | Base | Fine-tuned | Δ |
|---|---|---|---|
| Normalized SQL exact match (n = 500) | 4.20% (21) | 37.40% (187) | **+33.20 pp** |

Paired outcome (500 examples): fine-tuned-only correct **167** · base-only correct **1** ·
both correct **20** · both wrong **312**.

**Interpretation.** The adapter clearly learned the SchemaForge task and the BIRD-style SQL
conventions on databases it was trained on. This is **seen-training specialization**. It is
**not evidence of generalization**: the examples come from the training distribution, and the
metric (normalized string match) is stricter than execution equivalence, which is why the
absolute numbers are low. Nothing in Section A should be quoted as a generalization result.

## B. Schema-held-out generalization

| | Base | Fine-tuned | Δ |
|---|---|---|---|
| Execution accuracy (n = 534) | 39.14% (209) | 44.57% (238) | **+5.43 pp** |
| Executable SQL | 66.85% | 74.53% | **+7.68 pp** |

Paired outcome (534 examples): fine-tuned-only correct **61** · base-only correct **32** ·
both correct **177** · both wrong **264**.

**Data discipline.** The train/validation split is by `db_id`, never by example; the 534
validation examples come from 7 databases with **zero overlap** with the training databases
(asserted in code — see [`DATA_CONTRACT.md`](DATA_CONTRACT.md)).

**Interpretation.** This is the strongest *internal* evidence that the fine-tune generalizes to
unseen schemas: it gained 61 examples and lost 32 (a net +29). As a derived sanity check on
those frozen counts, an exact McNemar test (two-sided exact binomial test on the 93 discordant pairs) gives p ≈ 0.00346;
this is arithmetic on the reported counts, not a separate experiment. Caveats: it is a single
run and a single split (no seeds or confidence intervals across splits), and it uses
SchemaForge's own SQLite execution comparator, not the official BIRD evaluator. Compaction of
the largest schemas was applied by an adaptive per-database policy
([`CONTEXT_BUDGET.md`](CONTEXT_BUDGET.md)), and 2 of the 7 validation databases use it.

## C. External, untouched BIRD Mini-Dev

| Metric | Base | Fine-tuned | Δ |
|---|---|---|---|
| Execution accuracy (EX), n = 500 | 43.6% | **44.8%** | **+1.2 pp** |
| Soft-F1 | 47.6975 | 47.1816 | **−0.5159** |
| SQL parse success (diagnostic) | 0.942 | 0.984 | +0.042 |
| Execution success (diagnostic) | 0.82 | 0.87 | +0.05 |

**Data discipline.** BIRD Mini-Dev (locked original 500 SELECT-only SQLite variant, 11
databases) is evaluation-only and was never loaded as training data. Grading truth comes only
from the official archive's gold file. See [`EVALUATION.md`](EVALUATION.md).

**Interpretation — read carefully.**

- The +1.2 pp EX gain is **real but modest**: 500 examples, a single run, a net of about six
  questions. It should not be presented as a large or statistically established improvement.
- **Soft-F1 did not improve** (47.70 → 47.18). The official metrics therefore give a mixed
  picture: slightly higher EX, slightly lower Soft-F1.
- What changed most visibly is **output reliability**: the fine-tuned model emits parseable
  SQL more often (0.942 → 0.984) and executes successfully more often (0.82 → 0.87). Parse and
  execution success are LocalSQL-only diagnostics — they measure well-formedness, **not**
  correctness, and are never reported as EX.
- The fine-tuned model was trained on BIRD *train* data and prompted with the same
  schema-plus-optional-business-context format, which matches the Mini-Dev prompt contract;
  Mini-Dev itself was never used to select checkpoints or tune anything.

---

## What these results do **not** claim

- **No state-of-the-art claim.** SchemaForge is a ~4B open-weight model evaluated under one
  fixed prompt; no comparison to other systems or leaderboards is made or implied.
- **Seen-training improvement is not generalization** (Section A).
- **The deployed quantized model was not re-scored on these benchmarks.** All three
  evaluations above used the 4-bit NF4 base (+ LoRA) in Hugging Face/PEFT. The deployed
  artifact is a llama.cpp **Q4_K_M** base + hot-loaded F16 LoRA. A 10-example sanity check
  (not an accuracy benchmark) showed only 4/10 exact-output agreement between F16+LoRA and
  Q4_K_M+LoRA, so equivalence is **not** claimed. See [`PHASE7.md`](PHASE7.md).
- **Deployment observations are not benchmarks.** The single cloud request (16.0 s, 3.4
  generated tokens/s on 2 vCPU) and local llama.cpp timings in Phase 7 / Phase 11 are
  operational observations, kept separate from accuracy results.
- **Safe ≠ correct.** Passing deterministic safety, preflight and read-only execution does not
  mean an answer is right; `semantic_correctness` is always `not_verified` and confidence is
  intentionally unclaimed ([`PHASE10.md`](PHASE10.md)).
- **The public landing page's benchmark cards are presentation, not evidence.** The values on
  the deployed site's landing cards are a presentation layer and must not be cited as
  scientific results; this document is the authoritative source.

## Provenance and what was evaluated

| Item | Value |
|---|---|
| Base model | `Qwen/Qwen3-4B-Instruct-2507`, revision `cdbee75f17c01a7cc42f958dc650907174af0554` |
| Adapter | `checkpoint-1518` (2 epochs × 759 optimizer steps), `adapter_model.safetensors` SHA-256 `f7b78b3cb012219bdc9ef48ee2cf5a9105a9d395033da4f9c8a1af2f10ff34cc` |
| Training data | `birdsql/bird23-train-filtered`, DB-level split, Phase 5 candidate dataset (6,067 train / 534 validation), `max_seq_length=4096`, zero examples truncated |
| Prompt | The single canonical Phase 1 prompt, wrapped only by Qwen's chat template; `predicted_sql` is `raw_completion.strip()`, never repaired |
| Baseline result | Recorded in [`BASELINE.md`](BASELINE.md) (official EX 43.6, Soft-F1 47.6975) |
| Raw run artifacts | Kaggle run outputs are kept as local, uncommitted archives (not in Git); the frozen numbers above are the authoritative record in this repository |

Reproducibility boundaries are described in [`TRAINING.md`](TRAINING.md#reproducibility-boundaries).
