# LocalSQL Training: QLoRA Setup, Smoke Test (Phase 4), Certification and Final Run (Phase 5)

This document is the authoritative training reference. It covers the QLoRA setup, the Phase 4
smoke test, Phase 5B certification/resume infrastructure, and the final canonical run
(checkpoint-1518). Measured accuracy results live in [`RESULTS.md`](RESULTS.md), not here.

## Final training at a glance

| Item | Value |
|---|---|
| Base model | `Qwen/Qwen3-4B-Instruct-2507`, revision `cdbee75f17c01a7cc42f958dc650907174af0554` (unmodified weights; LoRA adapter only) |
| Training quantization | 4-bit **NF4**, double quantization, FP16 compute (QLoRA) |
| LoRA | rank 16, alpha 32, dropout 0.05, all 7 projections (`q/k/v/o` + `gate/up/down`) |
| Optimization | batch 1, gradient accumulation 8 (effective 8), LR 1e-4, warmup 5%, `paged_adamw_8bit`, gradient checkpointing, seed 42 |
| Schedule | 2 epochs = 759 optimizer steps/epoch x 2 = **1,518** steps over 6,067 examples |
| Objective | **Completion-only loss**: prompt tokens labelled -100, only gold-SQL tokens trained (explicit, unit-tested in `localsql.train.sft_data`) |
| Context length | `max_seq_length=4096`, approved for the compacted candidate dataset only; zero examples truncated, gold SQL never truncated |
| Data | `birdsql/bird23-train-filtered`, split by `db_id`; Phase 5 candidate dataset with adaptive per-database schema compaction ([`CONTEXT_BUDGET.md`](CONTEXT_BUDGET.md)); training file SHA-256 `e23a97ea746cef24b17f6bea8dc8440ab96313798837033ec76af9ca79830196` |
| Checkpoint selection | **Final checkpoint of the fixed 2-epoch schedule** (`checkpoint-1518`); the documented methodology has no score-based checkpoint-selection step, and Mini-Dev was never used to pick it |
| Adapter identity | `adapter_model.safetensors` SHA-256 `f7b78b3cb012219bdc9ef48ee2cf5a9105a9d395033da4f9c8a1af2f10ff34cc` |
| Hardware | Kaggle GPU (T4-class), run across resumable sessions; the resume design keeps a multi-session run equivalent to one uninterrupted 2-epoch run |

Training-time **NF4** (used for training and for all Phase 3-6 evaluation) is distinct from
deployment-time **Q4_K_M** GGUF quantization (Phase 7, [`PHASE7.md`](PHASE7.md)): the deployed
base is numerically different from the model that was evaluated, and no accuracy claim is made
for the quantized model.

Deployment artifacts derived from this adapter: F16 LoRA GGUF SHA-256
`53ee2c6dd036ebcccdf0c71bf682c961244ca4665e0cc53bd809ac98b944ba48`, Q4_K_M base SHA-256
`3df3d5bfa7290f20e8b0ad2b9bae78aa06fb198b97837e4ebc28b5867851c848` (never committed to Git).

## Reproducibility boundaries

Reproducible from this repository: data preparation and DB-level splits (deterministic, seeded),
the Phase 5 candidate dataset (`scripts/build_phase5_candidate.py`), SFT formatting and
completion-only masking, the training/resume/export tooling, the evaluation runners and the
official-evaluator adapter, GGUF conversion and manifests, and the whole serving stack.

Not bit-for-bit reproducible, by nature: the GPU training run itself (CUDA/bitsandbytes
non-determinism, session boundaries and library versions on Kaggle), so re-training yields a
statistically similar but not byte-identical adapter. The frozen adapter and GGUF files are
therefore identified by SHA-256 rather than by "re-run to get it". The raw Kaggle outputs and
model weights are **not** committed to Git; the evidence directories committed here
(`kaggle-phase3-export/`, `kaggle-phase4-evidence/`, `kaggle-phase5-tokenizer-evidence/`) are the
subset that was checked in. The narrative journal for every step is [`../PROJECT.md`](../PROJECT.md).

## Phase 4 smoke-test result (Kaggle, real run) -- COMPLETE

**20/20 optimizer steps completed, all losses/grad norms finite, final
train_loss 0.4100, peak GPU memory 11,550.2 MB on a Tesla T4.** Resolved
model/tokenizer revision `cdbee75f17c01a7cc42f958dc650907174af0554` (same
as the Phase 3 baseline). Adapter saved and reload-verified
(`adapter_active=true`). An initial attempt hit a CUDA allocator-
fragmentation OOM; the retry with `PYTORCH_ALLOC_CONF=expandable_segments:True`
succeeded with **no hyperparameter changes**. Full writeup and real token
profile are in `PROJECT.md` (Phase 4); the context-length decision this
phase left unresolved was later settled in Phase 5A/5B (compaction policy
+ real-tokenizer-confirmed `max_seq_length=4096` for the candidate
dataset) -- see `PROJECT.md` (Phase 5A/5B).

## Data: reused, not re-derived

Training examples come from Phase 1's `data/processed/train.jsonl`
verbatim -- 6,067 examples across 62 databases, with the canonical prompt,
gold-SQL completion, and evidence-dropout decision already baked in. This
phase never re-splits the dataset, never re-derives the prompt, and never
re-decides evidence dropout.

## SFT formatting

```python
messages_prompt_only = [{"role": "user", "content": example.prompt}]
messages_full = messages_prompt_only + [{"role": "assistant", "content": example.completion}]
```

Two `tokenizer.apply_chat_template` calls -- one for `messages_prompt_only`
(`add_generation_prompt=True`), one for `messages_full`
(`add_generation_prompt=False`) -- establish the exact prompt/completion
token boundary. No second SQL prompt is introduced; the existing canonical
LocalSQL prompt is the entire user turn.

## Completion-only loss

Everything before the boundary (schema/question/business context) gets
label `-100`; only the gold-SQL completion tokens are trainable. Padding
also gets `-100`. This is implemented explicitly
(`localsql.train.sft_data.build_sft_encoding`), not left to a framework's
default behavior, and is unit-tested (prompt masked, completion trainable,
padding masked, at least one trainable token, no gold leakage into the
prompt region).

## QLoRA configuration (starting point, not final)

4-bit NF4 (double quant, float16 compute) -- same representation as the
Phase 3 baseline, so a later base-vs-adapter comparison isolates
fine-tuning. LoRA r=16, alpha=32, dropout=0.05, on
`q/k/v/o_proj` + `gate/up/down_proj`. Batch size 1, gradient accumulation
8, LR 1e-4, warmup 0.05, gradient checkpointing, paged AdamW 8-bit, seed
42. `max_seq_length=4096` is explicitly **provisional** -- see below.

Real preflight (`qlora-smoke-preflight`, 16 examples/1 step): 33,030,144
trainable / 2,238,840,320 total parameters, loss 1.5232, finite grad norm,
adapter saved, no OOM -- first proof the mechanics work on real hardware.

## Token profiling before trusting `max_seq_length`

```powershell
uv run python scripts/run_qlora_smoke.py --run-id <id> --token-profile
```

Loads only the tokenizer (no 4-bit model) and reports both the prompt-only
and the full prompt+completion SFT sequence length distribution
(min/median/p90/p95/p99/max, counts above 4096/8192) over the real
training examples. This does **not** auto-adjust `configs/train.yaml` --
a human reviews the evidence first.

**Real profile, all 6,067 training examples** (full SFT sequence,
prompt+completion): min 549, median 2,506, p90 8,013.8, p95 27,927, p99
27,975, max 28,082 -- **1,398 examples (~23%) exceed 4096; 539 (~8.9%)
exceed 8192.** Real-Qwen boundary validation across all 6,067 examples:
**0 prefix mismatches** (the completion-only mask boundary is correct in
practice, not just in unit tests); median SQL completion contributes only
50 tokens, max 216 -- **the long tail is schema-driven, not SQL-target-
driven.** The tail is concentrated, not spread out: 9 of 62 databases have
any example over 4096, and just 2 databases (`works_cycles`: n=383, all
over both 4096 and 8192, max 28,082; `hockey`: n=156, all over both, max
13,870) account for the overwhelming majority of it.

**Smoke-test subset caveat**: the 200-example subset used for the smoke
run below was deliberately chosen to fit under 4096 (max 3,593 tokens) --
that 3,593 ceiling describes the smoke subset, not the full training set,
and must not be read as evidence that 4096 is adequate overall.

**4096 was NOT approved for full training against this ORIGINAL
(pre-compaction) dataset** (excludes ~23% of examples). Phase 5's
adaptive per-database schema compaction (see `PROJECT.md`, Phase 5A/5B)
resolves this: after compacting the flagged DBs (`works_cycles`,
`hockey`, and 7 others), the real-tokenizer profile of the resulting
candidate dataset shows **zero** examples over 4096 tokens, and
`max_seq_length=4096` is approved **for that candidate dataset only** --
never for this original data. Gold SQL is never silently truncated and no
database group is silently discarded -- any exclusion is explicit and
reported (`localsql.train.sft_data.build_sft_encoding` never truncates;
an over-length example is used whole or explicitly skipped and named in
`summary.json`).

## Smoke run

```powershell
uv sync --group model --group train
uv run python scripts/run_qlora_smoke.py --run-id qlora-smoke-1 \
  --max-train-examples 200 --max-steps 20
```

Loads the base model in 4-bit, attaches the LoRA adapter, trains for
`--max-steps` bounded optimizer steps over (at most) `--max-train-examples`
completion-only-masked examples, saves the adapter, and (unless
`--skip-verify`) runs the adapter-reload sanity check automatically.
Fails clearly (not silently) on NaN/Inf loss, CUDA OOM, or a model-load
failure -- no automatic fallback.

**Real result** (`qlora-smoke-1-alloc-retry`, 200 examples, 20 steps): all
200 usable (0 skipped for exceeding `max_seq_length`), 20/20 steps
completed, all losses/grad norms finite, final `train_loss = 0.40999`,
runtime 3098.01s, **peak GPU memory 11,550.2 MB** (of 14,911.7 MB on the
T4).

**CUDA OOM on the first attempt, and the fix**: the identical 200-example/
20-step configuration first failed with a CUDA out-of-memory error before
completing step 1 (T4 total 14.56 GiB; PyTorch had 9.50 GiB allocated and
3.47 GiB reserved-but-unallocated -- a classic allocator-fragmentation
pattern, not an actual capacity shortfall; the failing allocation request
was only 1.48 GiB). Setting

```bash
export PYTORCH_ALLOC_CONF=expandable_segments:True
```

before rerunning the exact same command succeeded. **No hyperparameter,
model, or dataset change was made** -- this is an environment/allocator
fix, not a training-configuration fix. If a real Kaggle T4/similar-VRAM
run OOMs on the smoke test, try this environment variable before changing
`configs/train.yaml`.

## Adapter reload sanity check

```powershell
uv run python scripts/run_qlora_smoke.py --run-id qlora-smoke-1 --verify-adapter
```

Reloads the base model + saved adapter, confirms the adapter is active,
and runs one tiny generation. **Not an accuracy evaluation** -- BIRD
Mini-Dev is scored separately, with the existing Phase 2 evaluator, only
once there is a real fine-tuned checkpoint worth evaluating.

**Real result**: `adapter_active=true`, `adapter_names=["default"]`.
Sample completion from the saved adapter: `SELECT T1.director_name FROM
movies AS T1 WHERE T1.movie_title = 'Sex, Drink and Bloodshed'` --
well-formed and SQL-only. Still not an accuracy claim.

## Known cleanup items (not blocking, not a Phase 4 rerun)

- **Transformers 5.x `warmup_ratio` deprecation warning**, observed during
  the real run -- `TrainingArguments` still honored it correctly; tracked
  as a Phase 5 compatibility cleanup, does not affect the completed smoke
  results.
- **Harmless greedy-generation warning** (temperature/top_p/top_k set but
  unused since `do_sample=False`), observed during adapter-reload
  verification -- cosmetic only.

## Artifacts

```text
data/runs/<run-id>/
├── run_config.json          resolved config + train-file sha256
├── train_metrics.jsonl      one line per logged training step
├── checkpoint/              Trainer's working directory
├── adapter/                 saved LoRA adapter + tokenizer
├── adapter_verification.json (if --verify-adapter run standalone)
├── train_token_profile.json (if --token-profile run)
└── summary.json             full provenance: model/tokenizer revision,
                              quantization, LoRA + optimization config,
                              peak GPU memory, runtime, software versions
```

All gitignored (`data/runs/*`), same as Phase 3's baseline run artifacts.

## Dry run (no GPU/model needed)

```powershell
uv run python scripts/run_qlora_smoke.py --run-id <id> --dry-run --max-train-examples 50
```

Validates the training data and config only.

## Checkpointing and resume (Phase 5B)

```powershell
uv run python scripts/run_qlora_smoke.py --run-id <id> \
  --save-steps 1 --save-total-limit 2
uv run python scripts/run_qlora_smoke.py --run-id <new-id> \
  --resume-from-checkpoint data/runs/<id>/checkpoint/checkpoint-<N>
```

`--save-steps` enables `Trainer`'s own full-state periodic checkpointing
(LoRA adapter, optimizer, LR scheduler, RNG, `trainer_state.json`) under
`checkpoint/checkpoint-<step>` -- not just the final adapter-only export
under `adapter/`. `--resume-from-checkpoint` is always explicit; nothing
auto-discovers or auto-resumes from an arbitrary directory, and a run-id
that already has `summary.json` is refused (start a new `--run-id` to
resume, pointing it at the prior run-id's checkpoint dir). `run_config.json`
and `summary.json` both record `checkpoint_dir`, `save_steps`,
`save_total_limit`, `resume_from_checkpoint`, `starting_global_step`, the
final `global_step`, and (reliability-hardened) `source_revision`/
`source_revision_origin` + `accelerate_version`/`trl_version`, so a resume
is independently verifiable from run artifacts alone. `--source-revision`
lets you pin the source commit explicitly -- `git archive`/Kaggle upload
strips `.git`, so without it (or a `SOURCE_REVISION` file at the repo
root) this falls back to a best-effort `git rev-parse HEAD` that can
silently come up empty.

`scripts/export_checkpoint.py` packages a checkpoint for durable transfer
off an ephemeral Kaggle VM -- and now FAILS CLOSED: it refuses to write
any package (`IncompleteCheckpointError`) if the checkpoint is missing
any required resumable-state category (model/adapter, optimizer,
scheduler, trainer state, RNG state, training arguments), and it packages
the run's EXACT training-data JSONL byte-for-byte, SHA256-verified
against `run_config.json` (`TrainingDataMismatchError` if it doesn't
match -- resuming against a different/reordered dataset is unsafe). See
`PROJECT.md` (Phase 5B) for the exact stop/resume certification procedure
(RUN A / RUN B) and the full reliability-correction writeup.

**Canonical throughput sampling requires a REAL per-example token-length
manifest, not the character-count estimator.** Run
`--token-profile` against the candidate dataset first (this also writes
`<profile>_token_lengths.jsonl`), then `scripts/build_certification_sets.py`
to build the sample from it:

```powershell
uv run python scripts/run_qlora_smoke.py --run-id phase5-candidate-profile \
  --input data/processed_phase5_candidate/train.jsonl --token-profile
uv run python scripts/build_certification_sets.py
```

See `PROJECT.md` (Phase 4 for the smoke-test rationale and real-hardware
evidence; Phase 5A/5B for the context-length policy resolution and GPU/
throughput/resume certification status).
