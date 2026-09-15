# LocalSQL Training (Phase 4): QLoRA Smoke Test

Phase 4 proves the QLoRA training path works correctly end to end -- on a
small subset and/or a bounded number of optimizer steps -- before any real,
full-length fine-tuning run is attempted. **This is a smoke test, not the
fine-tuning experiment.** No full training happens in this phase, and BIRD
Mini-Dev is never touched here.

## Result (Kaggle, real run) -- COMPLETE

**20/20 optimizer steps completed, all losses/grad norms finite, final
train_loss 0.4100, peak GPU memory 11,550.2 MB on a Tesla T4.** Resolved
model/tokenizer revision `cdbee75f17c01a7cc42f958dc650907174af0554` (same
as the Phase 3 baseline). Adapter saved and reload-verified
(`adapter_active=true`). An initial attempt hit a CUDA allocator-
fragmentation OOM; the retry with `PYTORCH_ALLOC_CONF=expandable_segments:True`
succeeded with **no hyperparameter changes**. Full writeup, real token
profile, and the Phase 5 context-length decision (explicitly **not**
resolved by this phase) are in `PROJECT.md` (Phase 4).

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

**4096 is NOT approved for Phase 5 full training** (excludes ~23% of
examples); **8192 is not automatically approved either** (still excludes
~8.9%). The long-context/schema strategy -- raise the limit further,
handle `works_cycles`/`hockey` specially, or something else -- is an
explicit **Phase 5 pre-training decision**, not made here. Whatever is
decided, gold SQL must never be silently truncated and no database group
may be silently discarded -- any exclusion must be explicit and reported.

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

See `PROJECT.md` (Phase 4) for the full rationale, real-hardware evidence,
and the unresolved Phase 5 context-length decision.
