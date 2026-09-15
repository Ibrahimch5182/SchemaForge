# LocalSQL Training (Phase 4): QLoRA Smoke Test

Phase 4 proves the QLoRA training path works correctly end to end -- on a
small subset and/or a bounded number of optimizer steps -- before any real,
full-length fine-tuning run is attempted. **This is a smoke test, not the
fine-tuning experiment.** No full training happens in this phase, and BIRD
Mini-Dev is never touched here.

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

## Token profiling before trusting `max_seq_length`

```powershell
uv run python scripts/run_qlora_smoke.py --run-id <id> --token-profile
```

Loads only the tokenizer (no 4-bit model) and reports both the prompt-only
and the full prompt+completion SFT sequence length distribution
(min/median/p90/p95/p99/max, counts above 4096/8192) over the real
training examples. This does **not** auto-adjust `configs/train.yaml` --
a human reviews the evidence first.

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

## Adapter reload sanity check

```powershell
uv run python scripts/run_qlora_smoke.py --run-id qlora-smoke-1 --verify-adapter
```

Reloads the base model + saved adapter, confirms the adapter is active,
and runs one tiny generation. **Not an accuracy evaluation** -- BIRD
Mini-Dev is scored separately, with the existing Phase 2 evaluator, only
once there is a real fine-tuned checkpoint worth evaluating.

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

See `PROJECT.md` (Phase 4) for the full rationale and acceptance criteria
for marking this phase complete.
