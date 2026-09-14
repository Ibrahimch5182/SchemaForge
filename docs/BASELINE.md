# LocalSQL Baseline (Phase 3): Untuned Qwen3-4B Inference

Phase 3 measures how the **untouched, non-fine-tuned** base model performs
on BIRD Mini-Dev, so the eventual QLoRA-tuned model has something concrete
to be compared against. **No training happens in this phase.**

```text
Qwen3-4B base  vs  Qwen3-4B + LocalSQL QLoRA   (comparison enabled by this phase)
```

## Why the untouched base model

`Qwen/Qwen3-4B-Instruct-2507` is used exactly as published -- no LoRA, no
prompt engineering beyond the existing canonical prompt, no repair layer.
Any weaknesses it shows (wrong SQL, contract violations, verbose prose
instead of SQL-only output) are legitimate baseline results, not something
to paper over. Later fine-tuning should visibly move these numbers.

## Why 4-bit NF4 is the primary baseline runtime

The future QLoRA model will be trained and run on the same 4-bit NF4 base
representation. Baselining on full precision instead would confound the
base-vs-fine-tuned comparison with a precision change on top of the
fine-tuning effect. Full-precision or alternative deployment-quantization
experiments are a later phase's concern, not this one's.

## Why this doesn't run on the user's laptop

The development machine has 16 GB system RAM and a 4 GB GPU -- nowhere
near enough to hold a 4-bit ~4B-parameter model plus activations for
generation, let alone run 500 sequential generations in reasonable time.
Phase 3 therefore splits cleanly:

- **Cloud (Linux, CUDA, e.g. Kaggle T4-class)**: the user runs
  `scripts/run_baseline.py` for real, producing `predictions.jsonl`.
- **Local (this machine)**: everything else -- manifest/config
  validation (`--dry-run`), the evaluator (Phase 2), and this
  infrastructure's tests -- runs with no GPU and no model weights.

Claude does not download or run the model locally, and does not run token
profiling locally either (it needs the real Qwen tokenizer + network).

## What "baseline" means here, precisely

One inference pass over the Phase 2 gold-free Mini-Dev generation manifest
(500 examples, no gold SQL in sight), producing one `predicted_sql` per
example via deterministic (non-sampling) decoding. Nothing in this phase
computes accuracy -- that is Phase 2's `scripts/evaluate_bird_minidev.py`,
run afterward as a separate step against the baseline's `predictions.jsonl`.

## Prompt: canonical LocalSQL prompt + Qwen's own chat envelope

No second Text-to-SQL prompt is introduced. Each example's existing
`GenerationExample.prompt` (Phase 1's canonical `build_prompt` output,
already used for BIRD training data and Mini-Dev) becomes the content of a
single user message:

```python
messages = [{"role": "user", "content": example.prompt}]
tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, ...)
```

The Qwen tokenizer's native chat template is the *only* model-specific
layer -- no manually reconstructed special tokens, no chain-of-thought
instructions (this Qwen3 variant is used non-thinking).

## Context mode

Primary Phase 3 benchmark: **`with_business_context`** -- Mini-Dev's BIRD
`evidence` is included via the manifest's prompt exactly as Phase 2 built
it (Phase 1's 50% training-time evidence dropout does not apply to external
evaluation). A `without_business_context` ablation is supported by
`resolve_generation_example()` (rebuilds the prompt via the same canonical
`build_prompt` with no context) but is **not run** in this phase --
conserving GPU/time by not doubling the benchmark pass.

## Output handling: intentionally strict

```python
predicted_sql = raw_completion.strip()
```

That is the entire normalization. No searching for `SELECT`, no stripping
`SQL:` prefixes or markdown fences, no prose extraction, no repair, no
second-LLM cleanup. If the untuned model violates the SQL-only contract,
that failure must remain visible in `predictions.jsonl` -- it is baseline
signal, not noise to clean up.

## Determinism and provenance

Decoding is greedy (`do_sample=False`; no temperature, no top-p). The
actual Hugging Face model commit SHA is resolved and recorded at run time
(never silently trusting mutable `main`); `configs/model.yaml` supports an
explicit `model.revision` pin. Every run's `summary.json` records enough
software/hardware/config provenance (Python/PyTorch/Transformers/
bitsandbytes/CUDA versions, GPU name/memory, resolved model + tokenizer
revision, quantization + generation config, seed, context mode, manifest
sha256) to reproduce which exact files and settings produced it.

## No silent truncation

Prompts are never chopped. `--token-profile` mode loads only the tokenizer
(no 4-bit model, still needs the "model" group + network + CUDA-free) and
reports prompt-length statistics (min/median/p90/p95/max, counts above the
configurable `warn_threshold`/`hard_limit` in `configs/model.yaml`) to
inform later training sequence-length decisions -- it does not gate or
modify this phase's generation run.

## Resume / crash safety

`generations.jsonl` is the append-only source of truth; `predictions.jsonl`
is fully rebuilt from it after every example, so a killed Kaggle session can
always be resumed with the same `--run-id`: already-`"ok"` example IDs are
skipped, nothing is duplicated. Resuming with a **different** model,
revision, quantization, generation config, context mode, manifest, or
`--limit` is refused (`ResumeConflictError`) -- pick a new `--run-id`
instead of silently mixing two configurations in one run's artifacts.

## Workflow

```powershell
# Local (this machine): validate infra only, no GPU/model needed
uv run python scripts/run_baseline.py \
  --manifest data\benchmarks\bird_mini_dev\generation\manifest.jsonl \
  --run-id qwen3-4b-base-nf4-smoke --dry-run --limit 5

# Cloud (Kaggle/Linux CUDA): the real baseline
uv sync --group model
uv run python scripts/run_baseline.py \
  --manifest data/benchmarks/bird_mini_dev/generation/manifest.jsonl \
  --run-id qwen3-4b-base-nf4

# Cloud: optional prompt-length profiling (tokenizer only)
uv run python scripts/run_baseline.py \
  --manifest data/benchmarks/bird_mini_dev/generation/manifest.jsonl \
  --run-id qwen3-4b-base-nf4 --token-profile

# Local or cloud (evaluation is separate from generation): score it
uv run python scripts/evaluate_bird_minidev.py \
  --predictions data/runs/qwen3-4b-base-nf4/predictions.jsonl
```

No Kaggle screenshots or results are included here -- the actual GPU run
is performed by the user, not fabricated by this document.
