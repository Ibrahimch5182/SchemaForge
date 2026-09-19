# Phase 7 — Quantization + Local Inference (llama.cpp / GGUF)

**Status: Phase 7 PASSED** for quantization, local deployment, provenance, and
inference benchmarking. Quantization quality equivalence is **NOT claimed**
(see Measured results). No further Phase 7 experimentation is required.

## Goal

Turn the frozen fine-tuned model into a reproducible local deployment
artifact and measure it on the target machine (Windows, 16 GB RAM, 4 GB VRAM):
quantization, provenance, disk-size reduction, latency, throughput, memory,
determinism, and a quantization regression sanity check.

## Frozen inputs (never changed here)

| Item | Value |
|---|---|
| Base model | `Qwen/Qwen3-4B-Instruct-2507` @ `cdbee75f17c01a7cc42f958dc650907174af0554` |
| Adapter | `checkpoint-1518` (r=16, alpha=32) |
| `adapter_model.safetensors` SHA256 | `f7b78b3cb012219bdc9ef48ee2cf5a9105a9d395033da4f9c8a1af2f10ff34cc` |
| Local adapter dir | `.artifacts/phase7/adapter-1518/phase5c-final-1518-safety/adapter` |

All constants live in `configs/phase7.yaml`. Phase 6 results are context only
and are not recomputed.

## Training-time NF4 vs deployment-time Q4_K_M

These are different things. **NF4** (bitsandbytes) was the base-weight
representation *during QLoRA training and Phase 3-6 evaluation*, on GPU.
**Q4_K_M** is a llama.cpp *GGUF block quantization* of the **base** weights for
CPU/small-GPU deployment. The deployed base therefore differs numerically from
everything evaluated in Phases 3-6, and no Phase 6 number transfers to it
automatically. Quality is measured, not assumed.

## Deployment mode: hot LoRA (primary)

The fine-tuned behavior comes from **base GGUF + LoRA GGUF**, loaded together
at runtime with upstream llama.cpp's `--lora LORA.gguf`. The LoRA stays a small
separate artifact (~130 MB) held constant while only the base representation
changes (F16 vs Q4_K_M), which is what makes the quantization sanity check
clean.

Why this is primary on Windows: the official pinned CPU binaries
(llama.cpp b10964, `llama-completion.exe`, `llama-quantize.exe`) do not include
`llama-export-lora.exe`, and this machine has no CMake/MSVC toolchain. A merged
GGUF is **optional packaging**: not required for Phase 7 PASS, not the
default, and its absence never blocks the workflow.

## Why Q4_K_M

Roughly 4.5-5 bits/weight: a 4B base shrinks from ~8 GB (F16) to ~2.5 GB, so it
fits in 16 GB RAM (and mostly in 4 GB VRAM with partial offload), while the
K-quant mixed-precision layout keeps sensitive tensors at higher precision.
**We make no claim that Q4_K_M preserves quality until measured.**

## Pipeline

```
HF base @ frozen revision  -> base F16 GGUF        (convert_hf_to_gguf.py)
PEFT LoRA checkpoint-1518  -> LoRA F16 GGUF        (convert_lora_to_gguf.py)
base F16 GGUF              -> base Q4_K_M GGUF     (llama-quantize)
runtime: base Q4_K_M + --lora LoRA F16 GGUF        (llama-completion)

OPTIONAL: base GGUF + LoRA GGUF -> merged GGUF     (llama-export-lora)
```

Nothing is merged in Python RAM. llama.cpp is **not vendored**; tools are
passed by explicit path and flags are not assumed stable (`--extra-arg`).
Artifacts: `base-gguf/base-f16.gguf`, `lora-gguf/lora-1518-f16.gguf`,
`q4_k_m/base-Q4_K_M.gguf` (never named "merged").

## CLIs

| Script | Purpose |
|---|---|
| `scripts/prepare_phase7.py` | Validate adapter SHA/metadata + base contract, create `.artifacts/phase7/` layout, write `manifests/phase7_manifest.json`. `--dry-run` writes nothing. |
| `scripts/convert_phase7.py {base,lora,quantize}` (+ optional `merge`) | Run one stage. `quantize` takes the BASE F16 GGUF by default. Preflight, no shell, fail on nonzero exit, write `*.partial.gguf` then rename, record command + input/output SHA256 + size in `manifests/stage-<name>.json` (+ `.log`). Identical rerun is a no-op; a rerun that would replace different content is refused. `--dry-run` prints the command. |
| `scripts/run_local_gguf.py` | Generate on the fixed sanity set with `--model BASE.gguf --lora LORA.gguf` (canonical prompt, greedy, CPU valid; `--lora` required unless `--allow-no-lora`). Records base and LoRA SHA256/size, deployment mode, command, latency, output. Nonzero exit on any failure. |
| `scripts/benchmark_gguf.py` | Benchmarks the effective model (Q4_K_M base + LoRA), reporting individual and combined size. Warm-up (excluded) + measured repeats: wall latency P50/P95, llama.cpp-reported prompt/generation tokens/sec, peak process RSS, size, determinism. Unavailable metrics are `null` + explanation. |
| `scripts/compare_gguf_outputs.py` | Quantization regression sanity check: F16 base + LoRA vs Q4_K_M base + the SAME LoRA (refuses if LoRA hashes differ). Exact normalized agreement, parse/shape diagnostics, changed examples. NOT an accuracy benchmark. |

### Design notes

- **Prompt**: `GenerationExample.prompt` verbatim (the canonical Phase 1/2
  prompt), wrapped only in the Qwen ChatML single-user-turn envelope
  (llama.cpp completion tools don't apply a chat template). Output is
  `raw_completion` → whitespace `strip()` (`normalize_predicted_sql`), never
  repaired. The only stripping beyond that is a literal trailing
  `[end of text]` marker some llama.cpp builds print; this is recorded per
  result (`end_of_text_marker_stripped`).
- **Sanity set**: 10 examples chosen by `sha256(salt:example_id)` ordering from
  the gold-free Phase 2 generation manifest (`GenerationExample` forbids gold
  fields). Deterministic and independent of file order. It is a quantization
  regression check, **not** an accuracy benchmark; Mini-Dev is not scored.
- **Latency**: each request is a fresh process, so wall latency includes model
  load; llama.cpp's own eval timings (excluding load) are reported separately.
- **Peak memory**: OS peak working set of the launched process (Windows
  `GetProcessMemoryInfo`, Linux `VmHWM`); includes touched mmapped model pages.

## Known verification items for the first real run

1. The prompt is passed via `-f`. Confirm with `--extra-arg --verbose-prompt`
   that the tokenized prompt ends with `<|im_start|>assistant\n` (some builds
   trim a trailing newline from prompt files). If it does not, adjust before
   trusting any comparison.
2. `-no-cnv` support differs between `llama-cli` and `llama-completion`
   (`--omit-no-cnv`).
3. The HF base directory cannot prove its own revision; use the cache
   `snapshots/<sha>` directory (checked against the frozen SHA) or the
   manifest records `unverified_by_path`.
4. The adapter was trained on an NF4-quantized base but is applied to an F16
   (then Q4_K_M) base at runtime; another reason quality must be measured.

## Later execution sequence

1. `prepare_phase7.py`.
2. Base snapshot at the frozen revision (outside this repo work).
3. `convert_phase7.py base`, `lora`, `quantize` (each with `--dry-run` first).
4. `run_local_gguf.py` twice with the same `--lora`: F16 base, then Q4_K_M base.
5. `compare_gguf_outputs.py` on the two results.
6. `benchmark_gguf.py` on Q4_K_M base + LoRA (optionally F16 base + LoRA).

## Measured results

### Artifacts

| Artifact | Size |
|---|---|
| Base F16 GGUF | 7.498 GB |
| Base Q4_K_M GGUF | 2.326 GB (~69% smaller than F16) |
| LoRA F16 GGUF | 0.062 GB |
| Deployed model (Q4_K_M base + LoRA) | 2.388 GB combined |

- Q4_K_M base SHA256: `3df3d5bfa7290f20e8b0ad2b9bae78aa06fb198b97837e4ebc28b5867851c848`
- LoRA GGUF SHA256: `53ee2c6dd036ebcccdf0c71bf682c961244ca4665e0cc53bd809ac98b944ba48`
- Deployment mode: `hot_lora` (base GGUF + runtime `--lora`); no merged GGUF used.

### Runtime benchmark (Q4_K_M base + LoRA, CPU)

| Metric | Value |
|---|---|
| Requests successful | 3/3 |
| Deterministic across repeats | true |
| Generation throughput (mean) | 10.86 tokens/sec |
| Prompt processing (mean) | 49.79 tokens/sec |
| Wall latency P50 / P95 (includes model load) | 22434.91 ms / 24850.92 ms |
| Inference-only latency P50 (excludes load) | 19222.24 ms |
| Peak RSS | 5,697,814,528 bytes (~5.70 GB) |

### Quantization regression sanity check

F16 base + LoRA vs Q4_K_M base + the **same** LoRA, on the fixed deterministic
10-example gold-free sample.

| Measure | F16 + LoRA | Q4_K_M + LoRA |
|---|---|---|
| Successful generations | 10/10 | 10/10 |
| Parseable SQL | 8/10 | 9/10 |
| Starts with SELECT/WITH | 9/10 | 10/10 |
| Markdown fences | 0 | 0 |

**Exact normalized output agreement: 4/10 (40%).**

### What these results do and do not show

- **Quantization quality equivalence is NOT claimed.** Q4_K_M changed the
  generated output materially on this sample: 6 of 10 outputs differ from the
  F16 + LoRA outputs.
- The parseability/shape counts are diagnostics only, not correctness. Q4
  scoring slightly higher on them is not evidence that it is better or equal.
- This sanity set is **NOT an accuracy benchmark**: n=10, no execution
  accuracy, no gold comparison. It says nothing about BIRD EX for the
  quantized model, and no Phase 6 number transfers to it.
- Training-time NF4 and deployment-time Q4_K_M remain separate; the deployed
  base differs numerically from what Phases 3-6 evaluated.

## Note: prompt-file trailing newline (added after Phase 8)

Phase 7 benchmark and sanity evidence remains **frozen and reproducible**; the
recorded measurements above are unchanged. Those historical runs used the
then-current llama.cpp prompt-file behavior, without the trailing-newline
guard: `-f` strips the final newline, so the model saw `assistant` instead of
`assistant<newline>`. Phase 8 production serving corrected this train/serve drift
(`LlamaSettings.guard_prompt_trailing_newline=True`; the Phase 7 default stays
`False` so the recorded runs stay reproducible). See `docs/PHASE8.md`. The
Phase 7 numbers should be read as measured under that historical prompt
behavior; they were not re-run or altered.

## Phase 7 PASS criteria — met

- Adapter and base contract validated; manifest written.
- base, lora, quantize stages completed with recorded SHA256/size; base
  Q4_K_M markedly smaller than base F16. `llama-export-lora`/merged GGUF was
  not required.
- Q4_K_M base + LoRA runs locally on CPU; all sanity prompts succeeded.
- Benchmark captured; repeats deterministic.
- Sanity comparison produced and its disagreement reported honestly, with no
  quality-preservation claim.
