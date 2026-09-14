"""Run the Qwen3-4B-Instruct-2507 (4-bit NF4) baseline against the Phase 2
gold-free Mini-Dev generation manifest. NO fine-tuning happens here.

Consumes ONLY `data/benchmarks/bird_mini_dev/generation/manifest.jsonl` --
never the grading reference or archive gold SQL. Output `predictions.jsonl`
conforms directly to the Phase 2 prediction contract and can be scored with
`scripts/evaluate_bird_minidev.py --predictions ...` (a separate step; this
script computes no accuracy itself).

Modes:
  --dry-run         Validate manifest/config/resume logic. No model, no CUDA.
  --token-profile   Load only the tokenizer and report prompt token-length
                     stats. No 4-bit model load; still needs the "model"
                     dependency group + network + the real Qwen tokenizer.
  (default)         Full generation. Requires CUDA + the "model" group.

Usage (on a CUDA cloud/Kaggle machine):
    uv sync --group model
    uv run python scripts/run_baseline.py \\
        --manifest data/benchmarks/bird_mini_dev/generation/manifest.jsonl \\
        --run-id qwen3-4b-base-nf4

Usage (local Windows dev, no GPU -- infra validation only):
    uv run python scripts/run_baseline.py \\
        --manifest data/benchmarks/bird_mini_dev/generation/manifest.jsonl \\
        --run-id qwen3-4b-base-nf4-smoke --dry-run --limit 5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from localsql.benchmark.models import GenerationExample  # noqa: E402
from localsql.model.config import (  # noqa: E402
    generation_summary,
    load_model_config,
    quantization_summary,
)
from localsql.model.generation import generate_one_example, resolve_generation_example  # noqa: E402
from localsql.model.run_artifacts import (  # noqa: E402
    Provenance,
    ResumeConflictError,
    RunConfig,
    RunDirectory,
    file_sha256,
    numeric_stats,
)

FORBIDDEN_MANIFEST_HINT = (
    "If this file contains a `sql`/`gold_sql` field, you likely pointed at the grading "
    "reference by mistake -- the baseline runner must only ever read the gold-free "
    "generation manifest."
)


def load_manifest(path: Path) -> list[GenerationExample]:
    examples = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                examples.append(GenerationExample.model_validate_json(line))
            except Exception as e:
                raise ValueError(
                    f"{path}:{line_no}: failed to parse as a gold-free GenerationExample: {e}\n"
                    f"{FORBIDDEN_MANIFEST_HINT}"
                ) from e
    return examples


def run_dry_run(manifest: list[GenerationExample], run_dir: RunDirectory, config: RunConfig) -> None:
    print(f"Manifest: {len(manifest)} gold-free examples parsed OK (no gold fields present).")
    print(f"Manifest sha256: {config.manifest_sha256}")
    try:
        run_dir.prepare(config)
    except ResumeConflictError as e:
        print(f"RESUME CHECK: would REFUSE -- {e}")
        return
    completed = run_dir.completed_example_ids()
    print(f"Run directory: {run_dir.root}")
    print(f"Already completed (resume): {len(completed)} / {len(manifest)}")
    print(f"Remaining to generate: {len(manifest) - len(completed)}")
    print("Dry run OK -- no model loaded, no CUDA required.")


def run_token_profile(
    manifest: list[GenerationExample], model_cfg, context_mode: str, run_dir: RunDirectory
) -> None:
    from localsql.model.qwen_backend import QwenBackend  # lazy: needs transformers

    backend = QwenBackend(model_cfg)
    resolved_revision = backend.load_tokenizer_only()
    print(f"Tokenizer loaded (revision {resolved_revision}). Counting prompt tokens ...")

    counts = []
    for ex in manifest:
        resolved = resolve_generation_example(ex, context_mode)
        counts.append(backend.count_prompt_tokens(resolved.prompt))

    stats = numeric_stats([float(c) for c in counts])
    thresholds = {}
    for name, limit in (
        ("warn_threshold", model_cfg.token_profile.warn_threshold),
        ("hard_limit", model_cfg.token_profile.hard_limit),
    ):
        thresholds[f"count_above_{limit}"] = sum(1 for c in counts if c > limit)

    report = {
        "context_mode": context_mode,
        "tokenizer_revision": resolved_revision,
        "example_count": len(counts),
        "prompt_token_stats": stats,
        **thresholds,
    }
    run_dir.root.mkdir(parents=True, exist_ok=True)
    profile_path = run_dir.root / "token_profile.json"
    profile_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nWrote {profile_path}")


def run_generation(
    manifest: list[GenerationExample],
    model_cfg,
    context_mode: str,
    run_dir: RunDirectory,
    run_config: RunConfig,
) -> None:
    from localsql.model.qwen_backend import QwenBackend  # lazy: needs torch/transformers/bitsandbytes

    backend = QwenBackend(model_cfg)
    print(f"Loading {model_cfg.model.id} (4-bit {model_cfg.runtime.quantization}) ...")
    info = backend.load()
    print(f"Loaded. Resolved revision: {info.resolved_revision}  GPU: {info.gpu_name}")

    resolved_config = RunConfig(
        run_id=run_config.run_id,
        model_id=run_config.model_id,
        model_revision=info.resolved_revision,
        quantization=run_config.quantization,
        generation=run_config.generation,
        context_mode=run_config.context_mode,
        manifest_path=run_config.manifest_path,
        manifest_sha256=run_config.manifest_sha256,
        limit=run_config.limit,
    )
    run_dir.prepare(resolved_config)  # raises ResumeConflictError on mismatch
    completed = run_dir.completed_example_ids()
    print(f"Resuming: {len(completed)} / {len(manifest)} already completed.")

    start = time.perf_counter()
    generated = 0
    failed = 0
    stopped_early = False
    for ex in manifest:
        if ex.example_id in completed:
            continue
        resolved_ex = resolve_generation_example(ex, context_mode)
        record = generate_one_example(resolved_ex, backend, info.resolved_revision, context_mode)
        run_dir.append_generation(record.to_dict())
        run_dir.rewrite_predictions(model_id=model_cfg.model.id)

        if record.status == "ok":
            generated += 1
        else:
            failed += 1
            print(f"  ERROR {ex.example_id}: {record.error}")
            if record.is_oom:
                print("GPU out-of-memory detected -- stopping run (CUDA context likely unusable).")
                stopped_early = True
                break

    total_runtime_s = time.perf_counter() - start
    completed_after = run_dir.completed_example_ids()

    latencies = []
    input_tokens = []
    output_tokens = []
    with run_dir.generations_path.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("status") == "ok":
                if rec.get("latency_ms") is not None:
                    latencies.append(rec["latency_ms"])
                if rec.get("input_tokens") is not None:
                    input_tokens.append(float(rec["input_tokens"]))
                if rec.get("output_tokens") is not None:
                    output_tokens.append(float(rec["output_tokens"]))

    summary = {
        "run_id": run_config.run_id,
        "model_id": model_cfg.model.id,
        "model_revision": info.resolved_revision,
        "tokenizer_revision": info.tokenizer_revision,
        "quantization": info.quantization,
        "generation_config": generation_summary(model_cfg),
        "context_mode": context_mode,
        "manifest_sha256": run_config.manifest_sha256,
        "requested_examples": len(manifest),
        "completed_examples": len(completed_after),
        "generated_this_run": generated,
        "failed_this_run": failed,
        "stopped_early_oom": stopped_early,
        "total_runtime_seconds": round(total_runtime_s, 2),
        "latency_ms_stats": numeric_stats(latencies),
        "input_token_stats": numeric_stats(input_tokens),
        "output_token_stats": numeric_stats(output_tokens),
        "peak_gpu_memory_mb": backend.peak_memory_mb(),
        "provenance": Provenance.collect(info).to_dict(),
    }
    run_dir.write_summary(summary)
    print(f"\nGenerated {generated} (failed {failed}) this run. Completed overall: {len(completed_after)}/{len(manifest)}")
    print(f"Summary: {run_dir.summary_path}")
    print(f"Predictions (Phase 2 contract): {run_dir.predictions_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--model-config", type=Path, default=REPO_ROOT / "configs" / "model.yaml")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--context-mode", choices=["with_business_context", "without_business_context"], default=None)
    parser.add_argument("--dry-run", action="store_true", help="Validate infra only. No model, no CUDA.")
    parser.add_argument("--token-profile", action="store_true", help="Tokenizer-only prompt length profiling.")
    args = parser.parse_args()

    model_cfg = load_model_config(args.model_config)
    context_mode = args.context_mode or model_cfg.prompt.context_mode

    manifest = load_manifest(args.manifest)
    if args.limit:
        manifest = manifest[: args.limit]
    if not manifest:
        print("BLOCKER: manifest is empty after applying --limit.")
        sys.exit(1)

    manifest_sha256 = file_sha256(args.manifest)
    run_config = RunConfig(
        run_id=args.run_id,
        model_id=model_cfg.model.id,
        model_revision=model_cfg.model.revision,  # placeholder; resolved before a real run
        quantization=quantization_summary(model_cfg),
        generation=generation_summary(model_cfg),
        context_mode=context_mode,
        manifest_path=str(args.manifest),
        manifest_sha256=manifest_sha256,
        limit=args.limit,
    )
    runs_root = REPO_ROOT / model_cfg.paths["runs_dir"]
    run_dir = RunDirectory(runs_root, args.run_id)

    if args.dry_run:
        run_dry_run(manifest, run_dir, run_config)
        return

    if args.token_profile:
        run_token_profile(manifest, model_cfg, context_mode, run_dir)
        return

    try:
        run_generation(manifest, model_cfg, context_mode, run_dir, run_config)
    except ResumeConflictError as e:
        print(f"BLOCKER: {e}")
        sys.exit(1)
    except Exception as e:
        # Model load failures (no CUDA, OOM at load, bad revision, etc.) stop
        # the run outright -- no automatic fallback model, no retry.
        print(f"BLOCKER: run stopped -- {type(e).__name__}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
