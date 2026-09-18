"""Run a checkpoint-1518-style QLoRA adapter (Qwen3-4B-Instruct-2507 base,
4-bit NF4) against the same Phase 2 gold-free Mini-Dev generation manifest
used by `scripts/run_baseline.py`. NO fine-tuning happens here.

This is the fine-tuned counterpart to the Phase 3 baseline runner. It
reuses the same manifest/prompt/generation/prediction machinery unchanged
(`GenerationExample`, `resolve_generation_example`, `generate_one_example`,
`RunDirectory`, the Phase 2 `predictions.jsonl` contract, `model.yaml`
generation config, `normalize_predicted_sql`) so the only difference from
the base model comparison is one variable: the LoRA adapter is loaded and
active on top of the identical NF4 base model. Never merges the adapter
into the base weights.

Consumes ONLY `data/benchmarks/bird_mini_dev/generation/manifest.jsonl` --
never the grading reference or archive gold SQL.

Modes:
  --dry-run   Validate manifest/config/adapter-file/resume logic. No model,
              no CUDA -- adapter validation here is filesystem-only.
  (default)   Full generation. Requires CUDA + the "model" and "train"
              (for `peft`) dependency groups.

Usage (on a CUDA cloud/Kaggle machine):
    uv sync --group model --group train
    uv run python scripts/run_finetuned.py \\
        --manifest data/benchmarks/bird_mini_dev/generation/manifest.jsonl \\
        --run-id qwen3-4b-checkpoint-1518 \\
        --adapter /path/to/checkpoint-1518 \\
        --context-mode with_business_context

Usage (local Windows dev, no GPU -- infra validation only):
    uv run python scripts/run_finetuned.py \\
        --manifest data/benchmarks/bird_mini_dev/generation/manifest.jsonl \\
        --run-id qwen3-4b-checkpoint-1518-smoke \\
        --adapter /path/to/checkpoint-1518 --dry-run --limit 5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from localsql.benchmark.models import GenerationExample  # noqa: E402
from localsql.model.config import (  # noqa: E402
    generation_summary,
    load_model_config,
    quantization_summary,
)
from localsql.model.generation import generate_one_example, resolve_generation_example  # noqa: E402
from localsql.model.qwen_backend import AdapterValidationError, validate_adapter_path  # noqa: E402
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
    "reference by mistake -- the fine-tuned runner must only ever read the gold-free "
    "generation manifest."
)

ADAPTER_CONFIG_FILE = "adapter_config.json"
ADAPTER_WEIGHTS_FILE = "adapter_model.safetensors"


def load_manifest(path: Path) -> list[GenerationExample]:
    """Same gold-free manifest contract as `scripts/run_baseline.py`."""
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


def adapter_model_id(base_model_id: str, adapter_path: Path) -> str:
    """A distinct RunConfig/predictions `model_id` for the adapter run.

    Deliberately different from the bare base model id so that (a) this
    run can never silently resume-merge with a base-model run under the
    same --run-id (RunConfig.matches() compares model_id), and (b) every
    predictions.jsonl row is self-evidently a fine-tuned result, not the
    Phase 3 baseline.
    """
    return f"{base_model_id}+lora:{adapter_path.resolve()}"


def run_dry_run(
    manifest: list[GenerationExample], run_dir: RunDirectory, config: RunConfig, adapter_path: Path
) -> None:
    print(f"Manifest: {len(manifest)} gold-free examples parsed OK (no gold fields present).")
    print(f"Manifest sha256: {config.manifest_sha256}")

    validate_adapter_path(adapter_path)
    print(f"Adapter directory OK: {adapter_path} (required files present).")

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


def run_generation(
    manifest: list[GenerationExample],
    model_cfg,
    context_mode: str,
    run_dir: RunDirectory,
    run_config: RunConfig,
    adapter_path: Path,
) -> None:
    from localsql.model.qwen_backend import QwenBackend  # lazy: needs torch/transformers/bitsandbytes/peft

    validate_adapter_path(adapter_path)
    adapter_weights_sha256 = file_sha256(adapter_path / ADAPTER_WEIGHTS_FILE)

    backend = QwenBackend(model_cfg)
    print(f"Loading {model_cfg.model.id} (4-bit {model_cfg.runtime.quantization}) + adapter {adapter_path} ...")
    info = backend.load(adapter_path=adapter_path)
    if not info.adapter_active:
        # load() already raises AdapterValidationError on this, but fail
        # closed here too rather than silently generating off the base model.
        raise AdapterValidationError(f"Adapter did not activate: {adapter_path}")
    print(f"Loaded. Resolved base revision: {info.resolved_revision}  GPU: {info.gpu_name}  adapter_active={info.adapter_active}")

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
        run_dir.rewrite_predictions(model_id=run_config.model_id)

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
        # Provenance (requirement: must make it impossible to mistake this
        # run for the base model).
        "base_model_id": model_cfg.model.id,
        "base_model_revision": info.resolved_revision,
        "adapter_path": str(adapter_path.resolve()),
        "adapter_weights_sha256": adapter_weights_sha256,
        "adapter_active": info.adapter_active,
        "model_id": run_config.model_id,
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
    parser.add_argument("--adapter", type=Path, required=True, help="Path to the PEFT LoRA adapter directory (e.g. checkpoint-1518).")
    parser.add_argument("--model-config", type=Path, default=REPO_ROOT / "configs" / "model.yaml")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--context-mode", choices=["with_business_context", "without_business_context"], default=None)
    parser.add_argument("--dry-run", action="store_true", help="Validate infra only. No model, no CUDA.")
    args = parser.parse_args()

    model_cfg = load_model_config(args.model_config)
    context_mode = args.context_mode or model_cfg.prompt.context_mode

    # Fail closed before any manifest/model work: an obviously bad adapter
    # path should never get as far as loading the manifest or the model.
    try:
        validate_adapter_path(args.adapter)
    except AdapterValidationError as e:
        print(f"BLOCKER: {e}")
        sys.exit(1)

    manifest = load_manifest(args.manifest)
    if args.limit:
        manifest = manifest[: args.limit]
    if not manifest:
        print("BLOCKER: manifest is empty after applying --limit.")
        sys.exit(1)

    manifest_sha256 = file_sha256(args.manifest)
    run_config = RunConfig(
        run_id=args.run_id,
        model_id=adapter_model_id(model_cfg.model.id, args.adapter),
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
        run_dry_run(manifest, run_dir, run_config, args.adapter)
        return

    try:
        run_generation(manifest, model_cfg, context_mode, run_dir, run_config, args.adapter)
    except ResumeConflictError as e:
        print(f"BLOCKER: {e}")
        sys.exit(1)
    except AdapterValidationError as e:
        print(f"BLOCKER: {e}")
        sys.exit(1)
    except Exception as e:
        # Model/adapter load failures (no CUDA, OOM at load, bad revision,
        # etc.) stop the run outright -- no automatic fallback to the base
        # model, no retry.
        print(f"BLOCKER: run stopped -- {type(e).__name__}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
