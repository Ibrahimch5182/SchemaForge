"""Phase 5C -- canonical resumable full-training runner.

This is the CANONICAL full-experiment QLoRA training path (2 epochs over
the full Phase 5 candidate dataset), deliberately a separate, clearly
named script from `scripts/run_qlora_smoke.py` (Phase 4 smoke test /
Phase 5B GPU-memory and throughput certification / stop-resume proof) --
it is not another smoke test, even though it reuses the SAME
`localsql.train.qlora_backend.QLoraBackend` for all actual training logic.

Canonical dataset (never re-split, never re-derived here):
    data/processed_phase5_candidate/train.jsonl        (6,067 examples)
    data/processed_phase5_candidate/validation.jsonl   (534 examples, not
        used for loss in this phase -- no generation-based or loss-based
        validation subsystem is introduced here; validation.jsonl is only
        recorded for provenance).

CRITICAL: multi-session training must be mathematically equivalent to one
uninterrupted `--num-train-epochs` run. This script NEVER changes
`TrainingArguments.max_steps`/`num_train_epochs` between sessions --
`--stop-after-global-step` only requests a graceful, checkpointed early
stop via a `TrainerCallback` (see `QLoraBackend.train_smoke`'s
`stop_after_global_step` parameter); the scheduler's total-step horizon,
computed from `num_train_epochs` and the (fixed) dataset size, is
identical on every session as long as `--num-train-epochs` and the
training file are unchanged. A resume validates that against the
checkpoint's source run where practical (see
`localsql.train.full_training.validate_resume_compatibility`) and refuses
to proceed on a mismatch.

Usage (on a CUDA cloud/Kaggle machine):
    uv sync --group model --group train

    # Session 1 -- fresh start, canonical 2-epoch horizon (1,518 total
    # optimizer steps for the real 6,067-example candidate set), stopping
    # after 400 optimizer steps for this session:
    uv run python scripts/run_qlora_full_training.py --run-id phase5c-session-1 \\
        --save-steps 100 --save-total-limit 3 --stop-after-global-step 400 \\
        --source-revision <commit>

    # Session 2 -- a NEW run-id, explicit resume from session 1's
    # checkpoint, continuing the SAME fixed 2-epoch horizon (never
    # restarted):
    uv run python scripts/run_qlora_full_training.py --run-id phase5c-session-2 \\
        --save-steps 100 --save-total-limit 3 --stop-after-global-step 800 \\
        --resume-from-checkpoint data/runs/phase5c-session-1/checkpoint/checkpoint-400 \\
        --source-revision <commit>

    # No --stop-after-global-step: run to canonical completion (all
    # num_train_epochs epochs) in one session, if wall-clock allows.
    uv run python scripts/run_qlora_full_training.py --run-id phase5c-full-run \\
        --save-steps 100 --save-total-limit 3 --source-revision <commit>

Usage (local Windows dev, no GPU -- infra validation only):
    uv run python scripts/run_qlora_full_training.py --run-id phase5c-check \\
        --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from localsql.train.config import (  # noqa: E402
    load_train_config,
    lora_summary,
    optimization_summary,
    quantization_summary,
)
from localsql.train.full_training import (  # noqa: E402
    compute_optimizer_step_schedule,
    validate_resume_compatibility,
)
from localsql.train.provenance import resolve_source_revision  # noqa: E402
from localsql.train.sft_data import build_sft_encoding  # noqa: E402

CANONICAL_TRAIN_FILE = REPO_ROOT / "data" / "processed_phase5_candidate" / "train.jsonl"
CANONICAL_VALIDATION_FILE = REPO_ROOT / "data" / "processed_phase5_candidate" / "validation.jsonl"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_candidate_examples(path: Path, limit: int | None):
    """Loose loader for the Phase 5 candidate JSONL shape (`serialized_schema`,
    `representation`, `policy_basis` -- NOT the strict Phase 1
    `PreparedTextToSQLExample` schema). Mirrors
    `scripts/run_qlora_smoke.py`'s `load_arbitrary_jsonl_for_profiling` --
    only `example_id`/`prompt`/`completion` are needed for training."""
    if not path.exists():
        print(f"BLOCKER: canonical training file not found: {path}")
        print("Run `uv run python scripts/build_phase5_candidate.py` first (Phase 5A).")
        sys.exit(1)
    examples = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            examples.append(
                SimpleNamespace(example_id=row["example_id"], prompt=row["prompt"], completion=row["completion"])
            )
    if limit:
        examples = examples[:limit]
    return examples


def write_adapter_verification(run_dir: Path, verification: dict) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / "adapter_verification.json"
    out_path.write_text(json.dumps(verification, indent=2), encoding="utf-8")
    return out_path


def run_dry_run(cfg, examples, num_train_epochs: int, run_dir: Path, train_path: Path) -> None:
    schedule = compute_optimizer_step_schedule(
        num_examples=len(examples),
        per_device_train_batch_size=cfg.optimization.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.optimization.gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
    )
    print(f"Canonical training file: {train_path} ({len(examples)} examples)")
    print(f"Expected optimizer steps/epoch: {schedule['steps_per_epoch']}")
    print(f"Expected total optimizer steps ({num_train_epochs} epochs): {schedule['total_optimizer_steps']}")
    if run_dir.exists() and (run_dir / "summary.json").exists():
        print(f"NOTE: run directory {run_dir} already has a completed summary.json -- "
              "a real run would refuse to overwrite it; use a new --run-id.")
    print("Dry run OK -- no model loaded, no CUDA required.")


def run_full_training(
    cfg,
    examples,
    run_dir: Path,
    train_path: Path,
    num_train_epochs: int,
    stop_after_global_step: int | None,
    skip_verify: bool,
    save_steps: int | None,
    save_total_limit: int | None,
    resume_from_checkpoint: str | None,
    source_revision: str | None,
) -> None:
    from localsql.train.qlora_backend import NonFiniteLossError, QLoraBackend

    if run_dir.exists() and (run_dir / "summary.json").exists():
        print(f"BLOCKER: run directory {run_dir} already has a completed summary.json. "
              "This runner does not overwrite a completed run -- use a new --run-id "
              "(pointing --resume-from-checkpoint at the prior session's checkpoint dir if continuing it).")
        sys.exit(1)
    run_dir.mkdir(parents=True, exist_ok=True)

    if resume_from_checkpoint and not Path(resume_from_checkpoint).exists():
        print(f"BLOCKER: --resume-from-checkpoint path does not exist: {resume_from_checkpoint}")
        sys.exit(1)

    schedule = compute_optimizer_step_schedule(
        num_examples=len(examples),
        per_device_train_batch_size=cfg.optimization.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.optimization.gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
    )
    print(f"Canonical schedule: {schedule['steps_per_epoch']} steps/epoch x {num_train_epochs} epochs "
          f"= {schedule['total_optimizer_steps']} total optimizer steps.")

    if stop_after_global_step is not None:
        if stop_after_global_step <= 0:
            print(f"BLOCKER: --stop-after-global-step must be positive, got {stop_after_global_step}.")
            sys.exit(1)
        if stop_after_global_step > schedule["total_optimizer_steps"]:
            print(
                f"BLOCKER: --stop-after-global-step={stop_after_global_step} exceeds the canonical total "
                f"optimizer-step horizon ({schedule['total_optimizer_steps']}) for {num_train_epochs} epochs "
                f"over {len(examples)} examples. Omit --stop-after-global-step to run to canonical completion."
            )
            sys.exit(1)

    train_file_sha256 = file_sha256(train_path)

    # Resume-compatibility validation -- BEFORE any GPU load. Refuses to
    # silently continue training under different canonical settings than
    # the checkpoint was produced under. Skipped only when the source
    # run's run_config.json genuinely isn't available (nothing to
    # validate against), never skipped just because it's inconvenient.
    if resume_from_checkpoint:
        current_config_for_validation = {
            "model_id": cfg.model.id,
            "max_seq_length": cfg.sequence.max_seq_length,
            "quantization": quantization_summary(cfg),
            "lora": lora_summary(cfg),
            "canonical_num_train_epochs": num_train_epochs,
            "canonical_total_optimizer_steps": schedule["total_optimizer_steps"],
            "train_file_sha256": train_file_sha256,
        }
        source_run_config_path = Path(resume_from_checkpoint).parent.parent / "run_config.json"
        if source_run_config_path.exists():
            source_run_config = json.loads(source_run_config_path.read_text(encoding="utf-8"))
            mismatches = validate_resume_compatibility(current_config_for_validation, source_run_config)
            if mismatches:
                print("BLOCKER: resume settings do not match the checkpoint's source run_config.json:")
                for m in mismatches:
                    print(f"  - {m}")
                print("Refusing to resume under different canonical settings than the checkpoint was produced under.")
                sys.exit(1)
            print(f"Resume-compatibility check passed against {source_run_config_path}.")
        else:
            print(
                f"NOTE: {source_run_config_path} not found -- cannot validate resume compatibility against it "
                "(nothing else to check against here); proceeding."
            )

    backend = QLoraBackend(cfg)
    print(f"Loading {cfg.model.id} for QLoRA training (4-bit {cfg.runtime.quantization}) ...")
    info = backend.load_for_training()
    print(f"Loaded. Resolved revision: {info.resolved_revision}  GPU: {info.gpu_name}")
    print(f"Trainable params: {info.trainable_param_count:,} / {info.total_param_count:,}")

    source_revision_info = resolve_source_revision(source_revision, REPO_ROOT)

    checkpoint_dir = run_dir / "checkpoint"
    run_config = {
        "run_id": run_dir.name,
        "run_type": "phase5c_canonical_full_training",
        "model_id": cfg.model.id,
        "resolved_revision": info.resolved_revision,
        "quantization": quantization_summary(cfg),
        "lora": lora_summary(cfg),
        "optimization": optimization_summary(cfg),
        "max_seq_length": cfg.sequence.max_seq_length,
        "completion_only_loss": cfg.loss.completion_only,
        "canonical_num_train_epochs": num_train_epochs,
        "canonical_total_optimizer_steps": schedule["total_optimizer_steps"],
        "expected_steps_per_epoch": schedule["steps_per_epoch"],
        "requested_stop_after_global_step": stop_after_global_step,
        "requested_train_examples": len(examples),
        "train_file": str(train_path),
        "train_file_sha256": train_file_sha256,
        "validation_file": str(CANONICAL_VALIDATION_FILE),
        "checkpoint_dir": str(checkpoint_dir),
        "save_steps": save_steps,
        "save_total_limit": save_total_limit,
        # Explicit resume source -- never inferred/auto-discovered. None
        # here means this session started this horizon from scratch.
        "resume_from_checkpoint": resume_from_checkpoint,
        "source_revision": source_revision_info["source_revision"],
        "source_revision_origin": source_revision_info["source_revision_origin"],
    }
    (run_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    print(f"Building completion-only-masked SFT encodings for {len(examples)} examples ...")
    encodings = []
    skipped_over_length = []
    for ex in examples:
        encoding = build_sft_encoding(
            backend._tokenizer, ex.example_id, ex.prompt, ex.completion, max_seq_length=cfg.sequence.max_seq_length
        )
        if encoding.exceeds_max_seq_length:
            skipped_over_length.append(ex.example_id)
            continue
        encodings.append(encoding)
    print(f"  usable: {len(encodings)}  skipped (exceeds max_seq_length={cfg.sequence.max_seq_length}): {len(skipped_over_length)}")
    if skipped_over_length:
        print(f"  NOTE: skipped example_ids: {skipped_over_length}")
    if not encodings:
        print("BLOCKER: no usable training examples after filtering by max_seq_length.")
        sys.exit(1)

    sequence_lengths = [e.total_token_count for e in encodings]
    print(
        f"Training (num_train_epochs={num_train_epochs}, stop_after_global_step={stop_after_global_step}, "
        f"resume_from_checkpoint={resume_from_checkpoint!r}) ..."
    )
    try:
        train_result = backend.train_smoke(
            encodings,
            checkpoint_dir,
            num_train_epochs=num_train_epochs,
            stop_after_global_step=stop_after_global_step,
            save_steps=save_steps,
            save_total_limit=save_total_limit,
            resume_from_checkpoint=resume_from_checkpoint,
        )
    except NonFiniteLossError as e:
        print(f"BLOCKER: {e}")
        sys.exit(1)

    adapter_dir = run_dir / "adapter"
    backend.save_adapter(adapter_dir)
    print(f"Adapter saved: {adapter_dir}")

    metrics_path = run_dir / "train_metrics.jsonl"
    with metrics_path.open("w", encoding="utf-8") as f:
        for entry in train_result["metrics_log"]:
            f.write(json.dumps(entry) + "\n")

    verification = None
    if not skip_verify and encodings:
        print("Running adapter-reload sanity check (NOT an accuracy evaluation) ...")
        verify_backend = QLoraBackend(cfg)
        verify_info = verify_backend.load_adapter_for_verification(adapter_dir)
        sample_prompt = examples[0].prompt
        raw_completion = verify_backend.generate_one(sample_prompt)
        verification = {
            **verify_info,
            "sample_example_id": examples[0].example_id,
            "sample_raw_completion": raw_completion,
        }
        verification_path = write_adapter_verification(run_dir, verification)
        print(f"  adapter_active={verification['adapter_active']}  sample completion: {raw_completion[:120]!r}")
        print(f"  Wrote {verification_path}")

    session_end_reason = "planned_boundary" if train_result["ended_by_planned_boundary"] else "canonical_completion"

    summary = {
        "run_id": run_dir.name,
        "run_type": "phase5c_canonical_full_training",
        "model_id": cfg.model.id,
        "resolved_revision": info.resolved_revision,
        "tokenizer_revision": info.tokenizer_revision,
        "quantization": info.quantization,
        "lora": info.lora,
        "trainable_param_count": info.trainable_param_count,
        "total_param_count": info.total_param_count,
        "optimization": optimization_summary(cfg),
        "max_seq_length": cfg.sequence.max_seq_length,
        "completion_only_loss": cfg.loss.completion_only,
        "canonical_num_train_epochs": num_train_epochs,
        "canonical_total_optimizer_steps": schedule["total_optimizer_steps"],
        "expected_steps_per_epoch": schedule["steps_per_epoch"],
        "requested_stop_after_global_step": stop_after_global_step,
        "requested_train_examples": len(examples),
        "usable_train_examples": len(encodings),
        "skipped_exceeds_max_seq_length_count": len(skipped_over_length),
        "skipped_exceeds_max_seq_length_example_ids": skipped_over_length,
        "any_sequence_truncated": False,  # structural guarantee: build_sft_encoding never truncates
        "sequence_length_min": min(sequence_lengths) if sequence_lengths else None,
        "sequence_length_max": max(sequence_lengths) if sequence_lengths else None,
        "starting_global_step": train_result["starting_global_step"],
        "final_global_step": train_result["global_step"],
        "resumed_from_checkpoint": train_result["resumed_from_checkpoint"],
        "session_end_reason": session_end_reason,
        "checkpoint_dir": str(checkpoint_dir),
        "save_steps": save_steps,
        "save_total_limit": save_total_limit,
        "runtime_seconds": train_result["runtime_seconds"],
        "peak_gpu_memory_allocated_mb": train_result["peak_gpu_memory_allocated_mb"],
        "peak_gpu_memory_reserved_mb": train_result["peak_gpu_memory_reserved_mb"],
        "train_result_metrics": train_result["train_result_metrics"],
        "adapter_path": str(adapter_dir),
        "adapter_verification": verification,
        "train_file_sha256": run_config["train_file_sha256"],
        "provenance": {
            "python_version": platform.python_version(),
            "torch_version": info.torch_version,
            "transformers_version": info.transformers_version,
            "peft_version": info.peft_version,
            "bitsandbytes_version": info.bitsandbytes_version,
            "accelerate_version": info.accelerate_version,
            "trl_version": info.trl_version,
            "cuda_version": info.cuda_version,
            "gpu_name": info.gpu_name,
            "gpu_total_memory_mb": info.gpu_total_memory_mb,
            "source_revision": source_revision_info["source_revision"],
            "source_revision_origin": source_revision_info["source_revision_origin"],
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSession complete ({session_end_reason}). final_global_step={train_result['global_step']} "
          f"of canonical total {schedule['total_optimizer_steps']}. Summary: {run_dir / 'summary.json'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--train-config", type=Path, default=REPO_ROOT / "configs" / "train.yaml")
    parser.add_argument("--input", type=Path, default=CANONICAL_TRAIN_FILE, help="Canonical training JSONL override (defaults to the Phase 5 candidate dataset).")
    parser.add_argument("--max-train-examples", type=int, default=None, help="Debug-only cap; the canonical run uses the full dataset.")
    parser.add_argument(
        "--num-train-epochs",
        type=int,
        default=None,
        help="Canonical fixed training horizon. Defaults to configs/train.yaml's "
        "optimization.planned_full_experiment_epochs (2).",
    )
    parser.add_argument(
        "--stop-after-global-step",
        type=int,
        default=None,
        help="Request a graceful, checkpointed stop once this optimizer step is reached, WITHOUT changing "
        "num_train_epochs or the scheduler's total-step horizon. Omit to run to canonical completion.",
    )
    parser.add_argument("--save-steps", type=int, default=None)
    parser.add_argument("--save-total-limit", type=int, default=None)
    parser.add_argument("--resume-from-checkpoint", type=str, default=None)
    parser.add_argument("--source-revision", type=str, default=None)
    parser.add_argument("--skip-verify", action="store_true", help="Skip the post-training adapter sanity check.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_train_config(args.train_config)
    num_train_epochs = args.num_train_epochs or cfg.optimization.planned_full_experiment_epochs
    runs_root = REPO_ROOT / cfg.paths["runs_dir"]
    run_dir = runs_root / args.run_id

    examples = load_candidate_examples(args.input, args.max_train_examples)
    if not examples:
        print("BLOCKER: no training examples after applying --max-train-examples.")
        sys.exit(1)

    if args.dry_run:
        run_dry_run(cfg, examples, num_train_epochs, run_dir, args.input)
        return

    try:
        run_full_training(
            cfg,
            examples,
            run_dir,
            args.input,
            num_train_epochs,
            args.stop_after_global_step,
            args.skip_verify,
            args.save_steps,
            args.save_total_limit,
            args.resume_from_checkpoint,
            args.source_revision,
        )
    except Exception as e:
        print(f"BLOCKER: run stopped -- {type(e).__name__}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
