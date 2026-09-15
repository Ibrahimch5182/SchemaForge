"""Phase 5B Task 6: package a resumable checkpoint for durable transfer
(reliability-hardened pass).

Ephemeral Kaggle VM storage (e.g. /kaggle/working) is not assumed to
survive past the current session. This script packages the LATEST (or an
explicitly named) resumable Trainer checkpoint from a
`scripts/run_qlora_smoke.py` run -- LoRA adapter, optimizer, LR scheduler,
RNG state, trainer_state.json, training arguments -- together with the
EXACT training data JSONL that run used (byte-for-byte, SHA256-verified
against what the run recorded), the exact training config,
candidate-dataset policy/provenance, model/tokenizer revision, source
revision (explicit/provenance-file/best-effort git), environment/library
versions, and a SHA256 manifest, into one self-contained directory you
can copy anywhere (Kaggle Datasets, Drive, local disk) and later extract
into a fresh, compatible environment. It deliberately never packages
4-bit base model weights (see
`localsql.train.checkpoint_export.classify_checkpoint_files`).

FAILS CLOSED (no package written) if the selected checkpoint is missing
any required resumable-state category (`IncompleteCheckpointError`) or if
the training data can't be verified byte-for-byte against the run's
recorded SHA256 (`TrainingDataMismatchError`) -- see
`localsql.train.checkpoint_export` for the exact categories checked.

Usage:
    uv run python scripts/export_checkpoint.py --run-id phase5b-mem-cert
    uv run python scripts/export_checkpoint.py --run-id phase5b-resume-a \\
        --checkpoint checkpoint-2 --out-dir data/exports \\
        --source-revision <commit-hash>
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from localsql.train.checkpoint_export import (  # noqa: E402
    IncompleteCheckpointError,
    TrainingDataMismatchError,
    build_export,
    find_latest_checkpoint,
    render_resume_instructions,
    verify_training_data_snapshot,
)
from localsql.train.provenance import resolve_source_revision  # noqa: E402

CANDIDATE_POLICY_PATH = REPO_ROOT / "data" / "processed_phase5_candidate" / "policy.json"
TRAIN_CONFIG_PATH = REPO_ROOT / "configs" / "train.yaml"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint dir name (e.g. checkpoint-2) under the run's checkpoint/ dir. "
        "Defaults to the latest (highest global_step) checkpoint found.",
    )
    parser.add_argument("--runs-dir", type=Path, default=REPO_ROOT / "data" / "runs")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data" / "exports")
    parser.add_argument(
        "--source-revision",
        type=str,
        default=None,
        help="Explicit source-repo revision to record for this export. Takes priority over a "
        "SOURCE_REVISION file at the repo root and a best-effort `git rev-parse HEAD` -- needed "
        "because `git archive`/Kaggle upload strips .git.",
    )
    args = parser.parse_args()

    run_dir = args.runs_dir / args.run_id
    run_config_path = run_dir / "run_config.json"
    checkpoint_root = run_dir / "checkpoint"

    if not run_config_path.exists():
        print(f"BLOCKER: {run_config_path} not found. Run scripts/run_qlora_smoke.py with this --run-id first.")
        sys.exit(1)
    if not CANDIDATE_POLICY_PATH.exists():
        print(f"BLOCKER: {CANDIDATE_POLICY_PATH} not found. Run scripts/build_phase5_candidate.py first.")
        sys.exit(1)

    run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else None

    if args.checkpoint:
        checkpoint_dir = checkpoint_root / args.checkpoint
        if not checkpoint_dir.exists():
            print(f"BLOCKER: checkpoint directory not found: {checkpoint_dir}")
            sys.exit(1)
    else:
        checkpoint_dir = find_latest_checkpoint(checkpoint_root)
        if checkpoint_dir is None:
            print(
                f"BLOCKER: no checkpoint-N directories found under {checkpoint_root}. "
                "Run scripts/run_qlora_smoke.py with --save-steps set first."
            )
            sys.exit(1)

    trainer_state_path = checkpoint_dir / "trainer_state.json"
    global_step = None
    if trainer_state_path.exists():
        global_step = json.loads(trainer_state_path.read_text(encoding="utf-8")).get("global_step")

    # Exact training-data snapshot: the file this run actually used, verified
    # byte-for-byte against what the run recorded at training time. Fails
    # closed (no package written) on any mismatch -- resuming Trainer state
    # against a different/reordered dataset is unsafe.
    train_file = Path(run_config["train_file"])
    if not train_file.is_absolute():
        train_file = REPO_ROOT / train_file
    try:
        training_data_sha256 = verify_training_data_snapshot(train_file, run_config.get("train_file_sha256"))
    except TrainingDataMismatchError as e:
        print(f"BLOCKER: {e}")
        sys.exit(1)

    export_source_revision = resolve_source_revision(args.source_revision, REPO_ROOT)

    export_name = f"{args.run_id}__{checkpoint_dir.name}"
    export_dir = args.out_dir / export_name

    resume_instructions = render_resume_instructions(
        source_run_id=args.run_id,
        checkpoint_name=checkpoint_dir.name,
        global_step=global_step,
        model_id=run_config.get("model_id", "unknown"),
        resolved_revision=run_config.get("resolved_revision"),
        train_file=run_config.get("train_file"),
        max_seq_length=run_config.get("max_seq_length"),
        save_steps=run_config.get("save_steps"),
        save_total_limit=run_config.get("save_total_limit"),
        source_revision=export_source_revision["source_revision"],
        source_revision_origin=export_source_revision["source_revision_origin"],
    )

    extra_files = {
        "training_data.jsonl": train_file,
        "train_config.yaml": TRAIN_CONFIG_PATH,
        "candidate_policy.json": CANDIDATE_POLICY_PATH,
        "run_config.json": run_config_path,
    }
    if summary_path.exists():
        extra_files["summary.json"] = summary_path

    training_time_provenance = (summary or {}).get("provenance", {})
    provenance = {
        "training_time": training_time_provenance,
        "export_time": {
            "python_version": platform.python_version(),
            "source_revision": export_source_revision["source_revision"],
            "source_revision_origin": export_source_revision["source_revision_origin"],
        },
        "training_data": {
            "published_as": "training_data.jsonl",
            "original_path": run_config["train_file"],
            "sha256": training_data_sha256,
            "verified_against_run_config_train_file_sha256": run_config.get("train_file_sha256") is not None,
        },
    }

    try:
        manifest = build_export(checkpoint_dir, export_dir, extra_files, resume_instructions, provenance=provenance)
    except IncompleteCheckpointError as e:
        print(f"BLOCKER: {e}")
        sys.exit(1)
    except FileExistsError as e:
        print(f"BLOCKER: {e}")
        sys.exit(1)
    except AssertionError as e:
        print(f"BLOCKER: {e}")
        sys.exit(1)

    print(f"Exported checkpoint '{checkpoint_dir.name}' from run '{args.run_id}' -> {export_dir}")
    print(f"  global_step at export: {global_step}")
    print(f"  training data packaged: {run_config['train_file']} (sha256={training_data_sha256})")
    print(f"  files packaged: {len(manifest['sha256'])}")
    if manifest["excluded_checkpoint_files"]:
        print(f"  excluded checkpoint files (see MANIFEST.json for reasons): {len(manifest['excluded_checkpoint_files'])}")
    print(
        f"  source revision: {export_source_revision['source_revision'] or 'unavailable'} "
        f"(origin: {export_source_revision['source_revision_origin']})"
    )
    print(f"\nWrote {export_dir / 'MANIFEST.json'} and {export_dir / 'RESUME_INSTRUCTIONS.md'}")


if __name__ == "__main__":
    main()
