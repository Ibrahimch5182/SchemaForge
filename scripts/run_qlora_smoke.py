"""Phase 4 QLoRA SMOKE TEST for LocalSQL -- NOT full-experiment training.

Proves the QLoRA training path (4-bit NF4 base, LoRA adapter, completion-
only loss) works end to end on Kaggle/T4, on a small subset and/or a
bounded number of optimizer steps. Reuses Phase 1's prepared `train.jsonl`
verbatim (prompt/completion/evidence-dropout already baked in) -- never
re-splits or re-derives the dataset. BIRD Mini-Dev is never touched here.

Modes:
  --dry-run        Validate data/config only. No model, no CUDA.
  --token-profile  Tokenizer-only profiling of the REAL training prompts
                    (and full SFT sequence length incl. completion). No
                    4-bit model load. Does NOT auto-adjust max_seq_length.
  (default)        Bounded QLoRA smoke training, then (unless
                    --skip-verify) an adapter-reload sanity check.
  --verify-adapter Standalone adapter-reload sanity check against an
                    existing run's saved adapter. NOT an accuracy eval.

Usage (on a CUDA cloud/Kaggle machine):
    uv sync --group model --group train
    uv run python scripts/run_qlora_smoke.py --run-id qlora-smoke-1 \\
        --max-train-examples 200 --max-steps 20

Usage (local Windows dev, no GPU -- infra validation only):
    uv run python scripts/run_qlora_smoke.py --run-id qlora-smoke-1-check \\
        --dry-run --max-train-examples 50
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from localsql.model.run_artifacts import numeric_stats  # noqa: E402
from localsql.train.config import (  # noqa: E402
    load_train_config,
    lora_summary,
    optimization_summary,
    quantization_summary,
)
from localsql.train.sft_data import build_sft_encoding, load_prepared_examples  # noqa: E402


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_adapter_verification(run_dir: Path, verification: dict) -> Path:
    """Persist the standalone `adapter_verification.json` artifact.

    Shared by both the post-training sanity check (run automatically at
    the end of `run_smoke_training`) and the standalone `--verify-adapter`
    mode, so the two paths can never drift out of sync again -- this fixes
    a bug where the post-training path only embedded the result into
    `summary.json` and never wrote this file.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / "adapter_verification.json"
    out_path.write_text(json.dumps(verification, indent=2), encoding="utf-8")
    return out_path


def load_examples(cfg, repo_root: Path, limit: int | None):
    train_path = repo_root / cfg.data.train_file
    if not train_path.exists():
        print(f"BLOCKER: training file not found: {train_path}")
        print("Run `uv run python scripts/prepare_bird.py` first (Phase 1).")
        sys.exit(1)
    examples = [e for e in load_prepared_examples(train_path) if e.split == "train"]
    if limit:
        examples = examples[:limit]
    return examples, train_path


def load_arbitrary_jsonl_for_profiling(path: Path, limit: int | None):
    """Load ANY JSONL with `example_id`/`prompt`/`completion` fields for
    `--token-profile` -- Phase 1's `train.jsonl`/`validation.jsonl`, or a
    Phase 5A candidate file (`data/processed_phase5_candidate/*.jsonl`).
    Unlike `load_examples`, this does not assume the strict
    `PreparedTextToSQLExample` schema or filter by split -- it profiles
    whatever rows are in the file, in file order.
    """
    if not path.exists():
        print(f"BLOCKER: input file not found: {path}")
        sys.exit(1)
    from types import SimpleNamespace

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
    return examples, path


def run_dry_run(cfg, examples, run_dir: Path) -> None:
    print(f"Training examples parsed OK: {len(examples)} (all split == 'train', gold completion present).")
    empty_completions = sum(1 for e in examples if not e.completion.strip())
    print(f"Empty completions: {empty_completions} (should be 0 -- Phase 1 already validated this).")
    if run_dir.exists() and (run_dir / "summary.json").exists():
        print(f"NOTE: run directory {run_dir} already has a completed summary.json -- "
              "a real run would refuse to overwrite it; use a new --run-id.")
    else:
        print(f"Run directory {run_dir} is fresh or incomplete -- a real run may proceed.")
    print("Dry run OK -- no model loaded, no CUDA required.")


def write_longest_examples_manifest(
    run_dir: Path, profile_name: str, example_ids: list[str], total_counts: list[float], top_n: int
) -> Path:
    """Deterministic manifest of the longest examples by real full-SFT
    token count -- for a later, separate GPU memory certification run
    (NOT performed here). Ties broken by `example_id` for determinism.
    """
    ranked = sorted(zip(example_ids, total_counts), key=lambda pair: (-pair[1], pair[0]))
    manifest = {
        "profile_name": profile_name,
        "top_n": top_n,
        "total_examples_considered": len(ranked),
        "entries": [
            {"rank": i + 1, "example_id": eid, "full_sft_token_count": count}
            for i, (eid, count) in enumerate(ranked[:top_n])
        ],
    }
    out_path = run_dir / f"{profile_name}_longest_examples.json"
    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out_path


def run_token_profile(
    cfg, examples, run_dir: Path, input_path: Path, profile_name: str, top_n_manifest: int = 50
) -> None:
    """Real-Qwen-tokenizer profiling of ANY prepared/candidate JSONL
    (`input_path`, defaults to Phase 1's train.jsonl via `--input`).
    Reports full-SFT and prompt-only distributions (count/min/median/p90/
    p95/p99/max, counts >3584/>4096/>8192) AND asserts the real prompt/
    full prefix boundary -- the same check Phase 4 validated at
    0/6,067 mismatches -- flagging any example where it fails rather than
    silently trusting the assumption.
    """
    from localsql.train.qlora_backend import QLoraBackend

    backend = QLoraBackend(cfg)
    resolved_revision = backend.load_tokenizer_only()
    print(f"Tokenizer loaded (revision {resolved_revision}). Profiling {len(examples)} examples from {input_path} ...")

    prompt_counts: list[float] = []
    total_counts: list[float] = []
    example_ids: list[str] = []
    prefix_mismatch_ids: list[str] = []
    for ex in examples:
        encoding = build_sft_encoding(backend._tokenizer, ex.example_id, ex.prompt, ex.completion, max_seq_length=None)
        prompt_counts.append(float(encoding.prompt_token_count))
        total_counts.append(float(encoding.total_token_count))
        example_ids.append(ex.example_id)
        if not encoding.prefix_matches:
            prefix_mismatch_ids.append(ex.example_id)

    thresholds = (3584, cfg.token_profile.warn_threshold, cfg.token_profile.hard_limit)

    def threshold_counts(values: list[float]) -> dict:
        return {f">{t}": sum(1 for v in values if v > t) for t in thresholds}

    report = {
        "input_file": str(input_path),
        "tokenizer_revision": resolved_revision,
        "example_count": len(examples),
        "prompt_only_token_stats": numeric_stats(prompt_counts),
        "prompt_only_counts_over_threshold": threshold_counts(prompt_counts),
        "full_sft_sequence_token_stats": numeric_stats(total_counts),
        "full_sft_sequence_counts_over_threshold": threshold_counts(total_counts),
        "prefix_boundary_check": {
            "description": (
                "Verifies the full sequence's prefix equals the standalone prompt-only "
                "encoding for every example -- the assumption completion-only masking "
                "depends on. Phase 4 validated this at 0/6,067 mismatches."
            ),
            "mismatch_count": len(prefix_mismatch_ids),
            "mismatch_example_ids": prefix_mismatch_ids[:50],  # cap -- see full report file if more
        },
        "configured_max_seq_length": cfg.sequence.max_seq_length,
        "note": "max_seq_length was NOT auto-adjusted from this profile -- review before changing configs/train.yaml.",
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    profile_path = run_dir / f"{profile_name}_token_profile.json"
    profile_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nWrote {profile_path}")
    if prefix_mismatch_ids:
        print(f"\nWARNING: {len(prefix_mismatch_ids)} example(s) failed the prefix-boundary check -- see report.")

    manifest_path = write_longest_examples_manifest(run_dir, profile_name, example_ids, total_counts, top_n_manifest)
    print(f"Wrote longest-examples certification manifest: {manifest_path}")
    print("(For a later, separate GPU memory certification run -- NOT performed by this command.)")


def run_smoke_training(cfg, examples, run_dir: Path, train_path: Path, max_steps: int, skip_verify: bool) -> None:
    from localsql.train.qlora_backend import NonFiniteLossError, QLoraBackend

    if run_dir.exists() and (run_dir / "summary.json").exists():
        print(f"BLOCKER: run directory {run_dir} already has a completed summary.json. "
              "This smoke runner does not support resume -- use a new --run-id.")
        sys.exit(1)
    run_dir.mkdir(parents=True, exist_ok=True)

    backend = QLoraBackend(cfg)
    print(f"Loading {cfg.model.id} for QLoRA training (4-bit {cfg.runtime.quantization}) ...")
    info = backend.load_for_training()
    print(f"Loaded. Resolved revision: {info.resolved_revision}  GPU: {info.gpu_name}")
    print(f"Trainable params: {info.trainable_param_count:,} / {info.total_param_count:,}")

    run_config = {
        "run_id": run_dir.name,
        "model_id": cfg.model.id,
        "resolved_revision": info.resolved_revision,
        "quantization": quantization_summary(cfg),
        "lora": lora_summary(cfg),
        "optimization": optimization_summary(cfg),
        "max_seq_length": cfg.sequence.max_seq_length,
        "completion_only_loss": cfg.loss.completion_only,
        "max_steps": max_steps,
        "requested_train_examples": len(examples),
        "train_file": str(train_path),
        "train_file_sha256": file_sha256(train_path),
    }
    (run_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    print(f"Building completion-only-masked SFT encodings for {len(examples)} examples ...")
    encodings = []
    skipped_over_length = 0
    for ex in examples:
        encoding = build_sft_encoding(
            backend._tokenizer, ex.example_id, ex.prompt, ex.completion, max_seq_length=cfg.sequence.max_seq_length
        )
        if encoding.exceeds_max_seq_length:
            skipped_over_length += 1
            continue
        encodings.append(encoding)
    print(f"  usable: {len(encodings)}  skipped (exceeds max_seq_length={cfg.sequence.max_seq_length}): {skipped_over_length}")
    if not encodings:
        print("BLOCKER: no usable training examples after filtering by max_seq_length.")
        sys.exit(1)

    checkpoint_dir = run_dir / "checkpoint"
    print(f"Training (max_steps={max_steps}) ...")
    try:
        train_result = backend.train_smoke(encodings, checkpoint_dir, max_steps=max_steps)
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

    summary = {
        "run_id": run_dir.name,
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
        "max_steps": max_steps,
        "requested_train_examples": len(examples),
        "usable_train_examples": len(encodings),
        "skipped_exceeds_max_seq_length": skipped_over_length,
        "global_step": train_result["global_step"],
        "runtime_seconds": train_result["runtime_seconds"],
        "peak_gpu_memory_mb": train_result["peak_gpu_memory_mb"],
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
            "cuda_version": info.cuda_version,
            "gpu_name": info.gpu_name,
            "gpu_total_memory_mb": info.gpu_total_memory_mb,
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSmoke training complete. Summary: {run_dir / 'summary.json'}")


def run_verify_adapter(cfg, run_dir: Path) -> None:
    from localsql.train.qlora_backend import QLoraBackend

    adapter_dir = run_dir / "adapter"
    if not adapter_dir.exists():
        print(f"BLOCKER: no saved adapter at {adapter_dir}. Run smoke training for this --run-id first.")
        sys.exit(1)

    backend = QLoraBackend(cfg)
    print(f"Reloading base model + adapter from {adapter_dir} ...")
    info = backend.load_adapter_for_verification(adapter_dir)
    print(f"adapter_active={info['adapter_active']}  adapter_names={info['adapter_names']}")

    sample_prompt = (
        "SYSTEM:\nYou are a text-to-SQL model.\nGenerate exactly one read-only SQL query "
        "that answers the question using only the provided database schema.\nReturn SQL "
        "only.\nDo not use markdown.\n\nDIALECT:\nsqlite\n\nSCHEMA:\ncustomers(\n  "
        "customer_id INTEGER PK\n)\n\nQUESTION:\nHow many customers are there?"
    )
    raw_completion = backend.generate_one(sample_prompt)
    print(f"Sample completion (NOT an accuracy check): {raw_completion!r}")

    result = {**info, "sample_raw_completion": raw_completion}
    out_path = write_adapter_verification(run_dir, result)
    print(f"\nWrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--train-config", type=Path, default=REPO_ROOT / "configs" / "train.yaml")
    parser.add_argument("--max-train-examples", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--token-profile", action="store_true")
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="For --token-profile: any JSONL with example_id/prompt/completion fields "
        "(e.g. data/processed_phase5_candidate/train.jsonl). Defaults to configs/train.yaml's train_file.",
    )
    parser.add_argument(
        "--top-n-manifest",
        type=int,
        default=50,
        help="For --token-profile: size of the longest-examples certification manifest.",
    )
    parser.add_argument("--verify-adapter", action="store_true")
    parser.add_argument("--skip-verify", action="store_true", help="Skip the post-training adapter sanity check.")
    args = parser.parse_args()

    cfg = load_train_config(args.train_config)
    runs_root = REPO_ROOT / cfg.paths["runs_dir"]
    run_dir = runs_root / args.run_id

    if args.verify_adapter:
        run_verify_adapter(cfg, run_dir)
        return

    if args.token_profile:
        if args.input:
            examples, input_path = load_arbitrary_jsonl_for_profiling(args.input, args.max_train_examples)
        else:
            examples, input_path = load_examples(cfg, REPO_ROOT, args.max_train_examples)
        if not examples:
            print("BLOCKER: no examples to profile after applying --max-train-examples.")
            sys.exit(1)
        run_token_profile(cfg, examples, run_dir, input_path, profile_name=input_path.stem, top_n_manifest=args.top_n_manifest)
        return

    examples, train_path = load_examples(cfg, REPO_ROOT, args.max_train_examples)
    if not examples:
        print("BLOCKER: no training examples after applying --max-train-examples.")
        sys.exit(1)

    if args.dry_run:
        run_dry_run(cfg, examples, run_dir)
        return

    try:
        run_smoke_training(cfg, examples, run_dir, train_path, args.max_steps, args.skip_verify)
    except Exception as e:
        print(f"BLOCKER: run stopped -- {type(e).__name__}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
