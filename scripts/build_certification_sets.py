"""Phase 5B: build the two GPU/throughput certification datasets.

Both are SCRATCH/CERTIFICATION data -- never canonical training data, never
touched by `scripts/run_qlora_smoke.py`'s normal training path unless
explicitly pointed at one of these files via `--input`/an override.

1. Longest-16 set (Task 1): exact copies of the 16 real-tokenizer-longest
   candidate training examples, per the Kaggle-produced
   `train_longest_examples.json`. Used for worst-case T4 memory
   certification (Task 2 -- run separately on Kaggle, not here).
2. Throughput sample (Task 3, real-token-manifest hardening pass): a
   deterministic, equally-spaced-by-real-rank ~64-example sample built
   ONLY from a real per-example token-length manifest
   (`<profile>_token_lengths.jsonl`, produced by a real Kaggle tokenizer
   run -- see `scripts/run_qlora_smoke.py`'s `--token-profile`). The
   earlier estimator-based (`localsql.schema_context.token_estimate`)
   throughput sample is NOT accepted for the canonical benchmark and is
   no longer built here -- this script BLOCKERs with the exact Kaggle
   command if the real manifest isn't available yet.

No gold SQL used anywhere. No content mutated or truncated. Fully offline
GIVEN the real token-length manifest already exists.

Usage:
    uv run python scripts/build_certification_sets.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from localsql.train.certification import (  # noqa: E402
    ThroughputSampleRepresentationError,
    build_longest_n_set,
    build_real_token_throughput_sample,
    file_sha256,
    load_jsonl_index,
    load_token_length_manifest,
    summarize_lengths,
)

CANDIDATE_TRAIN_PATH = REPO_ROOT / "data" / "processed_phase5_candidate" / "train.jsonl"
LONGEST_MANIFEST_PATH = (
    REPO_ROOT
    / "kaggle-phase5-tokenizer-evidence"
    / "phase5-candidate-train-profile"
    / "train_longest_examples.json"
)
TOKEN_LENGTHS_MANIFEST_PATH = (
    REPO_ROOT
    / "kaggle-phase5-tokenizer-evidence"
    / "phase5-candidate-train-profile"
    / "train_token_lengths.jsonl"
)
OUT_DIR = REPO_ROOT / "data" / "certification"

NEXT_KAGGLE_COMMAND = (
    "uv sync --group model --group train\n"
    "uv run python scripts/run_qlora_smoke.py --run-id phase5-candidate-profile \\\n"
    "    --input data/processed_phase5_candidate/train.jsonl --token-profile"
)


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--longest-n", type=int, default=16)
    parser.add_argument("--throughput-n", type=int, default=64)
    parser.add_argument("--candidate-train", type=Path, default=CANDIDATE_TRAIN_PATH)
    parser.add_argument("--longest-manifest", type=Path, default=LONGEST_MANIFEST_PATH)
    parser.add_argument("--token-lengths-manifest", type=Path, default=TOKEN_LENGTHS_MANIFEST_PATH)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    if not args.candidate_train.exists():
        print(f"BLOCKER: {args.candidate_train} not found. Run scripts/build_phase5_candidate.py first.")
        sys.exit(1)
    if not args.longest_manifest.exists():
        print(
            f"BLOCKER: {args.longest_manifest} not found. This must be the real-tokenizer-generated "
            "train_longest_examples.json from a Kaggle --token-profile run against the candidate "
            "dataset -- it is not something this offline script can produce itself."
        )
        sys.exit(1)

    candidate_sha256 = file_sha256(args.candidate_train)
    candidate_index = load_jsonl_index(args.candidate_train)
    longest_manifest = json.loads(args.longest_manifest.read_text(encoding="utf-8"))

    # --- Task 1: longest-N certification set (unchanged methodology) ---
    longest_set = build_longest_n_set(longest_manifest, candidate_index, args.longest_n)
    assert len(longest_set) == args.longest_n
    assert len(set(e["example_id"] for e in longest_set)) == args.longest_n

    longest_out = args.out_dir / f"longest_{args.longest_n}.jsonl"
    write_jsonl(longest_out, longest_set)
    manifest_sha256 = file_sha256(args.longest_manifest)

    real_lengths = {
        e["example_id"]: e["full_sft_token_count"]
        for e in sorted(longest_manifest["entries"], key=lambda e: e["rank"])[: args.longest_n]
    }
    longest_report = {
        "task": "longest_n_certification_set",
        "n": args.longest_n,
        "example_ids_descending_length": [e["example_id"] for e in longest_set],
        "real_full_sft_token_lengths": [real_lengths[e["example_id"]] for e in longest_set],
        "min_real_full_sft_token_length": min(real_lengths.values()),
        "max_real_full_sft_token_length": max(real_lengths.values()),
        "source_candidate_train_path": str(args.candidate_train),
        "source_candidate_train_sha256": candidate_sha256,
        "source_longest_manifest_path": str(args.longest_manifest),
        "source_longest_manifest_sha256": manifest_sha256,
        "output_path": str(longest_out),
        "output_sha256": file_sha256(longest_out),
    }
    (args.out_dir / f"longest_{args.longest_n}_report.json").write_text(
        json.dumps(longest_report, indent=2), encoding="utf-8"
    )
    print(f"Longest-{args.longest_n} certification set: {longest_out}")
    print(
        f"  real length range: {longest_report['min_real_full_sft_token_length']} - "
        f"{longest_report['max_real_full_sft_token_length']} tokens"
    )

    # --- Task 3: canonical throughput sample (real-token manifest only) ---
    if not args.token_lengths_manifest.exists():
        print(
            f"\nBLOCKER: {args.token_lengths_manifest} not found. The canonical Phase 5B throughput "
            "sample requires a REAL per-example token-length manifest -- the estimator-based approach "
            "was superseded and is not accepted. Generate it on Kaggle first with:\n\n"
            f"{NEXT_KAGGLE_COMMAND}\n\n"
            "which writes data/runs/phase5-candidate-profile/train_token_lengths.jsonl -- copy that "
            "file to kaggle-phase5-tokenizer-evidence/phase5-candidate-train-profile/train_token_lengths.jsonl "
            "(or pass --token-lengths-manifest) and rerun this script."
        )
        sys.exit(1)

    token_length_records = load_token_length_manifest(args.token_lengths_manifest)
    token_lengths_manifest_sha256 = file_sha256(args.token_lengths_manifest)
    real_lengths_by_id = {r["example_id"]: r["full_sft_token_count"] for r in token_length_records}

    try:
        throughput_sample = build_real_token_throughput_sample(
            token_length_records, candidate_index, target_n=args.throughput_n
        )
    except ThroughputSampleRepresentationError as e:
        print(f"\nBLOCKER: {e}")
        sys.exit(1)

    assert len(throughput_sample) == len(set(e["example_id"] for e in throughput_sample)), "duplicate ids in sample"

    throughput_out = args.out_dir / f"throughput_sample_{args.throughput_n}.jsonl"
    write_jsonl(throughput_out, throughput_sample)

    length_summary = summarize_lengths(throughput_sample, real_lengths_by_id)
    throughput_report = {
        "task": "real_token_throughput_sample",
        "target_n": args.throughput_n,
        "actual_n": len(throughput_sample),
        "selection_logic": (
            "REAL Qwen-tokenizer full-SFT token counts only (never the local character-count "
            "estimator). All profiled training examples sorted by (full_sft_token_count, example_id) "
            "ascending; target_n approximately equally spaced ranks selected via "
            "localsql.train.certification.select_equally_spaced_by_real_rank (round-half-up rounding "
            "rule, documented in code), spanning the shortest to the longest real example. No gold SQL, "
            "no difficulty labels, no random sampling, no hand-picked substitutions -- a representation "
            "gap fails closed (ThroughputSampleRepresentationError / BLOCKER) rather than being patched."
        ),
        "length_summary_real_tokens": length_summary,
        "not_for_model_quality": True,
        "source_candidate_train_sha256": candidate_sha256,
        "source_token_lengths_manifest_path": str(args.token_lengths_manifest),
        "source_token_lengths_manifest_sha256": token_lengths_manifest_sha256,
        "output_path": str(throughput_out),
        "output_sha256": file_sha256(throughput_out),
    }
    (args.out_dir / f"throughput_sample_{args.throughput_n}_report.json").write_text(
        json.dumps(throughput_report, indent=2), encoding="utf-8"
    )
    print(f"\nThroughput sample ({len(throughput_sample)} examples, REAL token lengths): {throughput_out}")
    print(f"  {length_summary}")


if __name__ == "__main__":
    main()
