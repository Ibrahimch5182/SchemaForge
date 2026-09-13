"""Evaluate LocalSQL predictions against the BIRD Mini-Dev benchmark (Phase 2).

No model inference happens here -- this only scores an existing prediction
file. Supports the official EX/Soft-F1 evaluator (vendored, unmodified) plus
LocalSQL-only diagnostics (parse rate, execution success), clearly separated
from official correctness.

Usage:
    uv run python scripts/evaluate_bird_minidev.py --predictions path\\to\\predictions.jsonl

Oracle sanity mode (plumbing check only -- NEVER a model result):
    uv run python scripts/evaluate_bird_minidev.py --oracle-sanity --limit 5
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

from localsql.benchmark.diagnostics import diagnose_prediction  # noqa: E402
from localsql.benchmark.minidev_loader import load_gold_sql  # noqa: E402
from localsql.benchmark.models import GenerationExample, GradingExample  # noqa: E402
from localsql.benchmark.official_adapter import (  # noqa: E402
    build_diff_jsonl,
    build_official_prediction_file,
    fetch_official_evaluator,
    run_official_ex,
    run_official_f1,
)
from localsql.benchmark.prediction import load_predictions_jsonl, validate_predictions  # noqa: E402
from localsql.benchmark.report import build_report  # noqa: E402


def load_config() -> dict:
    return yaml.safe_load((REPO_ROOT / "configs" / "benchmark.yaml").read_text(encoding="utf-8"))


def load_jsonl_models(path: Path, model_cls):
    with path.open(encoding="utf-8") as f:
        return [model_cls.model_validate_json(line) for line in f if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=None, help="LocalSQL predictions JSONL")
    parser.add_argument("--oracle-sanity", action="store_true", help="Use gold SQL as predictions (plumbing check only)")
    parser.add_argument("--mode", choices=["official", "diagnostics", "both"], default="both")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N examples (smoke testing)")
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--model-id", type=str, default=None)
    parser.add_argument("--num-cpus", type=int, default=None)
    parser.add_argument("--timeout-seconds", type=float, default=None)
    args = parser.parse_args()

    if not args.oracle_sanity and args.predictions is None:
        parser.error("either --predictions or --oracle-sanity is required")

    config = load_config()
    paths = {k: REPO_ROOT / v for k, v in config["paths"].items()}
    num_cpus = args.num_cpus or config["execution"]["num_cpus"]
    timeout_seconds = args.timeout_seconds or config["execution"]["timeout_seconds"]
    run_id = args.run_id or f"run-{int(time.time())}"

    grading_examples = load_jsonl_models(paths["grading"] / "reference.jsonl", GradingExample)
    generation_examples = load_jsonl_models(paths["generation"] / "manifest.jsonl", GenerationExample)
    if args.limit:
        grading_examples = grading_examples[: args.limit]
        generation_examples = generation_examples[: args.limit]
    selected_ids = {ex.example_id for ex in grading_examples}

    # --- gather predictions ---
    if args.oracle_sanity:
        print("*** ORACLE SANITY MODE: using gold SQL as predictions. NOT a model result. ***")
        predictions_by_id = {ex.example_id: ex.sql for ex in grading_examples}
        missing_ids: list[str] = []
    else:
        raw_records = load_predictions_jsonl(args.predictions)
        validation = validate_predictions(raw_records, generation_examples)
        if validation.duplicate_ids:
            print(f"WARNING: {len(validation.duplicate_ids)} duplicate prediction ids ignored")
        if validation.unknown_ids:
            print(f"WARNING: {len(validation.unknown_ids)} predictions reference unknown example_ids")
        if validation.db_id_mismatches:
            print(f"WARNING: {len(validation.db_id_mismatches)} predictions have db_id mismatches")
        if validation.non_string_sql:
            print(f"WARNING: {len(validation.non_string_sql)} predictions have non-string predicted_sql")
        predictions_by_id = {
            eid: rec.predicted_sql
            for eid, rec in validation.valid_records.items()
            if eid in selected_ids
        }
        missing_ids = [eid for eid in selected_ids if eid not in predictions_by_id]
        if missing_ids:
            print(f"NOTE: {len(missing_ids)} expected examples have no prediction (reported explicitly)")

    # --- LocalSQL diagnostics ---
    diagnostics = []
    if args.mode in ("diagnostics", "both"):
        for ex in grading_examples:
            sql = predictions_by_id.get(ex.example_id)
            if sql is None:
                continue
            db_path = paths["databases"] / ex.db_id / f"{ex.db_id}.sqlite"
            diagnostics.append(diagnose_prediction(ex.example_id, sql, db_path, ex.dialect, timeout_seconds))

    # --- official evaluator ---
    ex_result = None
    f1_result = None
    if args.mode in ("official", "both"):
        try:
            evaluator_dir = paths["third_party_evaluator"]
            if not (evaluator_dir / "evaluation_ex.py").exists():
                fetch_official_evaluator(evaluator_dir, config["source"]["official_evaluator"]["revision"])

            run_dir = paths["reports"] / f"{run_id}_official_inputs"
            run_dir.mkdir(parents=True, exist_ok=True)

            ids_in_order = [ex.example_id for ex in grading_examples]
            db_ids_in_order = [ex.db_id for ex in grading_examples]
            difficulties_in_order = [ex.difficulty for ex in grading_examples]

            pred_json_path = run_dir / "predictions_official_format.json"
            official_missing = build_official_prediction_file(
                predictions_by_id, ids_in_order, db_ids_in_order, pred_json_path
            )

            diff_jsonl_path = run_dir / "diff.jsonl"
            build_diff_jsonl(difficulties_in_order, diff_jsonl_path)

            if args.limit:
                full_gold_pairs = load_gold_sql(paths["raw"] / "mini_dev_sqlite_gold.sql")
                sliced_gold_path = run_dir / "gold_sliced.sql"
                sliced_gold_path.write_text(
                    "\n".join(f"{sql}\t{db_id}" for sql, db_id in full_gold_pairs[: args.limit]) + "\n",
                    encoding="utf-8",
                )
                gold_sql_path = sliced_gold_path
            else:
                gold_sql_path = paths["raw"] / "mini_dev_sqlite_gold.sql"

            ex_result = run_official_ex(
                evaluator_dir, pred_json_path, gold_sql_path, paths["databases"], diff_jsonl_path,
                num_cpus=num_cpus, timeout_seconds=timeout_seconds,
            )
            f1_result = run_official_f1(
                evaluator_dir, pred_json_path, gold_sql_path, paths["databases"], diff_jsonl_path,
                num_cpus=num_cpus, timeout_seconds=timeout_seconds,
            )
        except Exception as e:
            print(f"BLOCKER: official evaluator could not run: {type(e).__name__}: {e}")
            print("Official EX/Soft-F1 marked UNAVAILABLE in the report. Not replaced with a custom metric.")

    report = build_report(
        benchmark_id=config["benchmark_id"],
        benchmark_revision={
            "questions": config["source"]["questions"],
            "official_evaluator_revision": config["source"]["official_evaluator"]["revision"],
        },
        run_id=run_id,
        model_id=args.model_id,
        prediction_file=args.predictions,
        grading_examples=grading_examples,
        ex_result=ex_result,
        f1_result=f1_result,
        diagnostics=diagnostics,
        missing_prediction_ids=missing_ids,
        oracle_sanity=args.oracle_sanity,
    )

    report_path = paths["reports"] / f"{run_id}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    diag_path = paths["reports"] / f"{run_id}.diagnostics.jsonl"
    with diag_path.open("w", encoding="utf-8") as f:
        for d in diagnostics:
            f.write(json.dumps(d.__dict__) + "\n")

    official_raw_path = None
    if ex_result is not None:
        # Per-example official EX outcome (sql_idx -> res), for failure
        # analysis. Not an official artifact -- a LocalSQL-side record of
        # what the unmodified official evaluator returned per example.
        official_raw_path = paths["reports"] / f"{run_id}.official_ex_raw.jsonl"
        with official_raw_path.open("w", encoding="utf-8") as f:
            for r in ex_result["raw_results"]:
                ex = grading_examples[r["sql_idx"]]
                f.write(json.dumps({
                    "sql_idx": r["sql_idx"],
                    "example_id": ex.example_id,
                    "db_id": ex.db_id,
                    "difficulty": ex.difficulty,
                    "official_ex_correct": bool(r["res"]),
                }) + "\n")

    print(f"\nReport written: {report_path}")
    print(f"Per-example diagnostics: {diag_path}")
    if official_raw_path:
        print(f"Per-example official EX outcomes: {official_raw_path}")
    if ex_result:
        print(f"Official EX: {ex_result['overall']}")
    if f1_result:
        print(f"Official Soft-F1: {f1_result['overall']}")


if __name__ == "__main__":
    main()
