"""Assembly of the final LocalSQL evaluation report from official + diagnostic results."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from localsql.benchmark.diagnostics import PredictionDiagnostic, summarize_diagnostics
from localsql.benchmark.models import GradingExample


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _results_by_group(
    raw_results: list[dict], grading_examples: list[GradingExample], group_key: str
) -> dict:
    """`raw_results` is the official evaluator's [{sql_idx, res}, ...] list.
    `group_key` is "db_id" or "difficulty".
    """
    totals: dict[str, list[int]] = defaultdict(list)
    for r in raw_results:
        example = grading_examples[r["sql_idx"]]
        key = getattr(example, group_key) or "unknown"
        totals[key].append(r["res"])
    return {
        key: {"count": len(scores), "accuracy": round(sum(scores) / len(scores), 4)}
        for key, scores in sorted(totals.items())
    }


def build_report(
    *,
    benchmark_id: str,
    benchmark_revision: dict,
    run_id: str,
    model_id: Optional[str],
    prediction_file: Optional[Path],
    grading_examples: list[GradingExample],
    ex_result: Optional[dict],
    f1_result: Optional[dict],
    diagnostics: list[PredictionDiagnostic],
    missing_prediction_ids: list[str],
    oracle_sanity: bool = False,
) -> dict:
    """Combine official metrics + LocalSQL diagnostics into one JSON report.

    `ex_result` / `f1_result` may be None if the official evaluator could
    not run (a documented blocker), in which case those sections are marked
    unavailable rather than silently omitted.
    """
    report: dict = {
        "benchmark_id": benchmark_id,
        "benchmark_variant": "original_500_select_only_sqlite",
        "benchmark_revision": benchmark_revision,
        "run_id": run_id,
        "model_id": model_id,
        "oracle_sanity_run": oracle_sanity,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "total_examples": len(grading_examples),
        "missing_prediction_count": len(missing_prediction_ids),
        "missing_prediction_ids": missing_prediction_ids,
        "prediction_file_sha256": (
            file_sha256(prediction_file) if prediction_file and prediction_file.exists() else None
        ),
    }

    if ex_result is not None:
        report["official_execution_accuracy"] = {
            "overall": ex_result["overall"],
            "by_difficulty": {
                "simple": ex_result["simple"],
                "moderate": ex_result["moderate"],
                "challenging": ex_result["challenging"],
                "counts": ex_result["counts"],
            },
            "by_database": _results_by_group(ex_result["raw_results"], grading_examples, "db_id"),
        }
    else:
        report["official_execution_accuracy"] = {"status": "UNAVAILABLE", "reason": "see issues log"}

    if f1_result is not None:
        report["official_soft_f1"] = {
            "overall": f1_result["overall"],
            "by_difficulty": {
                "simple": f1_result["simple"],
                "moderate": f1_result["moderate"],
                "challenging": f1_result["challenging"],
                "counts": f1_result["counts"],
            },
            "by_database": _results_by_group(f1_result["raw_results"], grading_examples, "db_id"),
        }
    else:
        report["official_soft_f1"] = {"status": "UNAVAILABLE", "reason": "see issues log"}

    report["r_ves"] = {"status": "DEFERRED", "reason": "hardware/timing-sensitive; fixed at a later phase"}

    report["localsql_diagnostics"] = {
        "note": "LocalSQL diagnostics only -- NOT official BIRD metrics. "
        "parseable != executable != correct.",
        **summarize_diagnostics(diagnostics),
    }

    return report
