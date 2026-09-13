"""Thin adapter around the unmodified official BIRD Mini-Dev evaluator.

Vendors `evaluation_ex.py` / `evaluation_f1.py` / `evaluation_utils.py` from
`bird-bench/mini_dev` at a pinned commit (see `configs/benchmark.yaml`), then
imports them directly (rather than shelling out and scraping stdout) so we
can call their own `compute_acc_by_diff` / `compute_f1_by_diff` functions and
get back structured results -- upstream scoring logic is reused exactly as
published, never copied or modified.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import urllib.request
from pathlib import Path
from types import ModuleType

EVALUATOR_REPO_RAW_BASE = "https://raw.githubusercontent.com/bird-bench/mini_dev"
EVALUATOR_FILES = ["evaluation_ex.py", "evaluation_f1.py", "evaluation_utils.py"]

# Sentinel SQL used to fill in for predictions missing from a submission, so
# every positional index required by the official evaluator's pred/gt
# alignment stays present. This is guaranteed to score as incorrect.
MISSING_PREDICTION_SQL = "SELECT 'LOCALSQL_MISSING_PREDICTION' WHERE 1=0"


def fetch_official_evaluator(dest_dir: Path, revision: str, force: bool = False) -> dict:
    """Download the pinned evaluator files, recording their sha256 checksums."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = dest_dir / "SOURCE.json"
    checksums: dict[str, str] = {}

    for filename in EVALUATOR_FILES:
        dst = dest_dir / filename
        if dst.exists() and not force:
            checksums[filename] = hashlib.sha256(dst.read_bytes()).hexdigest()
            continue
        url = f"{EVALUATOR_REPO_RAW_BASE}/{revision}/evaluation/{filename}"
        with urllib.request.urlopen(url, timeout=60) as resp:
            content = resp.read()
        dst.write_bytes(content)
        checksums[filename] = hashlib.sha256(content).hexdigest()

    manifest = {
        "repo": "bird-bench/mini_dev",
        "revision": revision,
        "files": checksums,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _load_vendored_module(dest_dir: Path, name: str) -> ModuleType:
    """Import a vendored evaluator file as a module, with `dest_dir` on sys.path
    (evaluation_ex.py / evaluation_f1.py do `from evaluation_utils import ...`).
    """
    if str(dest_dir) not in sys.path:
        sys.path.insert(0, str(dest_dir))
    spec = importlib.util.spec_from_file_location(name, dest_dir / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def build_official_prediction_file(
    predictions_by_id: dict[str, str],
    grading_ids_in_order: list[str],
    grading_db_ids_in_order: list[str],
    out_path: Path,
) -> list[str]:
    """Build the official `{idx: "SQL\\t----- bird -----\\tdb_id"}` prediction JSON.

    Every position required by the official evaluator's index-aligned
    pred/gt pairing is filled -- a missing prediction gets a sentinel SQL
    that is guaranteed to score as incorrect, rather than shifting alignment
    for every example after it. Returns the list of example_ids that were
    filled in as missing.
    """
    missing: list[str] = []
    out = {}
    for idx, (example_id, db_id) in enumerate(zip(grading_ids_in_order, grading_db_ids_in_order)):
        sql = predictions_by_id.get(example_id)
        if sql is None:
            sql = MISSING_PREDICTION_SQL
            missing.append(example_id)
        out[str(idx)] = f"{sql}\t----- bird -----\t{db_id}"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return missing


def build_diff_jsonl(difficulties_in_order: list[str | None], out_path: Path) -> None:
    """Write the `.jsonl` difficulty file the official evaluator expects.

    Upstream's own `mini_dev_sqlite.json` is a JSON array, not JSONL; this
    reformats it (same content, different container) without altering any
    values.
    """
    with out_path.open("w", encoding="utf-8") as f:
        for difficulty in difficulties_in_order:
            f.write(json.dumps({"difficulty": difficulty}) + "\n")


def _run_metric(
    module_name: str,
    vendored_dir: Path,
    predicted_sql_path: Path,
    gold_sql_path: Path,
    db_root_path: Path,
    diff_jsonl_path: Path,
    num_cpus: int,
    timeout_seconds: float,
    compute_fn_name: str,
) -> dict:
    mod = _load_vendored_module(vendored_dir, module_name)
    mod.exec_result = []  # populated by result_callback during run_sqls_parallel

    pred_queries, _ = mod.package_sqls(str(predicted_sql_path), "", mode="pred")
    gt_queries, db_paths_gt = mod.package_sqls(
        str(gold_sql_path), str(db_root_path).rstrip("/\\") + "/", mode="gt"
    )
    if len(pred_queries) != len(gt_queries):
        raise AssertionError(
            f"prediction count ({len(pred_queries)}) != gold count ({len(gt_queries)}) "
            "after official package_sqls parsing"
        )

    query_pairs = list(zip(pred_queries, gt_queries))
    mod.run_sqls_parallel(
        query_pairs,
        db_places=db_paths_gt,
        num_cpus=num_cpus,
        meta_time_out=timeout_seconds,
        sql_dialect="SQLite",
    )
    exec_result = mod.sort_results(mod.exec_result)

    compute_fn = getattr(mod, compute_fn_name)
    simple, moderate, challenging, overall, counts = compute_fn(exec_result, str(diff_jsonl_path))
    return {
        "overall": round(overall, 4),
        "simple": round(simple, 4),
        "moderate": round(moderate, 4),
        "challenging": round(challenging, 4),
        "counts": {
            "simple": counts[0],
            "moderate": counts[1],
            "challenging": counts[2],
            "total": counts[3],
        },
        "raw_results": [{"sql_idx": r["sql_idx"], "res": r["res"]} for r in exec_result],
    }


def run_official_ex(
    vendored_dir: Path,
    predicted_sql_path: Path,
    gold_sql_path: Path,
    db_root_path: Path,
    diff_jsonl_path: Path,
    num_cpus: int = 1,
    timeout_seconds: float = 30.0,
) -> dict:
    """Run the unmodified official Execution Accuracy evaluator."""
    return _run_metric(
        "evaluation_ex",
        vendored_dir,
        predicted_sql_path,
        gold_sql_path,
        db_root_path,
        diff_jsonl_path,
        num_cpus,
        timeout_seconds,
        "compute_acc_by_diff",
    )


def run_official_f1(
    vendored_dir: Path,
    predicted_sql_path: Path,
    gold_sql_path: Path,
    db_root_path: Path,
    diff_jsonl_path: Path,
    num_cpus: int = 1,
    timeout_seconds: float = 30.0,
) -> dict:
    """Run the unmodified official Soft-F1 evaluator."""
    return _run_metric(
        "evaluation_f1",
        vendored_dir,
        predicted_sql_path,
        gold_sql_path,
        db_root_path,
        diff_jsonl_path,
        num_cpus,
        timeout_seconds,
        "compute_f1_by_diff",
    )
