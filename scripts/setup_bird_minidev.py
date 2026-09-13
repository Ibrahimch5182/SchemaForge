"""Set up the LocalSQL BIRD Mini-Dev benchmark workspace (Phase 2).

Fetches the official original-500 SELECT-only SQLite Mini-Dev resources,
verifies expected counts, builds the gold-free generation manifest and the
isolated grading reference, and vendors the pinned official evaluator.
Idempotent: already-present, correctly-sized resources are reused.

Usage:
    uv run python scripts/setup_bird_minidev.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from localsql.benchmark.manifest_builder import build_manifests  # noqa: E402
from localsql.benchmark.minidev_loader import (  # noqa: E402
    QUESTIONS_REPO_ID,
    analyze_source_consistency,
    download_questions_json,
    extract_archive_resources,
    get_questions_revision,
    load_gold_sql,
    load_questions,
)
from localsql.benchmark.official_adapter import fetch_official_evaluator  # noqa: E402


def load_config() -> dict:
    return yaml.safe_load((REPO_ROOT / "configs" / "benchmark.yaml").read_text(encoding="utf-8"))


def main() -> None:
    config = load_config()
    paths = {k: REPO_ROOT / v for k, v in config["paths"].items()}
    for d in (paths["raw"], paths["databases"], paths["generation"], paths["grading"], paths["reports"]):
        d.mkdir(parents=True, exist_ok=True)

    print(f"Loading Mini-Dev questions from {QUESTIONS_REPO_ID} (mini_dev_sqlite split) ...")
    questions_revision = get_questions_revision()
    questions_path = download_questions_json()
    questions = load_questions(questions_path)
    print(f"  {len(questions)} questions loaded (revision {questions_revision})")

    db_ids = sorted({q["db_id"] for q in questions})
    print(f"  {len(db_ids)} unique db_ids")

    print("Extracting official schema/gold/database resources from minidev.zip ...")
    summary = extract_archive_resources(paths["raw"], paths["databases"], db_ids=db_ids)
    print(f"  written: {len(summary['written'])}  reused: {len(summary['reused'])}")

    # --- verification ---
    problems = []
    if len(questions) != config["expected_examples"]:
        problems.append(f"expected {config['expected_examples']} examples, got {len(questions)}")
    if len(db_ids) != config["expected_databases"]:
        problems.append(f"expected {config['expected_databases']} databases, got {len(db_ids)}")
    for db_id in db_ids:
        db_file = paths["databases"] / db_id / f"{db_id}.sqlite"
        if not db_file.exists():
            problems.append(f"missing database file for db_id={db_id!r}")
    schema_path = paths["raw"] / "dev_tables.json"
    if not schema_path.exists():
        problems.append("missing dev_tables.json")
    gold_path = paths["raw"] / "mini_dev_sqlite_gold.sql"
    if not gold_path.exists():
        problems.append("missing mini_dev_sqlite_gold.sql")

    if problems:
        print("\nBLOCKERS:")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)

    print("Checking HF-vs-archive source consistency (diagnostic only) ...")
    cross_check_path = paths["raw"] / "mini_dev_sqlite.json"
    archive_questions = json.loads(cross_check_path.read_text(encoding="utf-8"))
    gold_pairs = load_gold_sql(gold_path)
    source_consistency = analyze_source_consistency(questions, archive_questions, gold_pairs)
    if source_consistency["hf_vs_archive_sql_mismatch_count"] or source_consistency["archive_internal_mismatch_count"]:
        print(
            f"  NOTE: {source_consistency['hf_vs_archive_sql_mismatch_count']} rows where HF's SQL "
            f"differs from the archive's own question file (indices: "
            f"{source_consistency['hf_vs_archive_sql_mismatch_indices']}). This is diagnostic only -- "
            "LocalSQL's grading reference is sourced from the archive's mini_dev_sqlite_gold.sql "
            "(canonical), never from HF's SQL field. See docs/EVALUATION.md."
        )
    else:
        print("  HF and archive sources agree on every row.")

    # --- pre-write verification: archive gold is the canonical grading source ---
    verify_problems = []
    if len(gold_pairs) != config["expected_examples"]:
        verify_problems.append(f"archive gold row count {len(gold_pairs)} != expected {config['expected_examples']}")
    archive_db_id_count = sum(1 for _, db_id in gold_pairs if db_id)
    if archive_db_id_count != config["expected_examples"]:
        verify_problems.append(f"archive gold db_id row count {archive_db_id_count} != expected {config['expected_examples']}")
    if verify_problems:
        print("\nBLOCKERS:")
        for p in verify_problems:
            print(f"  - {p}")
        sys.exit(1)

    print("Building gold-free generation manifest + isolated grading reference "
          "(grading SQL sourced from archive mini_dev_sqlite_gold.sql) ...")
    generation_examples, grading_examples = build_manifests(
        questions, schema_path, gold_path, paths["databases"], dialect=config["dialect"]
    )

    gen_ids = {ex.example_id for ex in generation_examples}
    grade_ids = {ex.example_id for ex in grading_examples}
    if len(gen_ids) != config["expected_examples"] or len(grade_ids) != config["expected_examples"]:
        print(
            f"\nBLOCKER: expected {config['expected_examples']} unique IDs, got "
            f"{len(gen_ids)} generation / {len(grade_ids)} grading"
        )
        sys.exit(1)

    gen_path = paths["generation"] / "manifest.jsonl"
    with gen_path.open("w", encoding="utf-8") as f:
        for ex in generation_examples:
            f.write(ex.model_dump_json() + "\n")

    grading_path = paths["grading"] / "reference.jsonl"
    with grading_path.open("w", encoding="utf-8") as f:
        for ex in grading_examples:
            f.write(ex.model_dump_json() + "\n")

    print(f"  wrote {len(generation_examples)} generation examples -> {gen_path}")
    print(f"  wrote {len(grading_examples)} grading examples -> {grading_path}")

    # Gold-leakage self-check (also covered by an automated test).
    gen_field_names = set(type(generation_examples[0]).model_fields)
    leaked = gen_field_names & {"sql", "gold_sql", "completion", "target", "target_sql"}
    if leaked:
        print(f"BLOCKER: generation manifest exposes gold fields: {leaked}")
        sys.exit(1)

    print("Vendoring pinned official evaluator files ...")
    evaluator_manifest = fetch_official_evaluator(
        paths["third_party_evaluator"], config["source"]["official_evaluator"]["revision"]
    )
    print(f"  vendored to {paths['third_party_evaluator']} (revision {evaluator_manifest['revision']})")

    setup_report = {
        "questions_source_repo": QUESTIONS_REPO_ID,
        "questions_revision": questions_revision,
        "expected_examples": config["expected_examples"],
        "actual_examples": len(questions),
        "expected_databases": config["expected_databases"],
        "actual_databases": len(db_ids),
        "db_ids": db_ids,
        "generation_examples": len(generation_examples),
        "grading_examples": len(grading_examples),
        "evaluator": evaluator_manifest,
        "archive_extraction_summary": summary,
        "canonical_gold_source": {
            "source": "BIRD Mini-Dev official archive (minidev.zip)",
            "file": "mini_dev_sqlite_gold.sql",
        },
        "hf_sql_role": "diagnostic_only",
        "source_consistency": source_consistency,
    }
    report_path = paths["reports"] / "setup_report.json"
    report_path.write_text(json.dumps(setup_report, indent=2), encoding="utf-8")
    print(f"\nSetup complete. Report: {report_path}")


if __name__ == "__main__":
    main()
