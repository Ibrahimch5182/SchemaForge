"""Inspect the real birdsql/bird23-train-filtered dataset.

Rerunnable. Produces human-readable console output and a machine-readable
JSON report under data/reports/. Does not hard-code the expected record
count or any other stat -- everything is discovered at runtime.

Usage:
    uv run python scripts/inspect_bird.py
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from localsql.data.bird_loader import (  # noqa: E402
    DATASET_REPO_ID,
    download_column_meaning,
    download_raw_jsonl,
    get_dataset_revision,
    load_column_meaning,
    load_raw_examples,
)


def length_stats(values: list[int]) -> dict:
    return {
        "min": min(values),
        "max": max(values),
        "mean": round(statistics.mean(values), 2),
        "median": statistics.median(values),
        "stdev": round(statistics.stdev(values), 2) if len(values) > 1 else 0.0,
    }


def main() -> None:
    print(f"Loading dataset: {DATASET_REPO_ID}")
    revision = get_dataset_revision()
    jsonl_path = download_raw_jsonl()
    meaning_path = download_column_meaning()
    examples = load_raw_examples(jsonl_path)
    column_meaning = load_column_meaning(meaning_path)

    total = len(examples)
    db_ids = [e.db_id for e in examples]
    unique_db_ids = sorted(set(db_ids))
    questions = [e.question for e in examples]
    sqls = [e.sql for e in examples]
    evidences = [e.evidence for e in examples]

    empty_question = sum(1 for q in questions if not q or not q.strip())
    empty_sql = sum(1 for s in sqls if not s or not s.strip())
    empty_db_id = sum(1 for d in db_ids if not d or not d.strip())
    nonempty_evidence = sum(1 for e in evidences if e and e.strip())
    empty_evidence = total - nonempty_evidence

    dup_question_count = total - len(set(questions))
    dup_sql_count = total - len(set(sqls))

    question_lengths = [len(q) for q in questions]
    sql_lengths = [len(s) for s in sqls]

    examples_per_db = Counter(db_ids)

    report = {
        "dataset_repo_id": DATASET_REPO_ID,
        "dataset_revision": revision,
        "source_file": str(jsonl_path),
        "column_meaning_file": str(meaning_path),
        "hf_split_name": "train",
        "columns": list(examples[0].model_dump(by_alias=True).keys()) if examples else [],
        "total_examples": total,
        "unique_db_id_count": len(unique_db_ids),
        "unique_db_ids": unique_db_ids,
        "examples_per_db_min": min(examples_per_db.values()),
        "examples_per_db_max": max(examples_per_db.values()),
        "null_or_empty_counts": {
            "db_id": empty_db_id,
            "question": empty_question,
            "sql": empty_sql,
            "evidence": empty_evidence,
        },
        "evidence_availability": {
            "with_evidence": nonempty_evidence,
            "without_evidence": empty_evidence,
            "fraction_with_evidence": round(nonempty_evidence / total, 4) if total else 0.0,
        },
        "duplicates": {
            "duplicate_question_count": dup_question_count,
            "duplicate_sql_count": dup_sql_count,
        },
        "question_length_chars": length_stats(question_lengths),
        "sql_length_chars": length_stats(sql_lengths),
        "column_meaning_entry_count": len(column_meaning),
    }

    print("\n=== birdsql/bird23-train-filtered inspection ===")
    print(f"revision:            {revision}")
    print(f"total examples:      {total}")
    print(f"unique db_id count:  {len(unique_db_ids)}")
    print(f"columns:             {report['columns']}")
    print(f"empty question/sql:  {empty_question} / {empty_sql}")
    print(f"evidence available:  {nonempty_evidence} ({report['evidence_availability']['fraction_with_evidence']:.1%})")
    print(f"duplicate questions: {dup_question_count}")
    print(f"duplicate sql:       {dup_sql_count}")
    print(f"question length:     {report['question_length_chars']}")
    print(f"sql length:          {report['sql_length_chars']}")
    print(f"column_meaning rows: {len(column_meaning)}")

    print("\n--- sample examples from 3 different databases ---")
    seen_dbs: set[str] = set()
    for ex in examples:
        if ex.db_id in seen_dbs:
            continue
        seen_dbs.add(ex.db_id)
        print(f"\n[{ex.db_id}] Q: {ex.question}")
        print(f"  evidence: {ex.evidence!r}")
        print(f"  SQL: {ex.sql}")
        if len(seen_dbs) >= 3:
            break

    reports_dir = REPO_ROOT / "data" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / "bird_inspection_report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nJSON report written to: {out_path}")


if __name__ == "__main__":
    main()
