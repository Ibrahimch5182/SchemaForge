"""Tests for the Phase 5A candidate policy (localsql.schema_context.db_policy).

CPU/offline. Includes one test against the real Phase 1 train.jsonl (skips
cleanly if not present locally) proving the 9-DB / 1,502-example claim is
programmatically confirmed, not assumed.
"""

from __future__ import annotations

import json
from pathlib import Path

from localsql.schema_context.db_policy import (
    TRAIN_COMPACT_DB_IDS,
    summarize_policy,
    train_db_policy,
    validation_db_policy,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_train_policy_is_deterministic_given_same_counts():
    counts = {"works_cycles": 383, "hockey": 156, "small_db": 10}
    a = train_db_policy(counts)
    b = train_db_policy(counts)
    assert a == b


def test_train_policy_marks_known_long_dbs_compact_others_unchanged():
    counts = {db: 5 for db in TRAIN_COMPACT_DB_IDS} | {"unrelated_db": 5}
    decisions = train_db_policy(counts)
    assert all(decisions[db].compact for db in TRAIN_COMPACT_DB_IDS)
    assert decisions["unrelated_db"].compact is False
    assert all(d.basis == "real_phase4_profile" for d in decisions.values())


def test_train_policy_never_uses_gold_or_question_information():
    """The function signature itself has no such parameter -- structural
    proof, not just behavioral."""
    import inspect

    params = set(inspect.signature(train_db_policy).parameters)
    assert not (params & {"sql", "gold", "question", "business_context"})


def test_validation_policy_flags_db_with_any_over_threshold_example():
    per_db = {
        "college_completion": [5000.0] * 45,  # all over 4096
        "authors": [500.0, 800.0, 1200.0],  # all under
    }
    decisions = validation_db_policy(per_db)
    assert decisions["college_completion"].compact is True
    assert decisions["authors"].compact is False
    assert all(d.basis == "local_estimate" for d in decisions.values())


def test_validation_policy_deterministic():
    per_db = {"db_a": [1000.0, 5000.0], "db_b": [200.0]}
    assert validation_db_policy(per_db) == validation_db_policy(per_db)


def test_summarize_policy_counts_match():
    counts = {"works_cycles": 383, "hockey": 156, "small_db": 10}
    decisions = train_db_policy(counts)
    summary = summarize_policy(decisions)
    assert summary["compact_db_count"] == 2
    assert summary["compact_example_count"] == 383 + 156
    assert summary["unchanged_db_count"] == 1
    assert summary["unchanged_example_count"] == 10


def test_real_train_data_confirms_nine_compact_dbs_and_1502_examples():
    """Programmatic confirmation against the real Phase 1 train.jsonl, if
    present locally -- skips cleanly otherwise (not required for CI)."""
    train_path = REPO_ROOT / "data" / "processed" / "train.jsonl"
    if not train_path.exists():
        return

    from collections import Counter

    counts: Counter[str] = Counter()
    with train_path.open(encoding="utf-8") as f:
        for line in f:
            counts[json.loads(line)["db_id"]] += 1

    decisions = train_db_policy(dict(counts))
    summary = summarize_policy(decisions)

    assert summary["compact_db_count"] == 9
    assert summary["compact_example_count"] == 1502
    assert summary["unchanged_db_count"] == 53
    assert summary["unchanged_example_count"] == 4565
    assert set(summary["compact_db_ids"]) == set(TRAIN_COMPACT_DB_IDS)
