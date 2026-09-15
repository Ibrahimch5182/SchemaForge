"""Phase 5A candidate dataset: adaptive per-database full-schema compaction.

CANDIDATE, not canonical: writes SEPARATE artifacts under
`data/processed_phase5_candidate/`, never touching Phase 1's original
`data/processed/{train,validation}.jsonl`. No training happens here.

Policy (see `localsql.schema_context.db_policy`):
  - TRAIN: databases in `TRAIN_COMPACT_DB_IDS` (REAL Phase 4 Kaggle
    tokenizer data -- see citation in that module) get the compact
    (description-free) full-schema serializer for ALL their examples;
    every other database is left byte-identical to Phase 1's original.
  - VALIDATION: Phase 4 never profiled validation with the real tokenizer,
    so a database is compacted iff at least one of its examples' ESTIMATED
    canonical full-SFT length exceeds 4096 tokens (local character-count
    estimator, clearly labeled, canonical token length only -- never
    validation gold correctness).

No question-conditioning, no gold SQL, no per-example table/column
dropping -- compact databases keep every table, column, data type,
primary key, and foreign-key relationship; non-compact databases are
completely unchanged.

Usage:
    uv run python scripts/build_phase5_candidate.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from localsql.data.bird_loader import COLUMN_MEANING_FILE, DATASET_REPO_ID, download_column_meaning, load_column_meaning  # noqa: E402
from localsql.data.models import DatabaseSchema  # noqa: E402
from localsql.data.prompt_builder import build_prompt  # noqa: E402
from localsql.data.schema_loader import fetch_official_train_tables_json, parse_database_schemas  # noqa: E402
from localsql.schema_context.compact_serializer import serialize_schema_compact  # noqa: E402
from localsql.schema_context.db_policy import (  # noqa: E402
    REAL_PHASE4_SOURCE_CITATION,
    TRAIN_COMPACT_DB_IDS,
    summarize_policy,
    train_db_policy,
    validation_db_policy,
)
from localsql.schema_context.token_estimate import CALIBRATED_CHARS_PER_TOKEN, estimate_tokens  # noqa: E402

OUT_DIR = REPO_ROOT / "data" / "processed_phase5_candidate"
ORIGINAL_DIR = REPO_ROOT / "data" / "processed"


def load_schemas() -> dict[str, DatabaseSchema]:
    schema_path = REPO_ROOT / "data" / "raw" / "train_tables.json"
    fetch_official_train_tables_json(schema_path)
    meaning_path = download_column_meaning(DATASET_REPO_ID, COLUMN_MEANING_FILE)
    column_meaning = load_column_meaning(meaning_path)
    return parse_database_schemas(schema_path, column_meaning, dialect="sqlite")


def load_examples(split: str) -> list[dict]:
    path = ORIGINAL_DIR / f"{split}.jsonl"
    if not path.exists():
        print(f"BLOCKER: {path} not found. Run `uv run python scripts/prepare_bird.py` first (Phase 1).")
        sys.exit(1)
    examples = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def assert_structural_preservation(schema: DatabaseSchema, compact_text: str) -> None:
    """Every table, column, PK annotation, and FK annotation from the
    source schema must appear -- on the right column's line, WITHIN that
    table's own block -- in the compact serialization. Column checks are
    scoped per-table because column names repeat across tables in large
    schemas (e.g. `works_cycles` has 65 tables); searching the whole text
    could match the wrong table's line.
    """
    table_blocks = {block.split("(", 1)[0]: block for block in compact_text.split("\n\n")}
    for table in schema.tables:
        assert table.name in table_blocks, f"table {table.name} missing from compact schema"
        block_lines = table_blocks[table.name].splitlines()
        column_lines = {}
        for col in table.columns:
            matches = [l for l in block_lines if l.strip().startswith(f"{col.name} ") or l.strip() == col.name]
            assert matches, f"column {table.name}.{col.name} missing from compact schema"
            column_lines[col.name] = matches[0]

        for col in table.columns:
            line = column_lines[col.name]
            if col.is_primary_key:
                assert " PK" in line, f"PK annotation missing for {table.name}.{col.name}: {line!r}"
            if col.foreign_key:
                fk_text = f"FK->{col.foreign_key.table}.{col.foreign_key.column}"
                assert fk_text in line, f"FK annotation missing for {table.name}.{col.name}: {line!r}"


def build_candidate_split(
    split: str,
    examples: list[dict],
    schemas: dict[str, DatabaseSchema],
    compact_db_ids: set[str],
    policy_basis: str,
) -> list[dict]:
    compact_cache: dict[str, str] = {}
    candidate = []
    for e in examples:
        db_id = e["db_id"]
        if db_id in compact_db_ids:
            if db_id not in compact_cache:
                compact_text = serialize_schema_compact(schemas[db_id])
                assert_structural_preservation(schemas[db_id], compact_text)
                compact_cache[db_id] = compact_text
            schema_text = compact_cache[db_id]
            prompt = build_prompt(schema_text, e["dialect"], e["question"], e.get("business_context"))
            record = {
                "example_id": e["example_id"],
                "db_id": db_id,
                "split": e["split"],
                "source": e["source"],
                "question": e["question"],
                "business_context": e.get("business_context"),
                "completion": e["completion"],
                "dialect": e["dialect"],
                "representation": "compact_full_schema",
                "policy_basis": policy_basis,
                "serialized_schema": schema_text,
                "prompt": prompt,
            }
        else:
            # Byte-identical to the original Phase 1 prepared example.
            record = {
                "example_id": e["example_id"],
                "db_id": db_id,
                "split": e["split"],
                "source": e["source"],
                "question": e["question"],
                "business_context": e.get("business_context"),
                "completion": e["completion"],
                "dialect": e["dialect"],
                "representation": "canonical_unchanged",
                "policy_basis": policy_basis,
                "serialized_schema": e["schema_serialized"],
                "prompt": e["prompt"],
            }
        candidate.append(record)
    return candidate


def assert_candidate_integrity(
    split: str, candidate: list[dict], original: list[dict], expected_count: int
) -> None:
    assert len(candidate) == len(original) == expected_count, (
        f"{split}: count changed -- candidate={len(candidate)} original={len(original)} expected={expected_count}"
    )
    candidate_ids = [c["example_id"] for c in candidate]
    assert len(candidate_ids) == len(set(candidate_ids)), f"{split}: duplicate example_id(s) in candidate"
    assert set(candidate_ids) == {o["example_id"] for o in original}, f"{split}: example_id set changed"
    assert {c["db_id"] for c in candidate} == {o["db_id"] for o in original}, f"{split}: db_id set changed"
    for c, o in zip(candidate, original):
        assert c["completion"] == o["completion"], f"{split}: completion mutated for {c['example_id']}"
        if c["representation"] == "canonical_unchanged":
            assert c["prompt"] == o["prompt"], f"{split}: unchanged-policy prompt was modified for {c['example_id']}"
            assert c["serialized_schema"] == o["schema_serialized"]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading schemas (reusing Phase 1's cached sources) ...")
    schemas = load_schemas()

    train_examples = load_examples("train")
    validation_examples = load_examples("validation")

    train_db_counts: dict[str, int] = defaultdict(int)
    for e in train_examples:
        train_db_counts[e["db_id"]] += 1

    train_decisions = train_db_policy(dict(train_db_counts))
    train_summary = summarize_policy(train_decisions)
    print(
        f"TRAIN policy (real Phase 4 data): {train_summary['compact_db_count']} DBs / "
        f"{train_summary['compact_example_count']} examples compact; "
        f"{train_summary['unchanged_db_count']} DBs / {train_summary['unchanged_example_count']} unchanged."
    )
    assert set(train_decisions) == set(train_db_counts), "train db set mismatch vs schema/example data"
    assert train_summary["compact_db_ids"] == sorted(TRAIN_COMPACT_DB_IDS), (
        "TRAIN_COMPACT_DB_IDS does not match the databases actually present -- policy inconsistent"
    )

    # --- validation: local-estimate-based, canonical length only ---
    per_db_lengths: dict[str, list[float]] = defaultdict(list)
    for e in validation_examples:
        per_db_lengths[e["db_id"]].append(estimate_tokens(e["prompt"] + e["completion"]))
    validation_decisions = validation_db_policy(dict(per_db_lengths))
    validation_summary = summarize_policy(validation_decisions)
    print(
        f"VALIDATION policy (local estimate): {validation_summary['compact_db_count']} DBs / "
        f"{validation_summary['compact_example_count']} examples compact; "
        f"{validation_summary['unchanged_db_count']} DBs / {validation_summary['unchanged_example_count']} unchanged."
    )

    train_compact_ids = {d for d, dec in train_decisions.items() if dec.compact}
    validation_compact_ids = {d for d, dec in validation_decisions.items() if dec.compact}

    train_candidate = build_candidate_split("train", train_examples, schemas, train_compact_ids, "real_phase4_profile")
    validation_candidate = build_candidate_split(
        "validation", validation_examples, schemas, validation_compact_ids, "local_estimate"
    )

    assert_candidate_integrity("train", train_candidate, train_examples, 6067)
    assert_candidate_integrity("validation", validation_candidate, validation_examples, 534)

    # --- original-file-unchanged check ---
    def file_hash(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    original_train_hash_before = file_hash(ORIGINAL_DIR / "train.jsonl")
    original_validation_hash_before = file_hash(ORIGINAL_DIR / "validation.jsonl")

    train_out = OUT_DIR / "train.jsonl"
    validation_out = OUT_DIR / "validation.jsonl"
    with train_out.open("w", encoding="utf-8") as f:
        for r in train_candidate:
            f.write(json.dumps(r) + "\n")
    with validation_out.open("w", encoding="utf-8") as f:
        for r in validation_candidate:
            f.write(json.dumps(r) + "\n")

    original_train_hash_after = file_hash(ORIGINAL_DIR / "train.jsonl")
    original_validation_hash_after = file_hash(ORIGINAL_DIR / "validation.jsonl")
    assert original_train_hash_before == original_train_hash_after, "original train.jsonl was modified!"
    assert original_validation_hash_before == original_validation_hash_after, "original validation.jsonl was modified!"

    policy = {
        "candidate_policy": "adaptive_per_database_full_schema_v1",
        "no_question_conditioning": True,
        "no_gold_sql_used": True,
        "selector_prototype_used": False,
        "notes": (
            "Question-conditioned schema budgeting (localsql.schema_context.relevance) "
            "was evaluated in Phase 5A but is NOT used for this candidate -- kept in the "
            "repo as documented, non-selected research tooling. See docs/CONTEXT_BUDGET.md."
        ),
        "train": {
            **train_summary,
            "policy_basis": "real_phase4_profile",
            "source_citation": REAL_PHASE4_SOURCE_CITATION,
        },
        "validation": {
            **validation_summary,
            "policy_basis": "local_estimate",
            "estimator_note": (
                "Phase 4 never profiled validation with the real Qwen tokenizer. This decision "
                "uses localsql.schema_context.token_estimate (calibrated character-count proxy), "
                "canonical (representation A) full-SFT length only -- never gold correctness."
            ),
        },
        "candidate_max_seq_length": 4096,
        "candidate_max_seq_length_status": "PROVISIONAL -- not final until real-tokenizer confirmation (see docs/CONTEXT_BUDGET.md)",
        "original_files": {
            "train_sha256": original_train_hash_after,
            "validation_sha256": original_validation_hash_after,
        },
        "candidate_files": {
            "train_sha256": file_hash(train_out),
            "validation_sha256": file_hash(validation_out),
        },
    }
    (OUT_DIR / "policy.json").write_text(json.dumps(policy, indent=2), encoding="utf-8")

    print(f"\nWrote {train_out}")
    print(f"Wrote {validation_out}")
    print(f"Wrote {OUT_DIR / 'policy.json'}")
    print("All integrity assertions passed. Original data/processed files verified byte-unchanged.")


if __name__ == "__main__":
    main()
