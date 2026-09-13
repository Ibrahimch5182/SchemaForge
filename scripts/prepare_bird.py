"""Full Phase 1 data preparation pipeline for LocalSQL.

Loads the real birdsql/bird23-train-filtered rows, attaches official BIRD
schema metadata, builds canonical prompt/completion examples, splits by
db_id, applies deterministic evidence dropout, validates every row, and
writes reproducible JSONL + report artifacts.

Usage:
    uv run python scripts/prepare_bird.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from localsql.data.bird_loader import (  # noqa: E402
    download_column_meaning,
    download_raw_jsonl,
    get_dataset_revision,
    load_column_meaning,
    load_raw_examples,
)
from localsql.data.models import PreparedTextToSQLExample, SourceMetadata  # noqa: E402
from localsql.data.prompt_builder import build_completion, build_prompt  # noqa: E402
from localsql.data.schema_loader import (  # noqa: E402
    fetch_official_train_tables_json,
    parse_database_schemas,
)
from localsql.data.schema_serializer import serialize_schema  # noqa: E402
from localsql.data.splitter import should_keep_business_context, split_database_ids  # noqa: E402
from localsql.data.validator import (  # noqa: E402
    check_split_leakage,
    validate_prepared_example,
    validate_raw_example,
)


def load_config() -> dict:
    config_path = REPO_ROOT / "configs" / "data.yaml"
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def length_stats(values: list[int]) -> dict:
    import statistics

    if not values:
        return {"min": 0, "max": 0, "mean": 0.0, "median": 0}
    return {
        "min": min(values),
        "max": max(values),
        "mean": round(statistics.mean(values), 2),
        "median": statistics.median(values),
    }


def main() -> None:
    config = load_config()
    seed = config["seed"]
    dialect = config["dialect"]
    train_fraction = config["train_db_fraction"]
    keep_probability = config["business_context_keep_probability"]
    repo_id = config["source"]["dataset_repo_id"]

    paths = config["paths"]
    raw_dir = REPO_ROOT / paths["raw_dir"]
    processed_dir = REPO_ROOT / paths["processed_dir"]
    reports_dir = REPO_ROOT / paths["reports_dir"]
    for d in (raw_dir, processed_dir, reports_dir):
        d.mkdir(parents=True, exist_ok=True)

    print(f"Loading raw BIRD rows from {repo_id} ...")
    revision = get_dataset_revision(repo_id)
    jsonl_path = download_raw_jsonl(repo_id, config["source"]["dataset_file"])
    meaning_path = download_column_meaning(repo_id, config["source"]["column_meaning_file"])
    raw_examples = load_raw_examples(jsonl_path)
    column_meaning = load_column_meaning(meaning_path)
    print(f"  {len(raw_examples)} raw rows loaded (revision {revision})")

    print("Fetching official BIRD schema metadata (train_tables.json) ...")
    tables_json_path = raw_dir / "train_tables.json"
    fetch_official_train_tables_json(tables_json_path)
    schemas = parse_database_schemas(tables_json_path, column_meaning, dialect=dialect)
    print(f"  {len(schemas)} database schemas loaded")

    # --- Stage 1: validate raw rows ---
    rejections: list[dict] = []
    valid_raw = []
    for ex in raw_examples:
        reasons = validate_raw_example(ex, schemas)
        if reasons:
            rejections.extend(
                {"row_index": r.row_index, "db_id": r.db_id, "reason": r.reason, "detail": r.detail}
                for r in reasons
            )
        else:
            valid_raw.append(ex)

    # --- Stage 2: db-level split (computed over valid rows' db_ids only) ---
    db_ids = [ex.db_id for ex in valid_raw]
    train_db_ids, validation_db_ids = split_database_ids(db_ids, seed=seed, train_fraction=train_fraction)
    check_split_leakage(train_db_ids, validation_db_ids)

    # --- Stage 3: build prepared examples ---
    schema_serialized_cache: dict[str, str] = {}
    seen_ids: set[str] = set()
    prepared_train: list[PreparedTextToSQLExample] = []
    prepared_validation: list[PreparedTextToSQLExample] = []
    context_kept_count = 0
    context_available_count = 0

    for ex in valid_raw:
        if ex.db_id not in schema_serialized_cache:
            schema_serialized_cache[ex.db_id] = serialize_schema(schemas[ex.db_id])
        schema_serialized = schema_serialized_cache[ex.db_id]

        evidence_available = bool(ex.evidence and ex.evidence.strip())
        business_context_kept = False
        business_context = None
        if evidence_available:
            context_available_count += 1
            example_key = f"{ex.db_id}:{ex.row_index}"
            business_context_kept = should_keep_business_context(example_key, seed, keep_probability)
            if business_context_kept:
                business_context = ex.evidence
                context_kept_count += 1

        prompt = build_prompt(schema_serialized, dialect, ex.question, business_context)
        completion = build_completion(ex.sql)
        example_id = f"{repo_id}:{ex.row_index:05d}"
        split = "train" if ex.db_id in train_db_ids else "validation"

        prepared = PreparedTextToSQLExample(
            example_id=example_id,
            db_id=ex.db_id,
            dialect=dialect,
            question=ex.question,
            business_context=business_context,
            schema_serialized=schema_serialized,
            prompt=prompt,
            completion=completion,
            source=SourceMetadata(
                dataset_repo_id=repo_id,
                dataset_revision=revision,
                row_index=ex.row_index,
                evidence_available=evidence_available,
                business_context_kept=business_context_kept,
            ),
            split=split,
        )

        reasons = validate_prepared_example(prepared, seen_ids)
        if reasons:
            rejections.extend(
                {"row_index": r.row_index, "db_id": r.db_id, "reason": r.reason, "detail": r.detail}
                for r in reasons
            )
            continue

        seen_ids.add(prepared.example_id)
        if split == "train":
            prepared_train.append(prepared)
        else:
            prepared_validation.append(prepared)

    # --- Stage 4: write processed artifacts ---
    train_path = processed_dir / "train.jsonl"
    validation_path = processed_dir / "validation.jsonl"
    with train_path.open("w", encoding="utf-8") as f:
        for ex in prepared_train:
            f.write(ex.model_dump_json() + "\n")
    with validation_path.open("w", encoding="utf-8") as f:
        for ex in prepared_validation:
            f.write(ex.model_dump_json() + "\n")

    # --- Stage 5: validation report ---
    rejection_reason_counts = Counter(r["reason"] for r in rejections)
    accepted = len(prepared_train) + len(prepared_validation)
    leakage = train_db_ids & validation_db_ids

    prompt_lengths = [len(ex.prompt) for ex in prepared_train + prepared_validation]
    completion_lengths = [len(ex.completion) for ex in prepared_train + prepared_validation]

    report = {
        "seed": seed,
        "dialect": dialect,
        "train_db_fraction_config": train_fraction,
        "business_context_keep_probability": keep_probability,
        "total_raw_examples": len(raw_examples),
        "accepted_examples": accepted,
        "rejected_examples": len(rejections),
        "rejection_reason_counts": dict(rejection_reason_counts),
        "rejections": rejections,
        "train_example_count": len(prepared_train),
        "validation_example_count": len(prepared_validation),
        "train_db_count": len(train_db_ids),
        "validation_db_count": len(validation_db_ids),
        "train_db_ids": sorted(train_db_ids),
        "validation_db_ids": sorted(validation_db_ids),
        "db_leakage": sorted(leakage),
        "db_leakage_count": len(leakage),
        "evidence_context_stats": {
            "examples_with_evidence": context_available_count,
            "examples_with_context_kept": context_kept_count,
            "kept_fraction_of_available": (
                round(context_kept_count / context_available_count, 4)
                if context_available_count
                else 0.0
            ),
        },
        "schema_coverage": {
            "databases_with_schema": len(schemas),
            "databases_referenced_by_data": len(set(db_ids)),
        },
        "prompt_length_chars": length_stats(prompt_lengths),
        "completion_length_chars": length_stats(completion_lengths),
    }

    report_path = reports_dir / "data_validation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== Phase 1 data preparation complete ===")
    print(f"accepted: {accepted}  rejected: {len(rejections)}")
    print(f"train examples: {len(prepared_train)}  ({len(train_db_ids)} dbs)")
    print(f"validation examples: {len(prepared_validation)}  ({len(validation_db_ids)} dbs)")
    print(f"db leakage: {len(leakage)}")
    print(f"wrote: {train_path}")
    print(f"wrote: {validation_path}")
    print(f"wrote: {report_path}")


if __name__ == "__main__":
    main()
