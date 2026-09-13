"""Builds the gold-free generation manifest and isolated grading reference.

Reuses Phase 1's canonical schema models, serializer, and prompt builder
unmodified -- Mini-Dev examples go through the exact same
`serialize_schema` / `build_prompt` functions as BIRD training examples.
"""

from __future__ import annotations

import json
from pathlib import Path

from localsql.data.prompt_builder import build_prompt
from localsql.data.schema_loader import parse_database_schemas
from localsql.data.schema_serializer import serialize_schema

from localsql.benchmark.minidev_loader import load_column_descriptions, load_gold_sql
from localsql.benchmark.models import GenerationExample, GradingExample

EXAMPLE_ID_PREFIX = "bird-mini-dev-sqlite"


def make_example_id(idx: int) -> str:
    return f"{EXAMPLE_ID_PREFIX}-{idx:04d}"


def _build_column_meaning(schema_json_path: Path, databases_dir: Path) -> dict[str, str]:
    """`{db_id}|{table}|{column}` -> description, across all databases in the schema file."""
    entries = json.loads(schema_json_path.read_text(encoding="utf-8"))
    column_meaning: dict[str, str] = {}
    for entry in entries:
        db_id = entry["db_id"]
        per_table = load_column_descriptions(databases_dir, db_id, entry["table_names_original"])
        for table_col, desc in per_table.items():
            table, col = table_col.split("|", 1)
            column_meaning[f"{db_id}|{table}|{col}"] = desc
    return column_meaning


def build_manifests(
    questions: list[dict],
    schema_json_path: Path,
    gold_sql_path: Path,
    databases_dir: Path,
    dialect: str = "sqlite",
) -> tuple[list[GenerationExample], list[GradingExample]]:
    """Build (generation_examples, grading_examples), verifying alignment.

    `questions` must be in official row order (as loaded from the HF
    `mini_dev_sqlite` split) and supplies question/evidence/difficulty
    metadata only. This order is cross-checked against the archive's
    `mini_dev_sqlite_gold.sql`, which the official evaluator reads
    positionally -- if the two sources disagree in order, evaluation would
    silently pair the wrong question with the wrong gold SQL, so a mismatch
    raises loudly rather than proceeding.

    Canonical grading truth: `GradingExample.sql` is sourced from the
    archive's `mini_dev_sqlite_gold.sql` -- the exact file the official
    evaluator scores against -- never from HF's `SQL` field. HF's `SQL`
    field is diagnostic-only (see `analyze_source_consistency`); using it
    for grading would silently diverge from official EX/Soft-F1 on any row
    where the two sources disagree (see docs/EVALUATION.md).
    """
    gold_pairs = load_gold_sql(gold_sql_path)
    if len(gold_pairs) != len(questions):
        raise AssertionError(
            f"question count ({len(questions)}) != gold SQL line count ({len(gold_pairs)})"
        )
    for idx, (q, (_gold_sql, gold_db_id)) in enumerate(zip(questions, gold_pairs)):
        if q["db_id"] != gold_db_id:
            raise AssertionError(
                f"row {idx}: HF db_id={q['db_id']!r} != gold.sql db_id={gold_db_id!r} "
                "(questions and official gold SQL are out of alignment)"
            )

    column_meaning = _build_column_meaning(schema_json_path, databases_dir)
    schemas_by_db = parse_database_schemas(schema_json_path, column_meaning, dialect=dialect)
    schema_text_by_db = {db_id: serialize_schema(schema) for db_id, schema in schemas_by_db.items()}

    generation_examples: list[GenerationExample] = []
    grading_examples: list[GradingExample] = []
    for idx, q in enumerate(questions):
        example_id = make_example_id(idx)
        db_id = q["db_id"]
        if db_id not in schema_text_by_db:
            raise AssertionError(f"row {idx}: no schema available for db_id={db_id!r}")
        schema_text = schema_text_by_db[db_id]
        evidence = (q.get("evidence") or "").strip() or None
        difficulty = q.get("difficulty")
        gold_sql = gold_pairs[idx][0]

        prompt = build_prompt(schema_text, dialect, q["question"], evidence)

        generation_examples.append(
            GenerationExample(
                example_id=example_id,
                db_id=db_id,
                dialect=dialect,
                question=q["question"],
                business_context=evidence,
                serialized_schema=schema_text,
                prompt=prompt,
                difficulty=difficulty,
            )
        )
        grading_examples.append(
            GradingExample(
                example_id=example_id,
                db_id=db_id,
                dialect=dialect,
                sql=gold_sql,  # canonical: archive mini_dev_sqlite_gold.sql, not HF's SQL field
                difficulty=difficulty,
                evidence=evidence,
            )
        )

    gen_ids = [ex.example_id for ex in generation_examples]
    grade_ids = [ex.example_id for ex in grading_examples]
    if len(set(gen_ids)) != len(questions) or len(set(grade_ids)) != len(questions):
        raise AssertionError("generation/grading example_ids are not all unique")
    if gen_ids != grade_ids:
        raise AssertionError("generation and grading example_id order diverged")

    return generation_examples, grading_examples
