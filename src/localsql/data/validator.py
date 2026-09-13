"""Validation of raw and prepared examples, with explicit rejection reasons.

No row is ever silently dropped: every rejection is recorded with a reason
code so the validation report fully accounts for the raw input count.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import sqlglot
from sqlglot.errors import SqlglotError

from localsql.data.models import DatabaseSchema, PreparedTextToSQLExample, RawBirdExample

# Rejection reason codes
EMPTY_QUESTION = "EMPTY_QUESTION"
EMPTY_SQL = "EMPTY_SQL"
MISSING_SCHEMA = "MISSING_SCHEMA"
EMPTY_SERIALIZED_SCHEMA = "EMPTY_SERIALIZED_SCHEMA"
EMPTY_PROMPT = "EMPTY_PROMPT"
COMPLETION_NOT_SQL_ONLY = "COMPLETION_NOT_SQL_ONLY"
COMPLETION_NOT_PARSEABLE_SQL = "COMPLETION_NOT_PARSEABLE_SQL"
DUPLICATE_EXAMPLE_ID = "DUPLICATE_EXAMPLE_ID"

_FORBIDDEN_COMPLETION_MARKERS = ("```", "SQL:", "sql:", "<think", "Explanation:")


@dataclass(frozen=True)
class Rejection:
    row_index: int
    db_id: str
    reason: str
    detail: str = ""


def validate_raw_example(
    example: RawBirdExample, schemas: dict[str, DatabaseSchema]
) -> list[Rejection]:
    """Validate a raw example before any prompt/schema construction."""
    reasons: list[Rejection] = []
    if not example.question or not example.question.strip():
        reasons.append(Rejection(example.row_index, example.db_id, EMPTY_QUESTION))
    if not example.sql or not example.sql.strip():
        reasons.append(Rejection(example.row_index, example.db_id, EMPTY_SQL))
    if example.db_id not in schemas:
        reasons.append(
            Rejection(example.row_index, example.db_id, MISSING_SCHEMA, "db_id has no schema")
        )
    return reasons


def validate_prepared_example(
    example: PreparedTextToSQLExample, seen_ids: set[str]
) -> list[Rejection]:
    """Validate a fully constructed prepared example."""
    reasons: list[Rejection] = []
    row_index = example.source.row_index

    if not example.schema_serialized or not example.schema_serialized.strip():
        reasons.append(Rejection(row_index, example.db_id, EMPTY_SERIALIZED_SCHEMA))
    if not example.prompt or not example.prompt.strip():
        reasons.append(Rejection(row_index, example.db_id, EMPTY_PROMPT))
    if any(marker in example.completion for marker in _FORBIDDEN_COMPLETION_MARKERS):
        reasons.append(
            Rejection(
                row_index,
                example.db_id,
                COMPLETION_NOT_SQL_ONLY,
                "completion contains non-SQL markers",
            )
        )
    elif example.completion.strip():
        try:
            sqlglot.parse_one(example.completion, dialect=example.dialect)
        except SqlglotError as e:
            reasons.append(
                Rejection(row_index, example.db_id, COMPLETION_NOT_PARSEABLE_SQL, str(e)[:200])
            )
    if example.example_id in seen_ids:
        reasons.append(
            Rejection(row_index, example.db_id, DUPLICATE_EXAMPLE_ID, example.example_id)
        )
    return reasons


def check_split_leakage(train_db_ids: Iterable[str], validation_db_ids: Iterable[str]) -> set[str]:
    """Return the set of db_ids present in both splits (empty means no leakage)."""
    overlap = set(train_db_ids) & set(validation_db_ids)
    if overlap:
        raise AssertionError(f"train/validation db_id leakage detected: {overlap}")
    return overlap
