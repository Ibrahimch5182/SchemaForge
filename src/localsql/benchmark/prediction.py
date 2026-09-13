"""Loading and validating LocalSQL benchmark prediction files against a manifest."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from localsql.benchmark.models import GenerationExample, PredictionRecord


@dataclass
class PredictionValidation:
    valid_records: dict[str, PredictionRecord] = field(default_factory=dict)
    duplicate_ids: list[str] = field(default_factory=list)
    unknown_ids: list[str] = field(default_factory=list)
    db_id_mismatches: list[str] = field(default_factory=list)
    non_string_sql: list[str] = field(default_factory=list)
    missing_ids: list[str] = field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        return not (
            self.duplicate_ids
            or self.unknown_ids
            or self.db_id_mismatches
            or self.non_string_sql
            or self.missing_ids
        )


def load_predictions_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def validate_predictions(
    raw_records: list[dict], manifest: list[GenerationExample]
) -> PredictionValidation:
    """Validate predictions against the gold-free generation manifest.

    Checks: every expected `example_id` present, no duplicates, `db_id`
    matches the manifest, `predicted_sql` is a string. Never raises on bad
    input -- every problem is reported so callers can decide how to proceed.
    """
    manifest_by_id = {ex.example_id: ex for ex in manifest}
    result = PredictionValidation()
    seen: set[str] = set()

    for raw in raw_records:
        example_id = raw.get("example_id")
        if example_id in seen:
            result.duplicate_ids.append(example_id)
            continue
        seen.add(example_id)

        if example_id not in manifest_by_id:
            result.unknown_ids.append(example_id)
            continue

        sql = raw.get("predicted_sql")
        if not isinstance(sql, str):
            result.non_string_sql.append(example_id)
            continue

        if raw.get("db_id") != manifest_by_id[example_id].db_id:
            result.db_id_mismatches.append(example_id)
            continue

        result.valid_records[example_id] = PredictionRecord.model_validate(raw)

    result.missing_ids = [
        ex.example_id for ex in manifest if ex.example_id not in result.valid_records
    ]
    return result
