"""Loading of official BIRD database schema metadata.

The `birdsql/bird23-train-filtered` rows do not carry schema information
themselves. The legitimate source of table/column/PK/FK metadata is the
official BIRD training release published at bird-bench.github.io, whose
`train.zip` archive contains a Spider-format `train_tables.json` describing
all 69 databases.

That archive is ~8.9 GB (it also bundles per-database SQLite files with
actual row data), which is far more than Phase 1 needs. Because the host
supports HTTP range requests, we extract only the `train/train_tables.json`
entry from the remote zip's central directory instead of downloading the
whole archive.

We never infer schema from gold SQL and never fabricate missing metadata.
"""

from __future__ import annotations

import json
import logging
import zipfile
from pathlib import Path
from typing import Any

from localsql.data.models import ColumnSchema, DatabaseSchema, ForeignKeyRef, TableSchema

logger = logging.getLogger(__name__)

OFFICIAL_TRAIN_ZIP_URL = "https://bird-bench.oss-cn-beijing.aliyuncs.com/train.zip"
OFFICIAL_TABLES_ENTRY = "train/train_tables.json"


def fetch_official_train_tables_json(cache_path: Path) -> Path:
    """Extract `train_tables.json` from the remote official BIRD zip.

    Uses HTTP range requests (via fsspec's HTTPFileSystem) to read only the
    zip central directory plus the single target entry, so the ~8.9 GB
    archive is never downloaded in full. Result is cached at `cache_path`.
    """
    if cache_path.exists():
        return cache_path

    import fsspec  # local import: only needed for this one-time extraction

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    fs = fsspec.filesystem("https")
    with fs.open(OFFICIAL_TRAIN_ZIP_URL, "rb") as remote_zip:
        with zipfile.ZipFile(remote_zip) as zf:
            with zf.open(OFFICIAL_TABLES_ENTRY) as src, cache_path.open("wb") as dst:
                dst.write(src.read())
    return cache_path


def _flatten_primary_keys(primary_keys: list[Any]) -> set[int]:
    """Spider format allows composite PKs as nested lists of column indices."""
    flat: set[int] = set()
    for entry in primary_keys:
        if isinstance(entry, list):
            flat.update(entry)
        else:
            flat.add(entry)
    return flat


def parse_database_schemas(
    tables_json_path: Path,
    column_meaning: dict[str, str] | None = None,
    dialect: str = "sqlite",
) -> dict[str, DatabaseSchema]:
    """Parse the official Spider-format `train_tables.json` into typed schemas.

    `column_meaning` is the optional db_id|table|column -> description map
    sourced from the BIRD dataset repo; descriptions are attached only when
    present, never fabricated.
    """
    column_meaning = column_meaning or {}
    raw = json.loads(tables_json_path.read_text(encoding="utf-8"))

    schemas: dict[str, DatabaseSchema] = {}
    for entry in raw:
        db_id = entry["db_id"]
        table_names = entry["table_names_original"]
        column_names = entry["column_names_original"]
        column_types = entry["column_types"]
        primary_keys = _flatten_primary_keys(entry["primary_keys"])
        foreign_keys = entry["foreign_keys"]

        fk_by_col_idx: dict[int, tuple[int, int]] = {}
        for from_idx, to_idx in foreign_keys:
            fk_by_col_idx[from_idx] = (from_idx, to_idx)

        tables: list[TableSchema] = [TableSchema(name=name, columns=[]) for name in table_names]
        columns_per_table: list[list[ColumnSchema]] = [[] for _ in table_names]

        for col_idx, (table_idx, col_name) in enumerate(column_names):
            if table_idx == -1:  # Spider's synthetic '*' pseudo-column
                continue
            table_name = table_names[table_idx]
            data_type = column_types[col_idx] if col_idx < len(column_types) else None
            is_pk = col_idx in primary_keys

            fk_ref: ForeignKeyRef | None = None
            if col_idx in fk_by_col_idx:
                _, to_idx = fk_by_col_idx[col_idx]
                to_table_idx, to_col_name = column_names[to_idx]
                fk_ref = ForeignKeyRef(table=table_names[to_table_idx], column=to_col_name)

            description = column_meaning.get(f"{db_id}|{table_name}|{col_name}")

            columns_per_table[table_idx].append(
                ColumnSchema(
                    name=col_name,
                    data_type=data_type.upper() if data_type else None,
                    is_primary_key=is_pk,
                    foreign_key=fk_ref,
                    description=description,
                )
            )

        tables = [
            TableSchema(name=table_names[i], columns=columns_per_table[i])
            for i in range(len(table_names))
        ]
        schemas[db_id] = DatabaseSchema(db_id=db_id, dialect=dialect, tables=tables)

    return schemas
