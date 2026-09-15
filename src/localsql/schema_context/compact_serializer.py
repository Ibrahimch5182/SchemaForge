"""Compact full-schema serialization (Phase 5A, Task 1B).

A SEPARATE, experimental representation from Phase 1's canonical
`localsql.data.schema_serializer.serialize_schema` -- that function remains
untouched and is still what Phase 1-3 training/evaluation artifacts use.
This module exists purely for the Phase 5A analysis of whether dropping
verbose column descriptions (while keeping every identifier, type, PK, and
FK relationship) meaningfully reduces schema length for the long-context
databases discovered in Phase 4.

Deterministic: the same `DatabaseSchema` value always serializes to the
same bytes (table/column order is taken directly from the source schema,
exactly like the canonical serializer).
"""

from __future__ import annotations

from localsql.data.models import ColumnSchema, DatabaseSchema, TableSchema


def _serialize_column_compact(column: ColumnSchema) -> str:
    parts = [column.name]
    if column.data_type:
        parts.append(column.data_type)
    if column.is_primary_key:
        parts.append("PK")
    if column.foreign_key:
        parts.append(f"FK->{column.foreign_key.table}.{column.foreign_key.column}")
    # No description -- the only difference from the canonical serializer.
    return "  " + " ".join(parts)


def _serialize_table_compact(table: TableSchema) -> str:
    column_lines = ",\n".join(_serialize_column_compact(c) for c in table.columns)
    return f"{table.name}(\n{column_lines}\n)"


def serialize_schema_compact(schema: DatabaseSchema) -> str:
    """Render a `DatabaseSchema` without column descriptions.

    Preserves ALL table names, column names, data types, primary keys, and
    foreign-key relationships -- removes only the trailing `-- description`
    comment the canonical serializer appends when available.
    """
    return "\n\n".join(_serialize_table_compact(t) for t in schema.tables)
