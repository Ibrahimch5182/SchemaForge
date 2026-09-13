"""Deterministic, compact serialization of `DatabaseSchema` for model prompts.

The same `DatabaseSchema` value always produces byte-for-byte identical
output: table/column order is taken directly from the source schema (itself
deterministic), and no non-deterministic collections (e.g. plain sets/dicts
without stable iteration) are used.
"""

from __future__ import annotations

from localsql.data.models import ColumnSchema, DatabaseSchema, TableSchema


def _serialize_column(column: ColumnSchema) -> str:
    parts = [column.name]
    if column.data_type:
        parts.append(column.data_type)
    if column.is_primary_key:
        parts.append("PK")
    if column.foreign_key:
        parts.append(f"FK->{column.foreign_key.table}.{column.foreign_key.column}")
    line = "  " + " ".join(parts)
    if column.description:
        line += f"  -- {column.description}"
    return line


def _serialize_table(table: TableSchema) -> str:
    column_lines = ",\n".join(_serialize_column(c) for c in table.columns)
    return f"{table.name}(\n{column_lines}\n)"


def serialize_schema(schema: DatabaseSchema) -> str:
    """Render a `DatabaseSchema` as a compact, deterministic text block."""
    return "\n\n".join(_serialize_table(t) for t in schema.tables)
