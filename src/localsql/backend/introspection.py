"""Deterministic schema introspection into the repo's canonical `DatabaseSchema`.

The output feeds `localsql.data.schema_serializer.serialize_schema` and
`build_prompt` unchanged, so serving uses the exact training/eval prompt
contract. Conventions match the BIRD-derived training schemas: declared types
upper-cased, PK/FK flags per column, tables in creation order, columns in
declaration order.
"""

from __future__ import annotations

import sqlite3
from typing import Protocol

from localsql.backend.errors import SchemaIntrospectionError
from localsql.backend.registry import RegisteredDatabase
from localsql.backend.sqlite_conn import open_readonly
from localsql.data.models import ColumnSchema, DatabaseSchema, ForeignKeyRef, TableSchema

_TABLE_INFO_SQL = 'SELECT cid, name, type, "notnull", dflt_value, pk FROM pragma_table_info(?) ORDER BY cid'
_FK_LIST_SQL = 'SELECT id, seq, "table", "from", "to" FROM pragma_foreign_key_list(?) ORDER BY id, seq'
_TABLES_SQL = (
    "SELECT name FROM sqlite_master WHERE type = 'table' "
    "AND name NOT LIKE 'sqlite\\_%' ESCAPE '\\' ORDER BY rowid"
)


class SchemaIntrospector(Protocol):
    """One implementation per dialect (a PostgreSQL one can be added later)."""

    def introspect(self, db: RegisteredDatabase) -> DatabaseSchema: ...


class SQLiteSchemaIntrospector:
    def introspect(self, db: RegisteredDatabase) -> DatabaseSchema:
        try:
            conn = open_readonly(db.path)
        except Exception as e:
            raise SchemaIntrospectionError("Schema could not be read.") from e
        try:
            tables = self._read_tables(conn)
        except sqlite3.Error as e:
            raise SchemaIntrospectionError("Schema could not be read.") from e
        finally:
            conn.close()
        if not tables:
            raise SchemaIntrospectionError("Database contains no tables.")
        return DatabaseSchema(db_id=db.id, dialect=db.dialect, tables=tables)

    @staticmethod
    def _table_info(conn: sqlite3.Connection, table: str) -> list[tuple]:
        # (name, declared_type, pk_index) ordered by column id
        return [(name, dtype, pk) for _cid, name, dtype, _nn, _dflt, pk in conn.execute(_TABLE_INFO_SQL, (table,))]

    def _read_tables(self, conn: sqlite3.Connection) -> list[TableSchema]:
        names = [r[0] for r in conn.execute(_TABLES_SQL)]
        info = {n: self._table_info(conn, n) for n in names}
        tables = []
        for name in names:
            fk_by_col = self._foreign_keys(conn, name, info)
            columns = [
                ColumnSchema(
                    name=col,
                    data_type=(dtype or "").strip().upper() or None,
                    is_primary_key=pk > 0,
                    foreign_key=fk_by_col.get(col),
                )
                for col, dtype, pk in info[name]
            ]
            tables.append(TableSchema(name=name, columns=columns))
        return tables

    @staticmethod
    def _foreign_keys(conn: sqlite3.Connection, table: str, info: dict) -> dict[str, ForeignKeyRef]:
        """column -> reference. Composite FKs yield one reference per column;
        a column in several FKs keeps the first (lowest id). An implicit `to`
        column resolves to the referenced table's PK column at the same
        position; if that cannot be determined the FK is omitted, never guessed."""
        out: dict[str, ForeignKeyRef] = {}
        for _id, seq, ref_table, from_col, to_col in conn.execute(_FK_LIST_SQL, (table,)).fetchall():
            if from_col in out:
                continue
            if to_col is None:
                ref_pks = [c for c, _t, pk in sorted(info.get(ref_table, []), key=lambda x: x[2]) if pk > 0]
                if seq >= len(ref_pks):
                    continue
                to_col = ref_pks[seq]
            out[from_col] = ForeignKeyRef(table=ref_table, column=to_col)
        return out
