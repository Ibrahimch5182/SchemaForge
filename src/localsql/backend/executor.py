"""Read-only SQLite executor (layer 2 of defense in depth).

Independent of the AST safety policy: even if hostile SQL reached `execute`,
the connection itself cannot write. Controls:

- SQLite `mode=ro` URI + `PRAGMA query_only=ON` (open_readonly)
- extension loading disabled; per-value length limit
- authorizer allowing only SELECT / READ / (non-denied) FUNCTION / RECURSIVE
  (so PRAGMA, ATTACH, INSERT/UPDATE/DELETE, DDL, transactions, savepoints are
  all denied by SQLite itself)
- wall-clock deadline via progress handler (interrupts long queries)
- deterministic truncation: fetch `max_rows + 1`, return at most `max_rows`
- one statement only (Python's sqlite3 rejects multi-statement execute)
- connection is always closed
"""

from __future__ import annotations

import math
import sqlite3
import time
from typing import Any, Protocol

from localsql.backend.errors import ExecutionError
from localsql.backend.models import ExecutionResult
from localsql.backend.registry import RegisteredDatabase
from localsql.backend.sqlite_conn import open_readonly

# sqlite3 authorizer action codes (numeric: stable across Python versions)
_SQLITE_OK, _SQLITE_DENY = 0, 1
_SQLITE_READ, _SQLITE_SELECT, _SQLITE_FUNCTION, _SQLITE_RECURSIVE = 20, 21, 31, 33

_DENIED_FUNCTIONS = frozenset({"load_extension", "readfile", "writefile", "edit", "fts3_tokenizer", "zipfile"})
_PROGRESS_OPS = 1000  # VM opcodes between deadline checks


class ReadOnlyExecutor(Protocol):
    def execute(self, db: RegisteredDatabase, sql: str) -> ExecutionResult: ...


def _authorizer(action: int, arg1: Any, arg2: Any, dbname: Any, source: Any) -> int:
    if action in (_SQLITE_SELECT, _SQLITE_RECURSIVE):
        return _SQLITE_OK
    if action == _SQLITE_READ:
        # arg1 = table name; block SQLite's pragma table-valued functions
        if isinstance(arg1, str) and arg1.lower().startswith("pragma_"):
            return _SQLITE_DENY
        return _SQLITE_OK
    if action == _SQLITE_FUNCTION:
        if isinstance(arg2, str) and arg2.lower() in _DENIED_FUNCTIONS:
            return _SQLITE_DENY
        return _SQLITE_OK
    return _SQLITE_DENY


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return f"<blob:{len(value)} bytes>"
    return str(value)


class SQLiteReadOnlyExecutor:
    def __init__(self, timeout_seconds: float = 10.0, max_rows: int = 500):
        if timeout_seconds <= 0 or max_rows <= 0:
            raise ValueError("timeout_seconds and max_rows must be positive")
        self._timeout = float(timeout_seconds)
        self._max_rows = int(max_rows)

    def execute(self, db: RegisteredDatabase, sql: str) -> ExecutionResult:
        start = time.monotonic()
        deadline = start + self._timeout
        timed_out = False

        def progress() -> int:
            nonlocal timed_out
            if time.monotonic() > deadline:
                timed_out = True
                return 1  # non-zero aborts the running statement
            return 0

        conn = open_readonly(db.path)
        try:
            conn.set_authorizer(_authorizer)
            conn.set_progress_handler(progress, _PROGRESS_OPS)
            try:
                cursor = conn.execute(sql)
                columns = [d[0] for d in (cursor.description or [])]
                fetched = cursor.fetchmany(self._max_rows + 1)
            except sqlite3.Error as e:
                raise self._map_error(e, timed_out) from None
        finally:
            conn.close()

        truncated = len(fetched) > self._max_rows
        kept = fetched[: self._max_rows]
        rows = [[_json_safe(v) for v in row] for row in kept]
        return ExecutionResult(
            columns=columns,
            rows=rows,
            returned_row_count=len(rows),
            truncated=truncated,
            max_rows=self._max_rows,
            elapsed_ms=round((time.monotonic() - start) * 1000, 3),
        )

    def _map_error(self, e: sqlite3.Error, timed_out: bool) -> ExecutionError:
        msg = str(e).lower()
        if timed_out or "interrupted" in msg:
            return ExecutionError(f"Query exceeded the {self._timeout:g}s execution limit.", code="timeout")
        if "not authorized" in msg or "readonly" in msg or "read-only" in msg:
            return ExecutionError("The database rejected the operation as not permitted.", code="authorization_denied")
        if "unable to open" in msg or "disk i/o" in msg or "malformed" in msg or "not a database" in msg:
            return ExecutionError("Database is not available.", code="database_unavailable")
        # SQL errors (syntax, no such table/column, ...) carry no paths or data.
        return ExecutionError(str(e)[:300], code="sql_error")
