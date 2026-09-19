"""Database-aware SQL preflight (between the AST safety policy and execution).

`EXPLAIN QUERY PLAN <sql>` makes SQLite compile the statement against the real
schema -- resolving tables, columns and functions -- WITHOUT running it. It
uses the same hardened read-only connection and authorizer as execution, so a
statement that would write is refused here as well. Failures become structured
codes; raw SQLite messages never leave this module (only a short identifier).
"""

from __future__ import annotations

import re
import sqlite3
import time
from typing import Optional, Protocol

from localsql.backend.control import CancelToken
from localsql.backend.errors import PreflightError, RequestCancelledError
from localsql.backend.executor import READ_ONLY_AUTHORIZER
from localsql.backend.registry import RegisteredDatabase
from localsql.backend.sqlite_conn import open_readonly

_IDENT = re.compile(r"^[\w .$\"'`\[\]-]{1,80}$")

_MESSAGES = {
    "unknown_table": "The generated SQL refers to a table that doesn't exist in this database.",
    "unknown_column": "The generated SQL refers to a column that doesn't exist in this database.",
    "ambiguous_column": "The generated SQL uses a column name that matches more than one table.",
    "unknown_function": "The generated SQL calls a function this database doesn't provide.",
    "invalid_syntax": "The generated SQL isn't valid SQLite syntax.",
    "invalid_sql": "The generated SQL isn't valid for this database.",
    "authorization_denied": "The database refused this statement as not permitted.",
}


class SQLPreflight(Protocol):
    def check(self, db: RegisteredDatabase, sql: str, cancel: Optional[CancelToken] = None) -> None: ...


def classify_sqlite_error(message: str) -> tuple[str, Optional[str]]:
    """(code, identifier-hint) for a SQLite compile error message."""
    m = message.strip()
    low = m.lower()
    detail: Optional[str] = None
    if ":" in m:
        tail = m.split(":", 1)[1].strip()
        detail = tail if _IDENT.match(tail) else None
    if low.startswith("no such table"):
        return "unknown_table", detail
    if low.startswith("no such column") or "has no column named" in low:
        return "unknown_column", detail
    if low.startswith("ambiguous column"):
        return "ambiguous_column", detail
    if low.startswith("no such function") or "wrong number of arguments" in low:
        return "unknown_function", detail
    if "not authorized" in low or "readonly" in low:
        return "authorization_denied", None
    if low.startswith("near ") or "syntax error" in low or "incomplete input" in low or "unrecognized token" in low:
        return "invalid_syntax", None
    return "invalid_sql", None


class SQLitePreflight:
    def __init__(self, timeout_seconds: float = 5.0):
        self._timeout = timeout_seconds

    def check(self, db: RegisteredDatabase, sql: str, cancel: Optional[CancelToken] = None) -> None:
        deadline = time.monotonic() + self._timeout
        conn = open_readonly(db.path)
        try:
            conn.set_authorizer(READ_ONLY_AUTHORIZER)
            conn.set_progress_handler(
                lambda: 1 if (time.monotonic() > deadline or (cancel is not None and cancel.cancelled)) else 0, 1000
            )
            try:
                conn.execute("EXPLAIN QUERY PLAN " + sql).fetchall()
            except sqlite3.Error as e:
                if cancel is not None and cancel.cancelled:
                    raise RequestCancelledError("The request was cancelled.") from None
                code, detail = classify_sqlite_error(str(e))
                raise PreflightError(_MESSAGES[code], code=code, detail=detail) from None
        finally:
            conn.close()
