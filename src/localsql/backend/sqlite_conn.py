"""Hardened read-only SQLite connections shared by introspection and execution."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from localsql.backend.errors import DatabaseUnavailableError

# Cap any single string/BLOB (e.g. zeroblob(2e9)) so hostile-but-read-only SQL
# cannot exhaust memory. Row/time limits are enforced by the executor.
MAX_VALUE_LENGTH_BYTES = 10_000_000


def open_readonly(path: Path, busy_timeout_s: float = 1.0) -> sqlite3.Connection:
    """Open `path` with SQLite `mode=ro`, `query_only`, extension loading
    disabled, and value-length limits. The caller owns closing."""
    uri = Path(path).resolve().as_uri() + "?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True, timeout=busy_timeout_s, isolation_level=None)
    except sqlite3.Error as e:
        raise DatabaseUnavailableError("Database could not be opened.") from e
    try:
        if hasattr(conn, "enable_load_extension"):
            conn.enable_load_extension(False)
        conn.execute("PRAGMA query_only = ON")
        conn.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, MAX_VALUE_LENGTH_BYTES)
    except Exception:
        conn.close()
        raise
    return conn
