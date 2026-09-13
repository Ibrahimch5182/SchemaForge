"""LocalSQL diagnostic checks -- NOT official BIRD metrics.

`parseable != executable != correct`. These diagnostics only establish the
first two. Correctness (`official_correct`) comes exclusively from the
official evaluator (`official_adapter.py`).
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import sqlglot
from sqlglot.errors import SqlglotError

PARSE_OK = "PARSEABLE"
PARSE_FAILED = "NOT_PARSEABLE"

EXEC_OK = "SUCCESS"
EXEC_ERROR = "ERROR"
EXEC_TIMEOUT = "TIMEOUT"
EXEC_SKIPPED = "SKIPPED"  # e.g. not attempted because parsing already failed context is irrelevant here


@dataclass
class PredictionDiagnostic:
    example_id: str
    parse_status: str
    execution_status: str
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    runtime_ms: Optional[float] = None


def _truncate(message: str, limit: int = 300) -> str:
    return message if len(message) <= limit else message[:limit] + "..."


def check_parse(sql: str, dialect: str = "sqlite") -> tuple[str, Optional[str], Optional[str]]:
    """Return (parse_status, error_type, error_message)."""
    try:
        sqlglot.parse_one(sql, dialect=dialect)
    except SqlglotError as e:
        return PARSE_FAILED, type(e).__name__, _truncate(str(e))
    return PARSE_OK, None, None


def check_execution(
    sql: str, db_path: Path, timeout_seconds: float = 30.0
) -> tuple[str, Optional[str], Optional[str], float]:
    """Attempt to execute `sql` read-only against `db_path`.

    Returns (execution_status, error_type, error_message, runtime_ms). This
    only proves the query runs without error -- it says nothing about
    whether the result is correct.
    """
    start = time.perf_counter()
    uri = f"file:{db_path.as_posix()}?mode=ro"
    timed_out = False

    def _abort_if_over_budget() -> int:
        nonlocal timed_out
        if time.perf_counter() - start > timeout_seconds:
            timed_out = True
            return 1  # non-zero return aborts the running SQLite statement
        return 0

    try:
        conn = sqlite3.connect(uri, uri=True)
        conn.set_progress_handler(_abort_if_over_budget, 1000)
        try:
            cursor = conn.cursor()
            cursor.execute(sql)
            cursor.fetchall()
        finally:
            conn.close()
    except sqlite3.Error as e:
        runtime_ms = (time.perf_counter() - start) * 1000
        if timed_out:
            return EXEC_TIMEOUT, type(e).__name__, _truncate(str(e)), runtime_ms
        return EXEC_ERROR, type(e).__name__, _truncate(str(e)), runtime_ms
    runtime_ms = (time.perf_counter() - start) * 1000
    return EXEC_OK, None, None, runtime_ms


def diagnose_prediction(
    example_id: str, sql: str, db_path: Path, dialect: str = "sqlite", timeout_seconds: float = 30.0
) -> PredictionDiagnostic:
    parse_status, parse_err_type, parse_err_msg = check_parse(sql, dialect)
    exec_status, exec_err_type, exec_err_msg, runtime_ms = check_execution(sql, db_path, timeout_seconds)

    error_type = exec_err_type or parse_err_type
    error_message = exec_err_msg or parse_err_msg

    return PredictionDiagnostic(
        example_id=example_id,
        parse_status=parse_status,
        execution_status=exec_status,
        error_type=error_type,
        error_message=error_message,
        runtime_ms=round(runtime_ms, 2),
    )


def summarize_diagnostics(diagnostics: list[PredictionDiagnostic]) -> dict:
    total = len(diagnostics)
    if total == 0:
        return {
            "total": 0,
            "parse_rate": 0.0,
            "execution_success_rate": 0.0,
            "execution_error_count": 0,
            "execution_timeout_count": 0,
        }
    parsed = sum(1 for d in diagnostics if d.parse_status == PARSE_OK)
    executed_ok = sum(1 for d in diagnostics if d.execution_status == EXEC_OK)
    errors = sum(1 for d in diagnostics if d.execution_status == EXEC_ERROR)
    timeouts = sum(1 for d in diagnostics if d.execution_status == EXEC_TIMEOUT)
    return {
        "total": total,
        "parse_rate": round(parsed / total, 4),
        "execution_success_rate": round(executed_ok / total, 4),
        "execution_error_count": errors,
        "execution_timeout_count": timeouts,
    }
