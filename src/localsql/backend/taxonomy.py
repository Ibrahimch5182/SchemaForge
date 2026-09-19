"""The single source of truth for how backend outcomes map to HTTP status.

`QueryResponse.status` says WHICH pipeline stage failed; `error.code` says
why. The frontend presentation (`frontend/src/lib/failure.ts`) and
docs/PHASE10.md follow this table; a test keeps the documentation in sync.
"""

from __future__ import annotations

from typing import Optional

STATUS_HTTP: dict[str, int] = {
    "ok": 200,
    "unsafe_sql": 422,  # deterministic safety policy rejected the generated SQL
    "validation_error": 422,  # safe SQL, but invalid for this database (preflight)
    "execution_error": 422,
    "model_error": 502,
    "schema_error": 500,
    "cancelled": 409,
}

# Code-specific overrides of the status default.
CODE_HTTP: dict[str, int] = {
    "timeout": 504,  # execution time limit
    "model_timeout": 504,
    "model_busy": 429,
    "model_not_configured": 503,
    "database_unavailable": 503,
}

# Non-pipeline errors (JSON error envelope, no QueryResponse body).
ENVELOPE_HTTP: dict[str, int] = {
    "unknown_database": 404,
    "database_unavailable": 503,
    "invalid_request": 422,
    "not_found": 404,
    "internal_error": 500,
}

# Every `error.code` a QueryResponse may carry, grouped by status.
PIPELINE_CODES: dict[str, tuple[str, ...]] = {
    "model_error": (
        "model_not_configured",
        "model_busy",
        "model_timeout",
        "model_error",
        "malformed_model_output",
        "internal_error",
    ),
    "unsafe_sql": (
        "not_read_only_query",
        "multiple_statements",
        "denied_function",
        "mutation",
        "ddl",
        "attach_detach",
        "pragma",
        "transaction",
        "session_command",
        "select_into",
        "parameters",
    ),
    "validation_error": (
        "unknown_table",
        "unknown_column",
        "ambiguous_column",
        "unknown_function",
        "invalid_syntax",
        "invalid_sql",
        "authorization_denied",
    ),
    "execution_error": ("timeout", "sql_error", "authorization_denied", "database_unavailable", "internal_error"),
    "schema_error": ("schema_error", "internal_error"),
    "cancelled": ("cancelled",),
}

# Safety-policy codes that mean "this is not a usable SQL query at all" rather
# than "this is SQL that must not run".
MALFORMED_SAFETY_CODES = frozenset({"empty_sql", "parse_error", "invalid_characters", "too_long"})

RETRY_AFTER_SECONDS = {"model_busy": 5}


def http_status(status: str, code: Optional[str]) -> int:
    if code is not None and code in CODE_HTTP and status != "ok":
        return CODE_HTTP[code]
    return STATUS_HTTP.get(status, 500)


def all_codes() -> set[str]:
    out = {c for codes in PIPELINE_CODES.values() for c in codes}
    return out | set(ENVELOPE_HTTP)
