"""Typed backend errors. Messages are safe to show to API callers: they never
contain filesystem paths, stack traces, or result data."""

from __future__ import annotations


class BackendError(Exception):
    code = "backend_error"

    def __init__(self, message: str, *, code: str | None = None):
        super().__init__(message)
        self.message = message
        if code:
            self.code = code


class ConfigError(BackendError):
    code = "config_error"


class UnknownDatabaseError(BackendError):
    code = "unknown_database"


class DatabaseUnavailableError(BackendError):
    """Registered id, but the file is missing / unreadable / escapes the root."""

    code = "database_unavailable"


class SchemaIntrospectionError(BackendError):
    code = "schema_error"


class ModelRuntimeError(BackendError):
    code = "model_error"


class RuntimeNotConfiguredError(ModelRuntimeError):
    code = "model_not_configured"


class ModelBusyError(ModelRuntimeError):
    """All inference slots are taken and the bounded wait queue is full/expired."""

    code = "model_busy"


class ModelTimeoutError(ModelRuntimeError):
    code = "model_timeout"


class RequestCancelledError(BackendError):
    code = "cancelled"


class ExecutionError(BackendError):
    """`code` is one of: timeout, sql_error, authorization_denied, database_unavailable."""

    code = "sql_error"


class PreflightError(BackendError):
    """The SQL is safe and parseable but invalid for THIS database (unknown
    table/column, ...). `detail` is a short, identifier-only hint."""

    code = "invalid_sql"

    def __init__(self, message: str, *, code: str | None = None, detail: str | None = None):
        super().__init__(message, code=code)
        self.detail = detail
