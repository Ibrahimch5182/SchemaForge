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


class ExecutionError(BackendError):
    """`code` is one of: timeout, sql_error, authorization_denied, database_unavailable."""

    code = "sql_error"
