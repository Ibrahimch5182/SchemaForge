"""Composition root: config (+ environment) -> a wired `QueryService`.

Shared by the FastAPI app, the CLI, and the smoke script so all three run the
same pipeline.
"""

from __future__ import annotations

import os
from typing import Mapping, Optional

from localsql.backend.config import BackendConfig, database_root, llama_settings_from_env, load_backend_config
from localsql.backend.errors import BackendError
from localsql.backend.control import InferenceGate
from localsql.backend.executor import SQLiteReadOnlyExecutor
from localsql.backend.introspection import SQLiteSchemaIntrospector
from localsql.backend.observability import configure_logging
from localsql.backend.preflight import SQLitePreflight
from localsql.backend.registry import DatabaseRegistry
from localsql.backend.runtime import LlamaCppRuntime, ModelRuntime, UnavailableRuntime
from localsql.backend.safety import SQLSafetyPolicy
from localsql.backend.service import DialectBackend, QueryService


def build_runtime(cfg: BackendConfig, env: Optional[Mapping[str, str]] = None) -> ModelRuntime:
    """llama.cpp hot-LoRA runtime from the environment. If it cannot be
    configured the service still starts (health/databases work) and /query
    reports `model_not_configured`."""
    if cfg.runtime.kind != "llama_cpp":
        raise BackendError(f"unsupported runtime kind '{cfg.runtime.kind}'", code="config_error")
    try:
        gate = InferenceGate(
            cfg.runtime.max_concurrent_generations, cfg.runtime.max_waiting_requests, cfg.runtime.queue_wait_seconds
        )
        return LlamaCppRuntime(llama_settings_from_env(cfg, env), gate=gate)
    except BackendError as e:
        return UnavailableRuntime(e.message)


def build_query_service(
    cfg: Optional[BackendConfig] = None,
    runtime: Optional[ModelRuntime] = None,
    env: Optional[Mapping[str, str]] = None,
) -> QueryService:
    env = os.environ if env is None else env
    cfg = cfg or load_backend_config(env=env)
    configure_logging(cfg.logging.level)
    registry = DatabaseRegistry(database_root(cfg), cfg.database_entries())
    dialects = {
        "sqlite": DialectBackend(
            introspector=SQLiteSchemaIntrospector(),
            executor=SQLiteReadOnlyExecutor(cfg.execution.timeout_seconds, cfg.execution.max_rows),
            preflight=SQLitePreflight(min(cfg.execution.timeout_seconds, 5.0)),
        )
    }
    return QueryService(
        registry=registry,
        dialects=dialects,
        runtime=runtime if runtime is not None else build_runtime(cfg, env),
        safety=SQLSafetyPolicy(max_sql_chars=cfg.execution.max_sql_chars),
    )
