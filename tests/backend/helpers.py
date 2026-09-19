"""Shared helpers for Phase 8-10 tests. Real temporary SQLite DBs; only the model
runtime is faked (tests never load a GGUF)."""

from __future__ import annotations

import hashlib
import threading
from pathlib import Path
from typing import Callable, Optional, Union

from localsql.backend.control import CancelToken
from localsql.backend.errors import ModelRuntimeError, RequestCancelledError
from localsql.backend.executor import SQLiteReadOnlyExecutor
from localsql.backend.introspection import SQLiteSchemaIntrospector
from localsql.backend.models import ExecutionResult, SafetyDecision
from localsql.backend.preflight import SQLitePreflight
from localsql.backend.registry import DatabaseEntry, DatabaseRegistry
from localsql.backend.runtime import ModelGeneration
from localsql.backend.safety import SQLSafetyPolicy
from localsql.backend.service import DialectBackend, QueryService
from localsql.backend.demo import create_demo_database


class FakeRuntime:
    """Stands in for the llama.cpp runtime. `output` may be a str, an exception
    to raise, or a callable(prompt) -> str."""

    def __init__(self, output: Union[str, Exception, Callable[[str], str]] = "SELECT 1"):
        self.output = output
        self.prompts: list[str] = []
        self.cancel_tokens: list[Optional[CancelToken]] = []

    def generate(self, prompt: str, cancel: Optional[CancelToken] = None) -> ModelGeneration:
        self.prompts.append(prompt)
        self.cancel_tokens.append(cancel)
        out = self.output(prompt) if callable(self.output) else self.output
        if isinstance(out, Exception):
            raise out
        return ModelGeneration(raw_completion=out, latency_ms=12.5, input_tokens=100, output_tokens=7,
                               metadata={"generated_tokens_per_second": 9.0})  # fmt: skip

    def describe(self) -> dict:
        return {"runtime": "fake", "configured": True}


class BlockingRuntime(FakeRuntime):
    """Blocks inside generate() until released or cancelled -- for cancellation/concurrency tests."""

    def __init__(self, output: str = "SELECT 1"):
        super().__init__(output)
        self.started = threading.Event()
        self.release = threading.Event()

    def generate(self, prompt: str, cancel: Optional[CancelToken] = None) -> ModelGeneration:
        self.started.set()
        while not self.release.is_set():
            if cancel is not None and cancel.cancelled:
                raise RequestCancelledError("The request was cancelled.")
            self.release.wait(0.01)
        return super().generate(prompt, cancel)


class SpyExecutor:
    """Wraps the real executor and records every SQL string that reaches it."""

    def __init__(self, inner: SQLiteReadOnlyExecutor):
        self.inner = inner
        self.executed: list[str] = []

    def execute(self, db, sql: str, cancel: Optional[CancelToken] = None) -> ExecutionResult:
        self.executed.append(sql)
        return self.inner.execute(db, sql, cancel=cancel)


class AllowAllSafety(SQLSafetyPolicy):
    """A deliberately bypassed safety layer, for defense-in-depth regression tests."""

    def check(self, sql: Optional[str]) -> SafetyDecision:
        return SafetyDecision(allowed=True)


def make_registry(tmp_path: Path, with_demo: bool = True) -> DatabaseRegistry:
    root = tmp_path / "dbroot"
    root.mkdir(parents=True, exist_ok=True)
    if with_demo:
        create_demo_database(root / "demo.sqlite")
    return DatabaseRegistry(root, [DatabaseEntry("demo", "demo.sqlite", "sqlite", "demo db")])


def make_service(tmp_path: Path, runtime=None, safety=None, max_rows: int = 500, timeout: float = 5.0, preflight: bool = True):
    """`preflight=False` builds the Phase 8 pipeline (safety -> executor), used by
    the defense-in-depth tests that target the executor in isolation."""
    registry = make_registry(tmp_path)
    spy = SpyExecutor(SQLiteReadOnlyExecutor(timeout, max_rows))
    runtime = runtime or FakeRuntime()
    service = QueryService(
        registry=registry,
        dialects={"sqlite": DialectBackend(SQLiteSchemaIntrospector(), spy, SQLitePreflight(timeout) if preflight else None)},
        runtime=runtime,
        safety=safety or SQLSafetyPolicy(),
    )
    return service, runtime, spy, registry


def file_sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def failing_runtime() -> FakeRuntime:
    return FakeRuntime(ModelRuntimeError("Model generation failed."))
