"""QueryService: the single orchestration of the Phase 8-10 pipeline.

resolve DB -> introspect schema -> canonical prompt -> model generation ->
existing SQL normalization -> safety policy -> database-aware preflight ->
read-only execution -> structured response.

No FastAPI dependency. Client errors (unknown/unavailable database) raise typed
`BackendError`s; every pipeline outcome after that is a `QueryResponse` with a
`status` and a stable `error.code` (see `taxonomy.py`). Unsafe SQL never
reaches preflight or an executor, and no stage claims semantic correctness:
`reliability` records exactly which deterministic checks ran.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from localsql.backend.control import CancelToken
from localsql.backend.errors import (
    BackendError,
    ExecutionError,
    ModelRuntimeError,
    PreflightError,
    RequestCancelledError,
    SchemaIntrospectionError,
)
from localsql.backend.executor import ReadOnlyExecutor
from localsql.backend.introspection import SchemaIntrospector
from localsql.backend.models import (
    ErrorInfo,
    QueryRequest,
    QueryResponse,
    ReliabilityInfo,
    SafetyDecision,
    Timings,
)
from localsql.backend.observability import get_logger, log_event
from localsql.backend.preflight import SQLPreflight
from localsql.backend.registry import DatabaseRegistry
from localsql.backend.runtime import ModelRuntime
from localsql.backend.safety import SQLSafetyPolicy
from localsql.backend.taxonomy import MALFORMED_SAFETY_CODES
from localsql.data.prompt_builder import build_prompt
from localsql.data.schema_serializer import serialize_schema
from localsql.model.generation import normalize_predicted_sql


@dataclass(frozen=True)
class DialectBackend:
    """Everything dialect-specific. Adding PostgreSQL later means registering
    another `DialectBackend`; QueryService does not change."""

    introspector: SchemaIntrospector
    executor: ReadOnlyExecutor
    preflight: Optional[SQLPreflight] = None


def new_request_id() -> str:
    return uuid.uuid4().hex


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 3)


# error.code / status -> structured log event name
_EVENT_BY_CODE = {
    "model_busy": "query.model_busy",
    "model_timeout": "query.model_timeout",
    "malformed_model_output": "query.malformed_output",
    "cancelled": "query.cancelled",
    "timeout": "query.execution_timeout",
}
_EVENT_BY_STATUS = {
    "unsafe_sql": "query.safety_rejected",
    "validation_error": "query.preflight_rejected",
    "model_error": "query.model_failure",
    "execution_error": "query.execution_failure",
    "schema_error": "query.schema_failure",
}


class QueryService:
    def __init__(
        self,
        registry: DatabaseRegistry,
        dialects: Mapping[str, DialectBackend],
        runtime: ModelRuntime,
        safety: SQLSafetyPolicy,
        logger: Optional[logging.Logger] = None,
    ):
        self._registry = registry
        self._dialects = dict(dialects)
        self._runtime = runtime
        self._safety = safety
        self._log = logger or get_logger()
        self._inflight: dict[str, CancelToken] = {}
        self._inflight_lock = threading.Lock()

    @property
    def runtime(self) -> ModelRuntime:
        return self._runtime

    def list_databases(self) -> list[dict]:
        return self._registry.list()

    # ------------------------------------------------------------ health
    def health(self) -> dict[str, Any]:
        info = {**self._runtime.describe(), **self._runtime_readiness()}
        return {"status": "ok", "model_runtime": info}

    def _runtime_readiness(self) -> dict[str, Any]:
        fn = getattr(self._runtime, "readiness", None)
        return dict(fn()) if callable(fn) else {}

    def readiness(self) -> dict[str, Any]:
        """Ready = a model runtime is configured with its artifacts present (subprocess)
        or its model server reachable and loaded (persistent), and at least one database
        is registered. Never runs inference."""
        reasons: list[str] = []
        desc = self._runtime.describe()
        rt = self._runtime_readiness()
        if not desc.get("configured", True):
            reasons.append("model_not_configured")
        elif rt and not rt.get("ready", True):
            reasons.append(rt.get("reason", "model_artifacts_missing"))
        if not self._registry.list():
            reasons.append("no_databases_registered")
        out = {"ready": not reasons, "reasons": reasons, "availability": rt.get("availability")}
        if "serving" in rt:  # persistent server: mode + reachable/loading/ready/busy/saturated (no paths, no URL)
            out["serving"] = rt["serving"]
        return out

    # ------------------------------------------------------ cancellation
    def cancel(self, request_id: str) -> bool:
        """Ask an in-flight request to stop. Idempotent; False if unknown/finished."""
        with self._inflight_lock:
            token = self._inflight.get(request_id)
        if token is None:
            return False
        token.cancel()
        return True

    # ------------------------------------------------------------- query
    def query(self, request: QueryRequest, request_id: Optional[str] = None, cancel: Optional[CancelToken] = None) -> QueryResponse:
        rid = request_id or new_request_id()
        token = cancel or CancelToken()
        with self._inflight_lock:
            self._inflight[rid] = token
        try:
            return self._run(request, rid, token)
        finally:
            with self._inflight_lock:
                if self._inflight.get(rid) is token:
                    del self._inflight[rid]

    def _run(self, request: QueryRequest, rid: str, token: CancelToken) -> QueryResponse:
        t0 = time.perf_counter()
        timings = Timings()
        db_id = request.database_id
        rel = ReliabilityInfo()
        ctx: dict[str, Any] = {}  # dialect / prompt hash / model info / sql / safety accumulated per stage

        def finish(status: str, **kw: Any) -> QueryResponse:
            timings.total_ms = _ms(t0)
            resp = QueryResponse(request_id=rid, database_id=db_id, status=status, timings=timings, reliability=rel, **{**ctx, **kw})
            if resp.error:
                level = logging.ERROR if resp.error.code == "internal_error" else logging.WARNING
                event = _EVENT_BY_CODE.get(resp.error.code) or _EVENT_BY_STATUS.get(status, "query.failure")
                log_event(self._log, event, level=level, request_id=rid, database_id=db_id, stage=resp.error.stage, error_code=resp.error.code)
            log_event(
                self._log, "query.completed", request_id=rid, database_id=db_id, status=status,
                stage=(resp.error.stage if resp.error else None),
                error_code=(resp.error.code if resp.error else None),
                safety_codes=[r.code for r in resp.safety.reasons] if resp.safety and not resp.safety.allowed else None,
                latency_ms=timings.total_ms,
                timings=timings.model_dump(),
                reliability={"safety": rel.safety, "preflight": rel.preflight, "execution": rel.execution},
            )  # fmt: skip
            return resp

        def fail(status: str, stage: str, code: str, message: str, detail: Optional[str] = None, **kw: Any) -> QueryResponse:
            return finish(status, error=ErrorInfo(stage=stage, code=code, message=message, detail=detail), **kw)

        def cancelled(stage: str) -> QueryResponse:
            return fail("cancelled", stage, "cancelled", "The request was cancelled.")

        # 1. resolve (client errors propagate as typed exceptions)
        try:
            db = self._registry.resolve(db_id)
        except BackendError as e:
            log_event(self._log, "query.rejected", level=logging.WARNING, request_id=rid, database_id=db_id, stage="resolve", error_code=e.code)
            raise
        dialect = self._dialects[db.dialect]
        ctx["dialect"] = db.dialect

        # 2. schema
        s = time.perf_counter()
        try:
            schema = dialect.introspector.introspect(db)
        except SchemaIntrospectionError as e:
            timings.schema_ms = _ms(s)
            return fail("schema_error", "schema", e.code, e.message)
        except Exception as e:  # noqa: BLE001 - unexpected; never leak details
            timings.schema_ms = _ms(s)
            log_event(self._log, "query.unexpected", level=logging.ERROR, request_id=rid, database_id=db_id, stage="schema", exception=type(e).__name__)
            return fail("schema_error", "schema", "internal_error", "Schema could not be read.")
        timings.schema_ms = _ms(s)
        if token.cancelled:
            return cancelled("schema")

        # 3. canonical prompt -- the SAME builder/serializer as training and evaluation
        prompt = build_prompt(serialize_schema(schema), db.dialect, request.question, request.business_context)
        ctx["prompt_sha256"] = hashlib.sha256(prompt.encode("utf-8")).hexdigest()

        # 4. model
        s = time.perf_counter()
        try:
            gen = self._runtime.generate(prompt, cancel=token)
        except RequestCancelledError:
            timings.model_ms = _ms(s)
            return cancelled("model")
        except ModelRuntimeError as e:
            timings.model_ms = _ms(s)
            ctx["model"] = self._runtime.describe()
            return fail("model_error", "model", e.code, e.message)
        except Exception as e:  # noqa: BLE001
            timings.model_ms = _ms(s)
            ctx["model"] = self._runtime.describe()
            log_event(self._log, "query.unexpected", level=logging.ERROR, request_id=rid, database_id=db_id, stage="model", exception=type(e).__name__)
            return fail("model_error", "model", "internal_error", "Model generation failed.")
        timings.model_ms = _ms(s)
        ctx["model"] = {
            **self._runtime.describe(),
            "input_tokens": gen.input_tokens,
            "output_tokens": gen.output_tokens,
            "generation_latency_ms": round(gen.latency_ms, 3),
            **gen.metadata,
        }
        if token.cancelled:
            return cancelled("model")

        # 5. existing normalization (whitespace trim only; never repaired)
        sql = normalize_predicted_sql(gen.raw_completion)
        ctx["generated_sql"] = sql

        # 6. safety -- unsafe SQL never reaches preflight or execution
        s = time.perf_counter()
        decision: SafetyDecision = self._safety.check(sql)
        timings.safety_ms = _ms(s)
        ctx["safety"] = decision
        if not decision.allowed:
            rel.safety = "failed"
            first = decision.reasons[0]
            if {r.code for r in decision.reasons} <= MALFORMED_SAFETY_CODES:
                # Not a usable SQL query at all (empty, unparseable, ...): a model-output problem, not a policy violation.
                return fail("model_error", "model", "malformed_model_output", "The model's output was not a usable SQL query.")
            return fail("unsafe_sql", "safety", first.code, first.message)
        rel.safety = "passed"

        # 7. database-aware preflight: compile against the real schema, no execution
        if dialect.preflight is not None:
            s = time.perf_counter()
            try:
                dialect.preflight.check(db, sql, cancel=token)
            except RequestCancelledError:
                timings.preflight_ms = _ms(s)
                return cancelled("preflight")
            except PreflightError as e:
                timings.preflight_ms = _ms(s)
                rel.preflight = "failed"
                return fail("validation_error", "preflight", e.code, e.message, detail=e.detail)
            except BackendError as e:
                timings.preflight_ms = _ms(s)
                rel.preflight = "failed"
                return fail("execution_error", "execution", e.code, e.message)
            except Exception as e:  # noqa: BLE001
                timings.preflight_ms = _ms(s)
                rel.preflight = "failed"
                log_event(self._log, "query.unexpected", level=logging.ERROR, request_id=rid, database_id=db_id, stage="preflight", exception=type(e).__name__)
                return fail("validation_error", "preflight", "invalid_sql", "The generated SQL couldn't be validated for this database.")
            timings.preflight_ms = _ms(s)
            rel.preflight = "passed"
        if token.cancelled:
            return cancelled("preflight")

        # 8. read-only execution
        s = time.perf_counter()
        try:
            result = dialect.executor.execute(db, sql, cancel=token)
        except RequestCancelledError:
            timings.execution_ms = _ms(s)
            return cancelled("execution")
        except (ExecutionError, BackendError) as e:
            timings.execution_ms = _ms(s)
            rel.execution = "failed"
            return fail("execution_error", "execution", e.code, e.message)
        except Exception as e:  # noqa: BLE001
            timings.execution_ms = _ms(s)
            rel.execution = "failed"
            log_event(self._log, "query.unexpected", level=logging.ERROR, request_id=rid, database_id=db_id, stage="execution", exception=type(e).__name__)
            return fail("execution_error", "execution", "internal_error", "Query execution failed.")
        timings.execution_ms = _ms(s)
        rel.execution = "passed"
        return finish("ok", result=result)
