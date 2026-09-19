"""QueryService: the single orchestration of the Phase 8 pipeline.

resolve DB -> introspect schema -> canonical prompt -> model generation ->
existing SQL normalization -> safety validation -> read-only execution ->
structured response.

No FastAPI dependency. Client errors (unknown/unavailable database) raise typed
`BackendError`s; every pipeline outcome after that is a `QueryResponse` with a
`status`. Unsafe SQL is never passed to an executor.
"""

from __future__ import annotations

import hashlib
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from localsql.backend.errors import BackendError, ExecutionError, ModelRuntimeError, SchemaIntrospectionError
from localsql.backend.executor import ReadOnlyExecutor
from localsql.backend.introspection import SchemaIntrospector
from localsql.backend.models import ErrorInfo, QueryRequest, QueryResponse, SafetyDecision, Timings
from localsql.backend.observability import get_logger, log_event
from localsql.backend.registry import DatabaseRegistry
from localsql.backend.runtime import ModelRuntime
from localsql.backend.safety import SQLSafetyPolicy
from localsql.data.prompt_builder import build_prompt
from localsql.data.schema_serializer import serialize_schema
from localsql.model.generation import normalize_predicted_sql


@dataclass(frozen=True)
class DialectBackend:
    """Everything dialect-specific. Adding PostgreSQL later means registering
    another `DialectBackend`; QueryService does not change."""

    introspector: SchemaIntrospector
    executor: ReadOnlyExecutor


def new_request_id() -> str:
    return uuid.uuid4().hex


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 3)


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

    @property
    def runtime(self) -> ModelRuntime:
        return self._runtime

    def list_databases(self) -> list[dict]:
        return self._registry.list()

    def health(self) -> dict[str, Any]:
        return {"status": "ok", "model_runtime": self._runtime.describe()}

    def query(self, request: QueryRequest, request_id: Optional[str] = None) -> QueryResponse:
        rid = request_id or new_request_id()
        t0 = time.perf_counter()
        timings = Timings()
        db_id = request.database_id

        def finish(status: str, **kw: Any) -> QueryResponse:
            timings.total_ms = _ms(t0)
            resp = QueryResponse(request_id=rid, database_id=db_id, status=status, timings=timings, **kw)
            log_event(
                self._log, "query.completed", request_id=rid, database_id=db_id, status=status,
                stage=(resp.error.stage if resp.error else None),
                error_code=(resp.error.code if resp.error else None),
                safety_codes=[r.code for r in resp.safety.reasons] if resp.safety and not resp.safety.allowed else None,
                latency_ms=timings.total_ms,
            )  # fmt: skip
            return resp

        # 1. resolve (client errors propagate as typed exceptions)
        try:
            db = self._registry.resolve(db_id)
        except BackendError as e:
            log_event(self._log, "query.rejected", level=logging.WARNING, request_id=rid, database_id=db_id, stage="resolve", error_code=e.code)
            raise
        dialect = self._dialects[db.dialect]

        # 2. schema
        s = time.perf_counter()
        try:
            schema = dialect.introspector.introspect(db)
        except SchemaIntrospectionError as e:
            timings.schema_ms = _ms(s)
            return finish("schema_error", dialect=db.dialect, error=ErrorInfo(stage="schema", code=e.code, message=e.message))
        except Exception as e:  # noqa: BLE001 - unexpected; never leak details
            timings.schema_ms = _ms(s)
            log_event(self._log, "query.unexpected", level=logging.ERROR, request_id=rid, database_id=db_id, stage="schema", exception=type(e).__name__)
            return finish("schema_error", dialect=db.dialect, error=ErrorInfo(stage="schema", code="internal_error", message="Schema could not be read."))
        timings.schema_ms = _ms(s)

        # 3. canonical prompt -- the SAME builder/serializer as training and evaluation
        prompt = build_prompt(serialize_schema(schema), db.dialect, request.question, request.business_context)
        prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        base = dict(dialect=db.dialect, prompt_sha256=prompt_sha)

        # 4. model
        s = time.perf_counter()
        try:
            gen = self._runtime.generate(prompt)
        except ModelRuntimeError as e:
            timings.model_ms = _ms(s)
            return finish("model_error", **base, model=self._runtime.describe(), error=ErrorInfo(stage="model", code=e.code, message=e.message))
        except Exception as e:  # noqa: BLE001
            timings.model_ms = _ms(s)
            log_event(self._log, "query.unexpected", level=logging.ERROR, request_id=rid, database_id=db_id, stage="model", exception=type(e).__name__)
            return finish("model_error", **base, model=self._runtime.describe(), error=ErrorInfo(stage="model", code="internal_error", message="Model generation failed."))
        timings.model_ms = _ms(s)
        model_info = {
            **self._runtime.describe(),
            "input_tokens": gen.input_tokens,
            "output_tokens": gen.output_tokens,
            "generation_latency_ms": round(gen.latency_ms, 3),
            **gen.metadata,
        }

        # 5. existing normalization (whitespace trim only; never repaired)
        sql = normalize_predicted_sql(gen.raw_completion)

        # 6. safety -- unsafe SQL never reaches the executor
        s = time.perf_counter()
        decision: SafetyDecision = self._safety.check(sql)
        timings.safety_ms = _ms(s)
        if not decision.allowed:
            first = decision.reasons[0]
            return finish("unsafe_sql", **base, model=model_info, generated_sql=sql, safety=decision,
                          error=ErrorInfo(stage="safety", code=first.code, message=first.message))  # fmt: skip

        # 7. read-only execution
        s = time.perf_counter()
        try:
            result = dialect.executor.execute(db, sql)
        except (ExecutionError, BackendError) as e:
            timings.execution_ms = _ms(s)
            return finish("execution_error", **base, model=model_info, generated_sql=sql, safety=decision,
                          error=ErrorInfo(stage="execution", code=e.code, message=e.message))  # fmt: skip
        except Exception as e:  # noqa: BLE001
            timings.execution_ms = _ms(s)
            log_event(self._log, "query.unexpected", level=logging.ERROR, request_id=rid, database_id=db_id, stage="execution", exception=type(e).__name__)
            return finish("execution_error", **base, model=model_info, generated_sql=sql, safety=decision,
                          error=ErrorInfo(stage="execution", code="internal_error", message="Query execution failed."))  # fmt: skip
        timings.execution_ms = _ms(s)
        return finish("ok", **base, model=model_info, generated_sql=sql, safety=decision, result=result)
