"""Thin FastAPI adapter over `QueryService`. No business logic lives here.

    uv run uvicorn localsql.backend.api:create_app_from_env --factory

Endpoints: GET /health, GET /databases, POST /query. Every response carries an
`X-Request-ID` header. Errors use one safe envelope and never echo input,
filesystem paths, or stack traces.
"""

from __future__ import annotations

import re
import uuid
from typing import Any, Callable, Optional, Sequence

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from localsql.backend.errors import BackendError, DatabaseUnavailableError, UnknownDatabaseError
from localsql.backend.models import QueryRequest, QueryResponse
from localsql.backend.observability import get_logger, log_event
from localsql.backend.service import QueryService

REQUEST_ID_HEADER = "X-Request-ID"
_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9_.-]{8,64}$")

# QueryResponse.status -> HTTP status
_STATUS_TO_HTTP = {"ok": 200, "unsafe_sql": 422, "execution_error": 422, "model_error": 502, "schema_error": 500}


def get_query_service(request: Request) -> QueryService:
    return request.app.state.query_service


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "unknown")


def _error_body(request: Request, code: str, message: str, fields: Optional[list] = None) -> dict[str, Any]:
    err: dict[str, Any] = {"code": code, "message": message}
    if fields is not None:
        err["fields"] = fields
    return {"request_id": _request_id(request), "error": err}


def create_app(service: QueryService, cors_origins: Sequence[str] = ()) -> FastAPI:
    app = FastAPI(title="SchemaForge", version="0.8.0")
    app.state.query_service = service
    log = get_logger()

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next: Callable):
        supplied = request.headers.get(REQUEST_ID_HEADER, "")
        request.state.request_id = supplied if _REQUEST_ID_RE.match(supplied) else uuid.uuid4().hex
        response = await call_next(request)
        response.headers[REQUEST_ID_HEADER] = request.state.request_id
        return response

    if cors_origins:
        # Added after the request-id middleware => outermost, so CORS headers
        # also cover error responses. Explicit origins only; no credentials.
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(cors_origins),
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type", REQUEST_ID_HEADER],
            expose_headers=[REQUEST_ID_HEADER],
            allow_credentials=False,
            max_age=600,
        )

    @app.exception_handler(RequestValidationError)
    async def on_validation_error(request: Request, exc: RequestValidationError):
        # Report only field locations/types -- never the offending input values.
        fields = [{"field": ".".join(str(p) for p in e["loc"] if p != "body"), "issue": e["type"]} for e in exc.errors()]
        return JSONResponse(_error_body(request, "invalid_request", "Request validation failed.", fields), status_code=422)

    @app.exception_handler(StarletteHTTPException)
    async def on_http_error(request: Request, exc: StarletteHTTPException):
        code = "not_found" if exc.status_code == 404 else "http_error"
        return JSONResponse(_error_body(request, code, "Request could not be served."), status_code=exc.status_code)

    @app.exception_handler(BackendError)
    async def on_backend_error(request: Request, exc: BackendError):
        status = 404 if isinstance(exc, UnknownDatabaseError) else 503 if isinstance(exc, DatabaseUnavailableError) else 500
        return JSONResponse(_error_body(request, exc.code, exc.message), status_code=status)

    @app.exception_handler(Exception)
    async def on_unhandled(request: Request, exc: Exception):
        log_event(log, "api.unhandled", request_id=_request_id(request), exception=type(exc).__name__)
        return JSONResponse(_error_body(request, "internal_error", "Internal server error."), status_code=500)

    @app.get("/health")
    def health(svc: QueryService = Depends(get_query_service)) -> dict:
        return svc.health()

    @app.get("/databases")
    def databases(svc: QueryService = Depends(get_query_service)) -> dict:
        return {"databases": svc.list_databases()}

    # Plain `def`: FastAPI runs it in a worker thread, so the blocking model
    # subprocess does not stall the event loop.
    @app.post("/query", response_model=QueryResponse)
    def query(body: QueryRequest, request: Request, svc: QueryService = Depends(get_query_service)):
        resp = svc.query(body, request_id=request.state.request_id)
        code = _STATUS_TO_HTTP.get(resp.status, 500)
        if resp.error and resp.error.code == "timeout":
            code = 504
        if resp.error and resp.error.code == "model_not_configured":
            code = 503
        return JSONResponse(resp.model_dump(mode="json"), status_code=code)

    return app


def create_app_from_env() -> FastAPI:
    """Uvicorn factory: builds the service from configs/backend.yaml + env."""
    from localsql.backend.bootstrap import build_query_service
    from localsql.backend.config import cors_origins, load_backend_config

    cfg = load_backend_config()
    return create_app(build_query_service(cfg), cors_origins=cors_origins(cfg))
