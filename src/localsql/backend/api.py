"""Thin FastAPI adapter over `QueryService`. No business logic lives here.

    uv run uvicorn localsql.backend.api:create_app_from_env --factory

Endpoints: GET /health, GET /ready, GET /databases, POST /query, POST /query/{id}/cancel.
Every response carries an
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
from localsql.backend.ratelimit import SlidingWindowLimiter, client_key
from localsql.backend.service import QueryService
from localsql.backend.taxonomy import RETRY_AFTER_SECONDS, http_status

REQUEST_ID_HEADER = "X-Request-ID"
_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9_.-]{8,64}$")


def get_query_service(request: Request) -> QueryService:
    return request.app.state.query_service


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "unknown")


def _error_body(request: Request, code: str, message: str, fields: Optional[list] = None) -> dict[str, Any]:
    err: dict[str, Any] = {"code": code, "message": message}
    if fields is not None:
        err["fields"] = fields
    return {"request_id": _request_id(request), "error": err}


def create_app(
    service: QueryService,
    cors_origins: Sequence[str] = (),
    *,
    production: bool = False,
    rate_limit_per_minute: int = 0,
    trusted_proxy_hops: int = 0,
    max_body_bytes: int = 65536,
) -> FastAPI:
    """`production=True` hides the interactive API docs. `rate_limit_per_minute>0`
    limits POST /query per client (Phase 11 public demo); `trusted_proxy_hops` says how
    many reverse proxies append to X-Forwarded-For. Oversized bodies are refused."""
    docs = {"docs_url": None, "redoc_url": None, "openapi_url": None} if production else {}
    app = FastAPI(title="SchemaForge", version="0.11.0", **docs)
    app.state.query_service = service
    log = get_logger()
    limiter = SlidingWindowLimiter(rate_limit_per_minute) if rate_limit_per_minute > 0 else None
    app.state.rate_limiter = limiter

    def _early(request: Request, status: int, code: str, message: str, headers: Optional[dict] = None) -> JSONResponse:
        resp = JSONResponse(_error_body(request, code, message), status_code=status, headers=headers)
        resp.headers[REQUEST_ID_HEADER] = request.state.request_id
        return resp

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next: Callable):
        supplied = request.headers.get(REQUEST_ID_HEADER, "")
        request.state.request_id = supplied if _REQUEST_ID_RE.match(supplied) else uuid.uuid4().hex
        if request.method == "POST":
            declared = request.headers.get("content-length", "")
            if declared and (not declared.isdigit() or int(declared) > max_body_bytes):
                # A declared length is enforced here; the reverse proxy additionally caps streamed bodies.
                return _early(request, 413, "payload_too_large", "Request body is too large.")
            if limiter is not None and request.url.path == "/query":
                key = client_key(request.client.host if request.client else None, request.headers.get("x-forwarded-for"), trusted_proxy_hops)
                allowed, retry = limiter.check(key)
                if not allowed:
                    log_event(log, "api.rate_limited", level=30, request_id=request.state.request_id, retry_after_s=retry)
                    return _early(request, 429, "rate_limited", "Too many requests. Try again shortly.", {"Retry-After": str(retry)})
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
            expose_headers=[REQUEST_ID_HEADER, "Retry-After"],
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

    @app.get("/ready")
    def ready(svc: QueryService = Depends(get_query_service)):
        """Readiness: 200 when a model runtime is configured with its artifacts
        present and a database is registered; 503 otherwise. Never runs inference."""
        info = svc.readiness()
        return JSONResponse(info, status_code=200 if info["ready"] else 503)

    @app.get("/databases")
    def databases(svc: QueryService = Depends(get_query_service)) -> dict:
        return {"databases": svc.list_databases()}

    # Plain `def`: FastAPI runs it in a worker thread, so the blocking model
    # subprocess does not stall the event loop.
    @app.post("/query", response_model=QueryResponse)
    def query(body: QueryRequest, request: Request, svc: QueryService = Depends(get_query_service)):
        resp = svc.query(body, request_id=request.state.request_id)
        error_code = resp.error.code if resp.error else None
        headers = {}
        if error_code in RETRY_AFTER_SECONDS:
            headers["Retry-After"] = str(RETRY_AFTER_SECONDS[error_code])
        return JSONResponse(resp.model_dump(mode="json"), status_code=http_status(resp.status, error_code), headers=headers)

    @app.post("/query/{target_request_id}/cancel")
    def cancel_query(target_request_id: str, svc: QueryService = Depends(get_query_service)):
        """Best-effort, idempotent: stops an in-flight query (kills the model
        process / interrupts SQLite). `cancelled` is False if it is unknown or already finished."""
        if not _REQUEST_ID_RE.match(target_request_id):
            return JSONResponse({"request_id": target_request_id[:64], "cancelled": False})
        return {"request_id": target_request_id, "cancelled": svc.cancel(target_request_id)}

    return app


def create_app_from_env() -> FastAPI:
    """Uvicorn factory: builds the service from configs/backend.yaml + env."""
    from localsql.backend.bootstrap import build_query_service
    from localsql.backend.config import api_limits, cors_origins, is_production, load_backend_config, validate_production_cors

    cfg = load_backend_config()
    origins = cors_origins(cfg)
    production = is_production()
    if production:
        validate_production_cors(origins)  # fail fast: explicit https allow-list only
    return create_app(build_query_service(cfg), cors_origins=origins, production=production, **api_limits(cfg))
