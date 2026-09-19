# syntax=docker/dockerfile:1
# SchemaForge backend (FastAPI + QueryService). Model weights are NEVER in this image: the backend
# only speaks HTTP to the private llama.cpp model-server container.
#   docker build -f docker/backend.Dockerfile -t schemaforge-backend .

FROM python:3.11-slim-bookworm AS build
COPY --from=ghcr.io/astral-sh/uv:0.11.23 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
# Dependency layer first (cached until pyproject/uv.lock change). Default groups minus dev = runtime + api.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
COPY configs ./configs
RUN uv sync --frozen --no-dev

FROM python:3.11-slim-bookworm
RUN useradd --system --uid 10001 --no-create-home --shell /usr/sbin/nologin schemaforge \
 && mkdir -p /data/databases && chown 10001:10001 /data/databases
COPY --from=build /app /app
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    SCHEMAFORGE_DB_ROOT=/data/databases
WORKDIR /app
USER 10001
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=4s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"
# One worker on purpose: the inference gate and rate limiter are per-process (docs/PHASE11.md).
CMD ["uvicorn", "localsql.backend.api:create_app_from_env", "--factory", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-server-header"]
