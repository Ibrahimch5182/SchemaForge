"""Backend configuration: `configs/backend.yaml` + environment overrides.

Pydantic + PyYAML only. Machine-specific paths (llama.cpp, GGUFs) come from
the environment, never from committed files.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from localsql.backend.errors import ConfigError
from localsql.backend.registry import DatabaseEntry
from localsql.deploy.config import load_phase7_config
from localsql.backend.server_runtime import ServerSettings
from localsql.deploy.runtime import LlamaSettings

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "backend.yaml"

ENV_CONFIG = "SCHEMAFORGE_BACKEND_CONFIG"
ENV_DB_ROOT = "SCHEMAFORGE_DB_ROOT"
ENV_LLAMA_EXE = "SCHEMAFORGE_LLAMA_EXE"
ENV_BASE_GGUF = "SCHEMAFORGE_BASE_GGUF"
ENV_LORA_GGUF = "SCHEMAFORGE_LORA_GGUF"
ENV_THREADS = "SCHEMAFORGE_LLAMA_THREADS"
ENV_NGL = "SCHEMAFORGE_LLAMA_NGL"
ENV_CORS_ORIGINS = "SCHEMAFORGE_CORS_ORIGINS"  # comma-separated; overrides api.cors_allowed_origins
# Phase 11 (persistent serving / public deployment)
ENV_RUNTIME_KIND = "SCHEMAFORGE_RUNTIME_KIND"  # llama_cpp (subprocess, dev/fallback) | llama_server (persistent)
ENV_SERVER_URL = "SCHEMAFORGE_LLAMA_SERVER_URL"  # private URL of llama-server, e.g. http://model-server:8080
ENV_PARALLEL = "SCHEMAFORGE_LLAMA_PARALLEL"  # server --parallel slots == backend inference concurrency
ENV_MODEL_TIMEOUT = "SCHEMAFORGE_MODEL_TIMEOUT_SECONDS"
ENV_APP_ENV = "SCHEMAFORGE_ENV"  # "production" turns on strict CORS validation and hides /docs
ENV_RATE_LIMIT = "SCHEMAFORGE_RATE_LIMIT_PER_MINUTE"  # per-client POST /query limit; 0 disables
ENV_TRUSTED_PROXY_HOPS = "SCHEMAFORGE_TRUSTED_PROXY_HOPS"  # reverse proxies in front (0 = use the socket peer)
ENV_MAX_BODY_BYTES = "SCHEMAFORGE_MAX_BODY_BYTES"


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class DatabaseEntryConfig(_Frozen):
    id: str
    file: str
    dialect: str = "sqlite"
    description: Optional[str] = None


class DatabasesConfig(_Frozen):
    root: str
    entries: list[DatabaseEntryConfig] = Field(default_factory=list)


class ExecutionConfig(_Frozen):
    timeout_seconds: float = Field(gt=0)
    max_rows: int = Field(gt=0)
    max_sql_chars: int = Field(gt=0)


class ServerConfig(_Frozen):
    """Persistent llama-server client settings (the URL itself comes from the environment)."""

    connect_timeout_seconds: float = Field(default=3.0, gt=0)
    readiness_timeout_seconds: float = Field(default=2.0, gt=0)
    readiness_cache_seconds: float = Field(default=1.0, ge=0)
    cache_prompt: bool = False


class RuntimeConfig(_Frozen):
    kind: str
    phase7_config: str
    timeout_seconds: float = Field(gt=0)
    # Bounded inference concurrency (each request = one multi-GB llama.cpp process).
    max_concurrent_generations: int = Field(default=1, ge=1)
    max_waiting_requests: int = Field(default=2, ge=0)
    queue_wait_seconds: float = Field(default=20.0, ge=0)
    server: ServerConfig = ServerConfig()


class ApiConfig(_Frozen):
    # Browser origins allowed to call the API (the Phase 9 frontend dev/preview servers).
    cors_allowed_origins: list[str] = Field(default_factory=list)
    # Public-demo protections (Phase 11). 0 disables the per-client limiter.
    rate_limit_per_minute: int = Field(default=0, ge=0)
    trusted_proxy_hops: int = Field(default=0, ge=0)
    max_body_bytes: int = Field(default=65536, gt=0)


class LoggingConfig(_Frozen):
    level: str = "INFO"


class BackendConfig(_Frozen):
    databases: DatabasesConfig
    execution: ExecutionConfig
    runtime: RuntimeConfig
    api: ApiConfig = ApiConfig()
    logging: LoggingConfig = LoggingConfig()

    def database_entries(self) -> list[DatabaseEntry]:
        return [DatabaseEntry(e.id, e.file, e.dialect, e.description) for e in self.databases.entries]


def _resolve_repo_path(value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else REPO_ROOT / p


def load_backend_config(path: Optional[Path] = None, env: Optional[Mapping[str, str]] = None) -> BackendConfig:
    env = os.environ if env is None else env
    path = Path(path or env.get(ENV_CONFIG) or DEFAULT_CONFIG_PATH)
    try:
        cfg = BackendConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    except (OSError, yaml.YAMLError, ValidationError) as e:
        raise ConfigError(f"Invalid backend config {path.name}: {e}") from e
    override = env.get(ENV_DB_ROOT)
    if override:
        cfg = cfg.model_copy(update={"databases": cfg.databases.model_copy(update={"root": override})})
    return cfg


def cors_origins(cfg: BackendConfig, env: Optional[Mapping[str, str]] = None) -> list[str]:
    env = os.environ if env is None else env
    raw = env.get(ENV_CORS_ORIGINS)
    if raw is None:
        return list(cfg.api.cors_allowed_origins)
    return [o.strip() for o in raw.split(",") if o.strip()]


def database_root(cfg: BackendConfig) -> Path:
    return _resolve_repo_path(cfg.databases.root)


def llama_settings_from_env(cfg: BackendConfig, env: Optional[Mapping[str, str]] = None) -> LlamaSettings:
    """Phase 7 hot-LoRA settings: paths from env, inference params from the
    Phase 7 config. Raises ConfigError naming any missing variable."""
    env = os.environ if env is None else env
    missing = [n for n in (ENV_LLAMA_EXE, ENV_BASE_GGUF, ENV_LORA_GGUF) if not env.get(n)]
    if missing:
        raise ConfigError("Missing environment variable(s): " + ", ".join(missing))
    p7 = load_phase7_config(_resolve_repo_path(cfg.runtime.phase7_config))
    inf = p7.inference
    try:
        threads = int(env[ENV_THREADS]) if env.get(ENV_THREADS) else None
        ngl = int(env[ENV_NGL]) if env.get(ENV_NGL) else inf.n_gpu_layers
    except ValueError as e:
        raise ConfigError(f"{ENV_THREADS}/{ENV_NGL} must be integers") from e
    return LlamaSettings(
        executable=Path(env[ENV_LLAMA_EXE]),
        model=Path(env[ENV_BASE_GGUF]),
        lora=Path(env[ENV_LORA_GGUF]),
        context_size=inf.context_size,
        max_new_tokens=inf.max_new_tokens,
        seed=inf.seed,
        n_gpu_layers=ngl,
        threads=threads,
        timeout_seconds=cfg.runtime.timeout_seconds,
        guard_prompt_trailing_newline=True,  # serve the exact `assistant<newline>` envelope (see LlamaSettings)
    )


def runtime_kind(cfg: BackendConfig, env: Optional[Mapping[str, str]] = None) -> str:
    env = os.environ if env is None else env
    return (env.get(ENV_RUNTIME_KIND) or cfg.runtime.kind).strip()


def is_production(env: Optional[Mapping[str, str]] = None) -> bool:
    env = os.environ if env is None else env
    return (env.get(ENV_APP_ENV) or "").strip().lower() == "production"


def validate_production_cors(origins: list[str]) -> None:
    """Production CORS must be an explicit HTTPS allow-list: no wildcard, no
    localhost/http, no paths. Fails at startup rather than silently serving."""
    from urllib.parse import urlparse

    if not origins:
        raise ConfigError(f"Production requires an explicit CORS allow-list ({ENV_CORS_ORIGINS}).")
    for o in origins:
        u = urlparse(o)
        if "*" in o or u.scheme != "https" or not u.hostname or u.path not in ("", "/") or u.hostname in ("localhost", "127.0.0.1", "::1"):
            raise ConfigError("Production CORS origins must be explicit https origins (no wildcard, localhost or path).")


def _env_int(env: Mapping[str, str], name: str, default: int) -> int:
    raw = env.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError as e:
        raise ConfigError(f"{name} must be an integer") from e


def api_limits(cfg: BackendConfig, env: Optional[Mapping[str, str]] = None) -> dict[str, int]:
    env = os.environ if env is None else env
    return {
        "rate_limit_per_minute": _env_int(env, ENV_RATE_LIMIT, cfg.api.rate_limit_per_minute),
        "trusted_proxy_hops": _env_int(env, ENV_TRUSTED_PROXY_HOPS, cfg.api.trusted_proxy_hops),
        "max_body_bytes": _env_int(env, ENV_MAX_BODY_BYTES, cfg.api.max_body_bytes),
    }


def parallel_slots(env: Optional[Mapping[str, str]] = None) -> int:
    env = os.environ if env is None else env
    n = _env_int(env, ENV_PARALLEL, 1)
    if n < 1:
        raise ConfigError(f"{ENV_PARALLEL} must be >= 1")
    return n


def server_settings_from_env(cfg: BackendConfig, env: Optional[Mapping[str, str]] = None) -> ServerSettings:
    """Persistent-server client settings: URL from env (never committed), inference
    parameters from the Phase 7 config so serving matches what was benchmarked."""
    env = os.environ if env is None else env
    url = (env.get(ENV_SERVER_URL) or "").strip()
    if not url:
        raise ConfigError(f"Missing environment variable(s): {ENV_SERVER_URL}")
    if not url.lower().startswith(("http://", "https://")):
        raise ConfigError(f"{ENV_SERVER_URL} must be an http(s) URL")
    inf = load_phase7_config(_resolve_repo_path(cfg.runtime.phase7_config)).inference
    try:
        timeout = float(env[ENV_MODEL_TIMEOUT]) if env.get(ENV_MODEL_TIMEOUT) else cfg.runtime.timeout_seconds
    except ValueError as e:
        raise ConfigError(f"{ENV_MODEL_TIMEOUT} must be a number") from e
    if timeout <= 0:
        raise ConfigError(f"{ENV_MODEL_TIMEOUT} must be > 0")
    srv = cfg.runtime.server
    return ServerSettings(
        base_url=url,
        context_size=inf.context_size,
        max_new_tokens=inf.max_new_tokens,
        seed=inf.seed,
        request_timeout_seconds=timeout,
        connect_timeout_seconds=srv.connect_timeout_seconds,
        readiness_timeout_seconds=srv.readiness_timeout_seconds,
        readiness_cache_seconds=srv.readiness_cache_seconds,
        cache_prompt=srv.cache_prompt,
        n_gpu_layers=_env_int(env, ENV_NGL, inf.n_gpu_layers),
        parallel_slots=parallel_slots(env),
    )
