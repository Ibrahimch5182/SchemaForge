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


class RuntimeConfig(_Frozen):
    kind: str
    phase7_config: str
    timeout_seconds: float = Field(gt=0)
    # Bounded inference concurrency (each request = one multi-GB llama.cpp process).
    max_concurrent_generations: int = Field(default=1, ge=1)
    max_waiting_requests: int = Field(default=2, ge=0)
    queue_wait_seconds: float = Field(default=20.0, ge=0)


class ApiConfig(_Frozen):
    # Browser origins allowed to call the API (the Phase 9 frontend dev/preview servers).
    cors_allowed_origins: list[str] = Field(default_factory=list)


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
