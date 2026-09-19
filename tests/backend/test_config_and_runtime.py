from pathlib import Path

import pytest

from localsql.backend import config as bcfg
from localsql.backend.bootstrap import build_query_service, build_runtime
from localsql.backend.errors import ConfigError, ModelRuntimeError, RuntimeNotConfiguredError
from localsql.backend.models import QueryRequest
from localsql.backend.runtime import LlamaCppRuntime, UnavailableRuntime
from localsql.deploy.runtime import GgufRun, LlamaSettings


def test_default_backend_config_loads_and_matches_conventions():
    cfg = bcfg.load_backend_config(env={})
    assert cfg.execution.max_rows > 0 and cfg.execution.timeout_seconds > 0
    assert [e.id for e in cfg.databases.entries] == ["demo"]
    assert cfg.databases.root.startswith(".artifacts/phase8")  # generated data is under ignored .artifacts
    text = bcfg.DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")
    assert ".gguf" not in text.replace("GGUF", "") or "SCHEMAFORGE_" in text  # only env-var names, no machine paths
    assert "C:\\" not in text and "D:\\" not in text


def test_db_root_env_override():
    cfg = bcfg.load_backend_config(env={bcfg.ENV_DB_ROOT: "somewhere/else"})
    assert cfg.databases.root == "somewhere/else"


def test_invalid_config_is_config_error(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("databases: {root: x}\nexecution: {timeout_seconds: -1, max_rows: 1, max_sql_chars: 1}\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        bcfg.load_backend_config(bad, env={})


def _env(tmp_path):
    files = {}
    for name in ("llama.exe", "base.gguf", "lora.gguf"):
        p = tmp_path / name
        p.write_bytes(b"x")
        files[name] = str(p)
    return {
        bcfg.ENV_LLAMA_EXE: files["llama.exe"],
        bcfg.ENV_BASE_GGUF: files["base.gguf"],
        bcfg.ENV_LORA_GGUF: files["lora.gguf"],
    }


def test_llama_settings_from_env_reuse_phase7_inference_settings(tmp_path):
    cfg = bcfg.load_backend_config(env={})
    env = _env(tmp_path) | {bcfg.ENV_THREADS: "4"}
    s = bcfg.llama_settings_from_env(cfg, env)
    assert s.lora == Path(env[bcfg.ENV_LORA_GGUF]) and s.model == Path(env[bcfg.ENV_BASE_GGUF]) and s.threads == 4
    assert (s.context_size, s.max_new_tokens, s.seed, s.n_gpu_layers) == (8192, 512, 42, 0)  # Phase 7 config values
    assert s.timeout_seconds == cfg.runtime.timeout_seconds
    assert s.guard_prompt_trailing_newline is True  # serving tokenizes the exact assistant-newline envelope


def test_missing_env_is_named_and_falls_back_to_unavailable_runtime(tmp_path):
    cfg = bcfg.load_backend_config(env={})
    with pytest.raises(ConfigError, match="SCHEMAFORGE_LORA_GGUF"):
        bcfg.llama_settings_from_env(cfg, {bcfg.ENV_LLAMA_EXE: "x", bcfg.ENV_BASE_GGUF: "y"})
    rt = build_runtime(cfg, {})
    assert isinstance(rt, UnavailableRuntime)
    with pytest.raises(RuntimeNotConfiguredError):
        rt.generate("p")
    with pytest.raises(ConfigError):
        bcfg.llama_settings_from_env(cfg, _env(tmp_path) | {bcfg.ENV_NGL: "many"})


def _settings(tmp_path):
    cfg = bcfg.load_backend_config(env={})
    return bcfg.llama_settings_from_env(cfg, _env(tmp_path))


def _run(status="ok", raw="SELECT 1\n", error=None):
    def run(prompt, settings):
        run.calls.append((prompt, settings))
        return GgufRun(status, raw if status == "ok" else None, None, 1234.0,
                       {"prompt_tokens": 50, "generated_tokens": 5, "generated_tokens_per_second": 10.5,
                        "prompt_tokens_per_second": 60.0},
                       999, "peak", False, ["cmd"], 0 if status == "ok" else 1, error)  # fmt: skip

    run.calls = []
    return run


def test_llama_runtime_returns_raw_completion_and_metadata(tmp_path):
    run = _run(raw="  SELECT 1\n")
    rt = LlamaCppRuntime(_settings(tmp_path), run=run)
    gen = rt.generate("CANONICAL PROMPT")
    assert gen.raw_completion == "  SELECT 1\n"  # raw: normalization is the service's job
    assert (gen.input_tokens, gen.output_tokens, gen.latency_ms) == (50, 5, 1234.0)
    assert gen.metadata["generated_tokens_per_second"] == 10.5
    assert run.calls[0][0] == "CANONICAL PROMPT" and run.calls[0][1].lora is not None  # hot-LoRA settings


def test_llama_runtime_failure_message_is_generic(tmp_path):
    rt = LlamaCppRuntime(_settings(tmp_path), run=_run("error", error="exit 1: C:\\secret\\path failure"))
    with pytest.raises(ModelRuntimeError) as e:
        rt.generate("p")
    assert "secret" not in e.value.message
    rt = LlamaCppRuntime(_settings(tmp_path), run=_run("timeout"))
    with pytest.raises(ModelRuntimeError, match="timeout"):
        rt.generate("p")


def test_llama_runtime_requires_lora_and_existing_files(tmp_path):
    s = _settings(tmp_path)
    with pytest.raises(ModelRuntimeError, match="hot-LoRA"):
        LlamaCppRuntime(LlamaSettings(**{**s.__dict__, "lora": None}))
    with pytest.raises(ModelRuntimeError, match="not found"):
        LlamaCppRuntime(LlamaSettings(**{**s.__dict__, "model": tmp_path / "missing.gguf"}))


def test_describe_exposes_names_not_paths(tmp_path):
    d = LlamaCppRuntime(_settings(tmp_path), run=_run()).describe()
    assert d["deployment_mode"] == "hot_lora" and d["base_gguf"] == "base.gguf" and d["lora_gguf"] == "lora.gguf"
    assert str(tmp_path) not in str(d)


def test_build_query_service_with_injected_runtime_end_to_end(tmp_path):
    """Composition root wiring (real registry/introspector/safety/executor; fake runtime)."""
    from localsql.backend.demo import create_demo_database
    from tests.backend.helpers import FakeRuntime

    cfg = bcfg.load_backend_config(env={})
    root = tmp_path / "dbs"
    create_demo_database(root / "demo.sqlite")
    cfg = cfg.model_copy(update={"databases": cfg.databases.model_copy(update={"root": str(root)})})
    service = build_query_service(cfg, runtime=FakeRuntime("SELECT COUNT(*) FROM departments"), env={})
    resp = service.query(QueryRequest(database_id="demo", question="How many departments?"))
    assert resp.status == "ok" and resp.result.rows == [[3]]
