"""Phase 11: public-demo networking -- production CORS, docs hiding, rate limiting,
body limits, forwarded-header handling, and static checks of the deployment files."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from localsql.backend import config as bcfg  # noqa: E402
from localsql.backend.api import create_app  # noqa: E402
from localsql.backend.errors import ConfigError  # noqa: E402
from localsql.backend.ratelimit import SlidingWindowLimiter, client_key  # noqa: E402
from tests.backend.helpers import FakeRuntime, make_service  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
Q = {"database_id": "demo", "question": "How many employees are there?"}


def app_client(tmp_path, **kw):
    svc, *_ = make_service(tmp_path, FakeRuntime("SELECT COUNT(*) FROM employees"))
    return TestClient(create_app(svc, **kw), raise_server_exceptions=False)


# ------------------------------------------------------------------ CORS
@pytest.mark.parametrize("origins", [[], ["*"], ["http://app.example.com"], ["https://localhost"], ["https://127.0.0.1"], ["https://x.com/path"], ["https://*.vercel.app"], ["https://ok.com", "*"]])
def test_production_cors_rejects_unsafe_allow_lists(origins):
    with pytest.raises(ConfigError):
        bcfg.validate_production_cors(origins)


def test_production_cors_accepts_explicit_https_origins():
    bcfg.validate_production_cors(["https://schemaforge.vercel.app", "https://demo.example.com/"])


def test_app_from_env_fails_fast_in_production_without_allow_list(monkeypatch):
    from localsql.backend.api import create_app_from_env

    monkeypatch.setenv(bcfg.ENV_APP_ENV, "production")
    monkeypatch.delenv(bcfg.ENV_CORS_ORIGINS, raising=False)
    monkeypatch.setenv(bcfg.ENV_CORS_ORIGINS, "*")
    with pytest.raises(ConfigError):
        create_app_from_env()
    monkeypatch.setenv(bcfg.ENV_CORS_ORIGINS, "")  # empty allow-list also refused
    with pytest.raises(ConfigError):
        create_app_from_env()


def test_cors_allows_only_listed_origin_and_docs_are_hidden_in_production(tmp_path):
    c = app_client(tmp_path, cors_origins=["https://schemaforge.vercel.app"], production=True)
    ok = c.options("/query", headers={"Origin": "https://schemaforge.vercel.app", "Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "content-type,x-request-id"})
    assert ok.headers["access-control-allow-origin"] == "https://schemaforge.vercel.app"
    assert "access-control-allow-credentials" not in ok.headers
    bad = c.options("/query", headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"})
    assert "access-control-allow-origin" not in bad.headers
    assert c.get("/docs").status_code == 404 and c.get("/openapi.json").status_code == 404
    assert app_client(tmp_path / "dev").get("/docs").status_code == 200  # dev keeps the docs


def test_default_and_compose_config_have_no_wildcard_or_hardcoded_production_localhost():
    cfg = bcfg.load_backend_config(env={})
    assert "*" not in cfg.api.cors_allowed_origins
    for name in ("docker-compose.yml", "deploy/production.env.example", "deploy/Caddyfile"):
        text = (ROOT / name).read_text(encoding="utf-8")
        assert "localhost" not in text.lower() and "127.0.0.1" not in text, name


# ------------------------------------------------------------------ rate limiting / body limit
def test_sliding_window_limiter_with_fake_clock():
    now = [0.0]
    lim = SlidingWindowLimiter(2, 60, clock=lambda: now[0])
    assert lim.check("a") == (True, 0) and lim.check("a") == (True, 0)
    allowed, retry = lim.check("a")
    assert not allowed and 1 <= retry <= 60
    assert lim.check("b")[0] is True  # per-client
    now[0] = 61
    assert lim.check("a")[0] is True  # window slid
    tiny = SlidingWindowLimiter(1, 60, max_keys=3, clock=lambda: now[0])
    for i in range(50):
        tiny.check(f"k{i}")
    assert len(tiny._hits) <= 3  # bounded memory


def test_client_key_trusts_only_proxy_appended_hop():
    assert client_key("10.0.0.5", "1.2.3.4", 0) == "10.0.0.5"  # header ignored without a trusted proxy
    assert client_key("10.0.0.5", "6.6.6.6, 1.2.3.4", 1) == "1.2.3.4"  # spoofed left entry ignored
    assert client_key("10.0.0.5", None, 1) == "10.0.0.5" and client_key(None, None, 0) == "unknown"


def test_query_rate_limit_returns_429_envelope_with_request_id_and_retry_after(tmp_path):
    c = app_client(tmp_path, rate_limit_per_minute=2)
    hdr = {"X-Request-ID": "client-req-456"}
    assert c.post("/query", json=Q).status_code == 200
    assert c.post("/query", json=Q).status_code == 200
    r = c.post("/query", json=Q, headers=hdr)
    assert r.status_code == 429 and r.json()["error"]["code"] == "rate_limited"
    assert r.json()["request_id"] == "client-req-456" == r.headers["X-Request-ID"] and int(r.headers["Retry-After"]) >= 1
    assert c.get("/health").status_code == 200 and c.get("/ready").status_code in (200, 503) and c.get("/databases").status_code == 200  # not limited


def test_rate_limit_is_per_client_behind_a_trusted_proxy(tmp_path):
    c = app_client(tmp_path, rate_limit_per_minute=1, trusted_proxy_hops=1)
    assert c.post("/query", json=Q, headers={"X-Forwarded-For": "1.1.1.1"}).status_code == 200
    assert c.post("/query", json=Q, headers={"X-Forwarded-For": "2.2.2.2"}).status_code == 200
    assert c.post("/query", json=Q, headers={"X-Forwarded-For": "9.9.9.9, 1.1.1.1"}).status_code == 429  # spoof can't dodge


def test_spoofed_forwarded_for_is_ignored_without_trusted_proxy(tmp_path):
    c = app_client(tmp_path, rate_limit_per_minute=1)
    assert c.post("/query", json=Q, headers={"X-Forwarded-For": "1.1.1.1"}).status_code == 200
    assert c.post("/query", json=Q, headers={"X-Forwarded-For": "2.2.2.2"}).status_code == 429


def test_rate_limiter_disabled_by_default(tmp_path):
    c = app_client(tmp_path)
    assert all(c.post("/query", json=Q).status_code == 200 for _ in range(5))


def test_oversized_body_is_refused_with_envelope(tmp_path):
    c = app_client(tmp_path, max_body_bytes=200)
    r = c.post("/query", json={**Q, "business_context": "x" * 500}, headers={"X-Request-ID": "client-req-789"})
    assert r.status_code == 413 and r.json()["error"]["code"] == "payload_too_large" and r.headers["X-Request-ID"] == "client-req-789"
    assert c.post("/query", json=Q).status_code == 200
    assert c.post("/query/abcdefgh12/cancel").status_code == 200  # empty POST still fine


def test_api_limits_env_overrides():
    cfg = bcfg.load_backend_config(env={})
    assert bcfg.api_limits(cfg, {}) == {"rate_limit_per_minute": 0, "trusted_proxy_hops": 0, "max_body_bytes": 65536}
    got = bcfg.api_limits(cfg, {bcfg.ENV_RATE_LIMIT: "12", bcfg.ENV_TRUSTED_PROXY_HOPS: "1", bcfg.ENV_MAX_BODY_BYTES: "1000"})
    assert got == {"rate_limit_per_minute": 12, "trusted_proxy_hops": 1, "max_body_bytes": 1000}
    with pytest.raises(ConfigError):
        bcfg.api_limits(cfg, {bcfg.ENV_RATE_LIMIT: "many"})


# ------------------------------------------------------------------ deployment files (static)
def _compose():
    return yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))


def test_compose_keeps_model_server_private_and_backend_behind_proxy():
    services = _compose()["services"]
    assert {"model-server", "backend", "caddy", "seed-db"} <= set(services)
    assert "ports" not in services["model-server"] and "ports" not in services["backend"]  # only the proxy is public
    assert services["caddy"]["ports"]
    nets = _compose()["networks"]
    assert nets["model"].get("internal") is True  # no egress, unreachable from outside
    assert "edge" not in services["model-server"]["networks"]
    for name in ("model-server", "backend", "caddy"):
        assert not services[name].get("privileged")
    assert services["backend"]["environment"]["SCHEMAFORGE_RUNTIME_KIND"] == "llama_server"
    assert services["backend"]["environment"]["SCHEMAFORGE_ENV"] == "production"
    vols = [v for v in services["model-server"]["volumes"] if "/models" in v]
    assert vols and all(v.endswith(":ro") for v in vols)  # model artifacts mounted, never baked in


def test_env_template_and_ignore_files_keep_secrets_and_models_out_of_images_and_git():
    env = (ROOT / "deploy" / "production.env.example").read_text(encoding="utf-8")
    for var in ("SCHEMAFORGE_CORS_ORIGINS", "SCHEMAFORGE_DOMAIN", "MODEL_DIR", "BASE_GGUF_NAME", "LORA_GGUF_NAME", "LLAMA_NGL"):
        assert var in env
    assert "*" not in [l.split("=", 1)[1].strip() for l in env.splitlines() if l.startswith("SCHEMAFORGE_CORS_ORIGINS=")]
    di = (ROOT / ".dockerignore").read_text(encoding="utf-8")
    for pat in (".artifacts", "*.gguf", ".git", "data/", "frontend/node_modules"):
        assert pat in di
    dockerfile = (ROOT / "docker" / "backend.Dockerfile").read_text(encoding="utf-8")
    assert "USER " in dockerfile and "HEALTHCHECK" in dockerfile and ".gguf" not in dockerfile
