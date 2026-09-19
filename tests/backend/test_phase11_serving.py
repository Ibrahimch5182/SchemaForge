"""Phase 11: persistent llama.cpp serving runtime, runtime selection, readiness,
prompt parity, cancellation/timeout/saturation. No GGUF, Docker, network or GPU:
the llama-server HTTP API is mocked with `httpx.MockTransport`."""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path

import httpx
import pytest

from localsql.backend import config as bcfg
from localsql.backend.bootstrap import build_query_service, build_runtime
from localsql.backend.control import CancelToken, InferenceGate
from localsql.backend.errors import BackendError, ConfigError, ModelBusyError, ModelRuntimeError, ModelTimeoutError, RequestCancelledError
from localsql.backend.models import QueryRequest
from localsql.backend.runtime import LlamaCppRuntime, UnavailableRuntime
from localsql.backend.server_runtime import LlamaServerRuntime, ServerSettings, build_completion_payload
from localsql.deploy.process import ProcessResult
from localsql.deploy.runtime import LlamaSettings, build_chatml_prompt, run_gguf_once
from tests.backend.helpers import make_service

URL = "http://model-server.internal:8080"


def sse(*chunks: dict) -> bytes:
    return b"".join(b"data: " + json.dumps(c).encode() + b"\n\n" for c in chunks)


FINAL = {
    "content": "",
    "stop": True,
    "stop_type": "eos",
    "tokens_evaluated": 321,
    "tokens_predicted": 7,
    "timings": {"prompt_ms": 100.0, "predicted_ms": 700.0, "prompt_per_second": 3210.0, "predicted_per_second": 10.0},
}


def ok_stream(text: str = "SELECT COUNT(*) FROM employees", final: dict | None = None) -> bytes:
    half = len(text) // 2
    return sse({"content": text[:half], "stop": False}, {"content": text[half:], "stop": False}, final or FINAL)


class Server:
    """Scriptable fake llama-server. Records every request."""

    def __init__(self, completion=None, health=200):
        self.completion = completion or (lambda req: httpx.Response(200, content=ok_stream()))
        self.health = health
        self.requests: list[httpx.Request] = []

    def handler(self, req: httpx.Request) -> httpx.Response:
        self.requests.append(req)
        if req.url.path == "/health":
            if isinstance(self.health, Exception):
                raise self.health
            return httpx.Response(self.health, json={"status": "ok" if self.health == 200 else "loading"})
        if req.url.path == "/completion":
            out = self.completion(req)
            if isinstance(out, Exception):
                raise out
            return out
        return httpx.Response(404)

    @property
    def paths(self) -> list[str]:
        return [r.url.path for r in self.requests]

    def factory(self):
        return lambda s: httpx.Client(base_url=s.base_url, transport=httpx.MockTransport(self.handler), timeout=5)


def settings(**kw) -> ServerSettings:
    base = dict(base_url=URL, context_size=8192, max_new_tokens=512, seed=42, request_timeout_seconds=5.0, readiness_cache_seconds=0.0)
    return ServerSettings(**{**base, **kw})


def runtime_for(server: Server, gate=None, **kw) -> LlamaServerRuntime:
    return LlamaServerRuntime(settings(**kw), gate=gate, client_factory=server.factory())


class BlockingStream(httpx.SyncByteStream):
    """A response body that never finishes until closed (a long prefill/generation)."""

    def __init__(self):
        self.started = threading.Event()
        self.closed = threading.Event()

    def __iter__(self):
        self.started.set()
        self.closed.wait(10)
        return iter(())

    def close(self):
        self.closed.set()


# ------------------------------------------------------------------ request format / parsing
def test_request_format_is_greedy_chatml_streaming():
    srv = Server()
    gen = runtime_for(srv).generate("CANON PROMPT")
    (req,) = srv.requests
    assert req.method == "POST" and req.url.path == "/completion"
    body = json.loads(req.content)
    assert body == {
        "prompt": "<|im_start|>user\nCANON PROMPT<|im_end|>\n<|im_start|>assistant\n",
        "n_predict": 512, "temperature": 0, "top_k": 1, "seed": 42, "cache_prompt": False, "stream": True,
    }  # fmt: skip
    assert body["prompt"].endswith("assistant\n")  # the Phase 8 trailing-newline drift must not return
    assert gen.raw_completion == "SELECT COUNT(*) FROM employees"


def test_output_parsing_metadata_and_raw_passthrough():
    srv = Server(lambda r: httpx.Response(200, content=ok_stream(" SELECT 1 ")))
    gen = runtime_for(srv).generate("p")
    assert gen.raw_completion == " SELECT 1 "  # raw: normalization is QueryService's job, never the runtime's
    assert (gen.input_tokens, gen.output_tokens) == (321, 7)
    m = gen.metadata
    assert m["runtime_mode"] == "persistent_server" and m["generated_tokens_per_second"] == 10.0
    assert m["prompt_tokens_per_second"] == 3210.0 and m["server_generation_ms"] == 700.0 and m["stop_type"] == "eos"
    assert m["queue_wait_ms"] >= 0 and gen.latency_ms > 0


def test_sse_comments_and_blank_lines_are_ignored():
    body = b": keep-alive\n\n" + ok_stream("SELECT 2") + b"\n"
    assert runtime_for(Server(lambda r: httpx.Response(200, content=body))).generate("p").raw_completion == "SELECT 2"


# ------------------------------------------------------------------ prompt parity
def test_prompt_parity_with_subprocess_runtime(monkeypatch):
    """Both runtimes must put the SAME token text in front of the model: the ChatML envelope ending
    `assistant<newline>`. llama-completion `-f` strips one trailing newline, so the subprocess runtime
    writes one extra; the server receives the string verbatim."""
    seen: dict = {}

    def fake_run_captured(cmd, timeout=None, cwd=None, poll_interval_s=0.05, should_cancel=None):
        seen["file_text"] = Path(cmd[cmd.index("-f") + 1]).read_bytes().decode("utf-8")
        return ProcessResult(0, "SELECT 1", "", 1.0, False, None, "n/a")

    monkeypatch.setattr("localsql.deploy.runtime.run_captured", fake_run_captured)
    canon = "### Schema\nCREATE TABLE t (a INT);\n\n### Question\nHow many?\n"
    ls = LlamaSettings(executable=Path("x"), model=Path("m"), context_size=8192, max_new_tokens=512, seed=42,
                       lora=Path("l"), guard_prompt_trailing_newline=True)  # fmt: skip
    run_gguf_once(canon, ls)
    server_prompt = build_completion_payload(canon, settings())["prompt"]
    assert seen["file_text"] == server_prompt + "\n"  # file minus the ONE newline llama.cpp strips == server prompt
    assert server_prompt == build_chatml_prompt(canon) and server_prompt.endswith("<|im_start|>assistant\n")


def test_generation_settings_parity_with_phase7_config():
    cfg = bcfg.load_backend_config(env={})
    s = bcfg.server_settings_from_env(cfg, {bcfg.ENV_SERVER_URL: URL})
    payload = build_completion_payload("p", s)
    assert (s.context_size, s.max_new_tokens, s.seed) == (8192, 512, 42)  # the Phase 7 inference values
    assert payload["temperature"] == 0 and payload["top_k"] == 1 and payload["seed"] == 42 and payload["n_predict"] == 512
    assert s.cache_prompt is False  # deterministic cold prefill by default


def test_service_hands_both_runtimes_the_identical_canonical_prompt(tmp_path):
    from tests.backend.helpers import FakeRuntime

    a = FakeRuntime("SELECT COUNT(*) FROM employees")
    svc, *_ = make_service(tmp_path, a)
    svc.query(QueryRequest(database_id="demo", question="How many employees?"))
    canonical = a.prompts[0]
    srv = Server()
    svc2, *_ = make_service(tmp_path / "b", runtime_for(srv))
    svc2.query(QueryRequest(database_id="demo", question="How many employees?"))
    assert json.loads(srv.requests[0].content)["prompt"] == build_chatml_prompt(canonical)


# ------------------------------------------------------------------ failures
@pytest.mark.parametrize(
    "completion",
    [
        lambda r: httpx.Response(200, content=b"data: {not json}\n\n"),
        lambda r: httpx.Response(200, content=b'data: ["a"]\n\n'),
        lambda r: httpx.Response(200, content=sse({"content": 5, "stop": False})),
        lambda r: httpx.Response(200, content=sse({"content": "SELECT", "stop": False})),  # never sends stop
        lambda r: httpx.Response(200, content=sse({"error": {"code": 500, "message": "boom at /models/secret.gguf"}})),
        lambda r: httpx.Response(500, content=b"internal /models/secret.gguf"),
        lambda r: httpx.Response(503, json={"error": {"message": "Loading model"}}),
        lambda r: httpx.Response(400, json={"error": {"message": "exceeds the available context size"}}),
        lambda r: httpx.ConnectError("refused"),
        lambda r: httpx.ReadError("reset"),
    ],
)
def test_server_failures_are_generic_structured_errors_and_free_the_slot(completion, caplog):
    gate = InferenceGate(1, 0, 0)
    rt = runtime_for(Server(completion), gate=gate)
    with caplog.at_level(logging.DEBUG, logger="schemaforge.backend"):
        with pytest.raises(ModelRuntimeError) as e:
            rt.generate("p")
    assert e.value.code == "model_error" and e.value.message == "Model generation failed."
    assert "secret" not in e.value.message and URL not in e.value.message
    assert gate.snapshot()["running"] == 0  # slot released on every failure path
    assert any(json.loads(r.getMessage()).get("event") == "runtime.failure" for r in caplog.records)


def test_error_event_503_maps_to_busy():
    rt = runtime_for(Server(lambda r: httpx.Response(200, content=sse({"error": {"code": 503, "message": "no slot"}}))))
    with pytest.raises(ModelBusyError):
        rt.generate("p")


def test_timeout_closes_the_connection_and_releases_the_slot():
    stream = BlockingStream()
    gate = InferenceGate(1, 0, 0)
    rt = runtime_for(Server(lambda r: httpx.Response(200, stream=stream)), gate=gate, request_timeout_seconds=0.3)
    t0 = time.monotonic()
    with pytest.raises(ModelTimeoutError):
        rt.generate("p")
    assert time.monotonic() - t0 < 3 and stream.closed.is_set()
    assert gate.snapshot()["running"] == 0


def test_cancellation_aborts_the_http_request_promptly():
    stream = BlockingStream()
    gate = InferenceGate(1, 0, 0)
    rt = runtime_for(Server(lambda r: httpx.Response(200, stream=stream)), gate=gate)
    token = CancelToken()
    errors: list[Exception] = []

    def go():
        try:
            rt.generate("p", cancel=token)
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    t = threading.Thread(target=go)
    t.start()
    assert stream.started.wait(3)
    t0 = time.monotonic()
    token.cancel()
    t.join(3)
    assert not t.is_alive() and time.monotonic() - t0 < 2
    assert len(errors) == 1 and isinstance(errors[0], RequestCancelledError)
    assert stream.closed.is_set() and gate.snapshot()["running"] == 0


def test_already_cancelled_request_never_contacts_the_server():
    srv = Server()
    token = CancelToken()
    token.cancel()
    with pytest.raises(RequestCancelledError):
        runtime_for(srv).generate("p", cancel=token)
    assert srv.requests == []


# ------------------------------------------------------------------ concurrency
def _quiet(rt):
    """Background generate() whose stream ends without a final chunk once the test releases it."""
    try:
        rt.generate("p")
    except ModelRuntimeError:
        pass


def test_saturation_is_bounded_and_immediate_then_recovers():
    stream = BlockingStream()
    calls = {"n": 0}

    def completion(r):
        calls["n"] += 1
        return httpx.Response(200, stream=stream) if calls["n"] == 1 else httpx.Response(200, content=ok_stream())

    gate = InferenceGate(1, 0, 0)  # one slot, no queue
    rt = runtime_for(Server(completion), gate=gate)
    t = threading.Thread(target=lambda: _quiet(rt), daemon=True)
    t.start()
    assert stream.started.wait(3)
    assert rt.readiness()["serving"]["state"] == "saturated"
    t0 = time.monotonic()
    with pytest.raises(ModelBusyError):
        rt.generate("p2")
    assert time.monotonic() - t0 < 1  # rejected immediately: no unbounded queue
    stream.close()
    t.join(3)
    assert rt.generate("p3").raw_completion and gate.snapshot() ["running"] == 0
    assert rt.readiness()["serving"]["state"] == "ready"


def test_parallel_slots_allow_that_many_concurrent_requests():
    streams = [BlockingStream(), BlockingStream()]
    it = iter(streams)
    lock = threading.Lock()

    def completion(r):
        with lock:
            return httpx.Response(200, stream=next(it))

    gate = InferenceGate(2, 0, 0)
    rt = runtime_for(Server(completion), gate=gate, parallel_slots=2)
    ts = [threading.Thread(target=lambda: _quiet(rt), daemon=True) for _ in range(2)]
    [t.start() for t in ts]
    assert all(s.started.wait(3) for s in streams)
    assert gate.snapshot()["running"] == 2
    [s.close() for s in streams]
    [t.join(3) for t in ts]
    assert gate.snapshot()["running"] == 0


# ------------------------------------------------------------------ readiness
def test_readiness_states_never_run_inference():
    srv = Server(health=200)
    rt = runtime_for(srv)
    r = rt.readiness()
    assert r["ready"] is True and r["serving"] == {
        "mode": "persistent_server", "state": "ready",
        "checks": {"server_configured": True, "server_reachable": True, "model_ready": True},
    }  # fmt: skip
    srv.health = 503
    r = rt.readiness()
    assert r["ready"] is False and r["reason"] == "model_server_loading" and r["serving"]["state"] == "loading"
    assert r["serving"]["checks"] == {"server_configured": True, "server_reachable": True, "model_ready": False}
    srv.health = httpx.ConnectError("refused")
    r = rt.readiness()
    assert r["ready"] is False and r["reason"] == "model_server_unreachable" and r["serving"]["state"] == "unreachable"
    assert r["serving"]["checks"]["server_reachable"] is False
    assert set(srv.paths) == {"/health"}  # never /completion


def test_readiness_is_cached_briefly():
    srv = Server()
    rt = runtime_for(srv, readiness_cache_seconds=60)
    for _ in range(5):
        rt.readiness()
    assert srv.paths.count("/health") == 1


def test_describe_leaks_no_url_or_paths():
    d = runtime_for(Server()).describe()
    assert d["runtime"] == "llama_server" and d["runtime_mode"] == "persistent_server" and d["configured"] is True
    assert "model-server" not in json.dumps(d) and "8080" not in json.dumps(d)


# ------------------------------------------------------------------ service / API integration
def test_query_service_pipeline_reuses_the_same_server(tmp_path):
    srv = Server()
    rt = runtime_for(srv)
    svc, *_ = make_service(tmp_path, rt)
    for _ in range(3):
        resp = svc.query(QueryRequest(database_id="demo", question="How many employees are there?"))
        assert resp.status == "ok" and resp.result.rows == [[12]]
        assert resp.model["runtime_mode"] == "persistent_server" and resp.model["generated_tokens_per_second"] == 10.0
        assert "queue_wait_ms" in resp.model
    assert srv.paths == ["/completion"] * 3
    assert rt.gate.snapshot()["running"] == 0


def test_query_service_maps_server_outage_to_model_error(tmp_path):
    svc, *_ = make_service(tmp_path, runtime_for(Server(lambda r: httpx.ConnectError("refused"))))
    resp = svc.query(QueryRequest(database_id="demo", question="q"))
    assert resp.status == "model_error" and resp.error.code == "model_error" and URL not in resp.model_dump_json()


def test_health_and_ready_endpoints(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from localsql.backend.api import create_app

    srv = Server()
    svc, *_ = make_service(tmp_path, runtime_for(srv))
    c = TestClient(create_app(svc), raise_server_exceptions=False)
    h = c.get("/health").json()
    assert h["status"] == "ok" and h["model_runtime"]["runtime_mode"] == "persistent_server" and h["model_runtime"]["ready"] is True
    r = c.get("/ready")
    assert r.status_code == 200 and r.json()["ready"] is True and r.json()["serving"]["state"] == "ready"
    srv.health = 503
    r = c.get("/ready")
    assert r.status_code == 503 and r.json()["reasons"] == ["model_server_loading"] and r.json()["serving"]["state"] == "loading"
    srv.health = httpx.ConnectError("x")
    r = c.get("/ready")
    assert r.status_code == 503 and r.json()["reasons"] == ["model_server_unreachable"]
    assert c.get("/health").status_code == 200  # backend stays alive while the model server is down
    assert "/completion" not in srv.paths and URL not in c.get("/ready").text


# ------------------------------------------------------------------ runtime selection / config
def _fake_files(tmp_path):
    files = {}
    for n in ("llama.exe", "base.gguf", "lora.gguf"):
        (tmp_path / n).write_bytes(b"x")
        files[n] = str(tmp_path / n)
    return {bcfg.ENV_LLAMA_EXE: files["llama.exe"], bcfg.ENV_BASE_GGUF: files["base.gguf"], bcfg.ENV_LORA_GGUF: files["lora.gguf"]}


def test_runtime_selection(tmp_path):
    cfg = bcfg.load_backend_config(env={})
    assert cfg.runtime.kind == "llama_cpp"  # subprocess stays the development default
    assert isinstance(build_runtime(cfg, _fake_files(tmp_path)), LlamaCppRuntime)  # fallback preserved
    env = {bcfg.ENV_RUNTIME_KIND: "llama_server", bcfg.ENV_SERVER_URL: URL, bcfg.ENV_PARALLEL: "2", bcfg.ENV_NGL: "33"}
    rt = build_runtime(cfg, env)
    assert isinstance(rt, LlamaServerRuntime)
    assert rt.gate.max_concurrent == 2 and rt.describe()["n_gpu_layers"] == 33 and rt.describe()["parallel_slots"] == 2
    assert rt.gate.max_waiting == cfg.runtime.max_waiting_requests  # queue stays bounded


def test_server_runtime_without_url_or_bad_config_degrades_safely():
    cfg = bcfg.load_backend_config(env={})
    rt = build_runtime(cfg, {bcfg.ENV_RUNTIME_KIND: "llama_server"})
    assert isinstance(rt, UnavailableRuntime) and bcfg.ENV_SERVER_URL in rt.describe()["reason"]
    assert isinstance(build_runtime(cfg, {bcfg.ENV_RUNTIME_KIND: "llama_server", bcfg.ENV_SERVER_URL: "ftp://x"}), UnavailableRuntime)
    assert isinstance(build_runtime(cfg, {bcfg.ENV_RUNTIME_KIND: "llama_server", bcfg.ENV_SERVER_URL: URL, bcfg.ENV_PARALLEL: "0"}), UnavailableRuntime)
    with pytest.raises(BackendError):
        build_runtime(cfg, {bcfg.ENV_RUNTIME_KIND: "vllm"})


def test_model_timeout_env_override():
    cfg = bcfg.load_backend_config(env={})
    s = bcfg.server_settings_from_env(cfg, {bcfg.ENV_SERVER_URL: URL, bcfg.ENV_MODEL_TIMEOUT: "300"})
    assert s.request_timeout_seconds == 300 and bcfg.server_settings_from_env(cfg, {bcfg.ENV_SERVER_URL: URL}).request_timeout_seconds == cfg.runtime.timeout_seconds
    with pytest.raises(ConfigError):
        bcfg.server_settings_from_env(cfg, {bcfg.ENV_SERVER_URL: URL, bcfg.ENV_MODEL_TIMEOUT: "-1"})


def test_build_query_service_with_server_runtime_end_to_end(tmp_path):
    cfg = bcfg.load_backend_config(env={})
    svc = build_query_service(cfg, env={bcfg.ENV_RUNTIME_KIND: "llama_server", bcfg.ENV_SERVER_URL: URL})
    assert isinstance(svc.runtime, LlamaServerRuntime)


def test_logs_carry_serving_metrics_but_no_prompt_or_url(caplog):
    with caplog.at_level(logging.INFO, logger="schemaforge.backend"):
        runtime_for(Server()).generate("SECRET QUESTION TEXT")
    text = "\n".join(r.getMessage() for r in caplog.records)
    ev = [json.loads(r.getMessage()) for r in caplog.records if '"runtime.generated"' in r.getMessage()]
    assert ev and ev[0]["mode"] == "persistent_server" and ev[0]["generated_tokens_per_second"] == 10.0
    assert {"queue_wait_ms", "generation_latency_ms", "input_tokens", "output_tokens"} <= set(ev[0])
    assert "SECRET" not in text and "model-server" not in text
