"""Phase 10 API behavior: status mapping, readiness, cancel endpoint, no leakage."""

import json
import threading

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from localsql.backend.api import create_app  # noqa: E402
from localsql.backend.errors import ModelBusyError, ModelRuntimeError, ModelTimeoutError  # noqa: E402
from localsql.backend.runtime import LlamaCppRuntime, UnavailableRuntime  # noqa: E402
from localsql.deploy.runtime import GgufRun, LlamaSettings  # noqa: E402

from tests.backend.helpers import BlockingRuntime, FakeRuntime, make_service  # noqa: E402

RUNAWAY = "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT COUNT(*) FROM c"
BODY = {"database_id": "demo", "question": "How many employees are there?"}


def client_for(tmp_path, runtime=None, **kw):
    service, runtime, spy, _ = make_service(tmp_path, runtime, **kw)
    return TestClient(create_app(service), raise_server_exceptions=False), runtime, service


@pytest.mark.parametrize(
    "runtime,status,body_status,code,stage",
    [
        (FakeRuntime(ModelBusyError("The model is busy. Try again shortly.")), 429, "model_error", "model_busy", "model"),
        (FakeRuntime(ModelTimeoutError("Model generation timed out.")), 504, "model_error", "model_timeout", "model"),
        (FakeRuntime(ModelRuntimeError("Model generation failed.")), 502, "model_error", "model_error", "model"),
        (FakeRuntime(""), 502, "model_error", "malformed_model_output", "model"),
        (FakeRuntime(": SELECT 1"), 502, "model_error", "malformed_model_output", "model"),
        (UnavailableRuntime("Missing environment variable(s): X"), 503, "model_error", "model_not_configured", "model"),
        (FakeRuntime("DROP TABLE employees"), 422, "unsafe_sql", "not_read_only_query", "safety"),
        (FakeRuntime("SELECT * FROM nope"), 422, "validation_error", "unknown_table", "preflight"),
        (FakeRuntime("SELECT nope FROM employees"), 422, "validation_error", "unknown_column", "preflight"),
        (FakeRuntime(RUNAWAY), 504, "execution_error", "timeout", "execution"),
    ],
)
def test_status_mapping_is_stable_and_consistent(tmp_path, runtime, status, body_status, code, stage):
    c, *_ = client_for(tmp_path, runtime, timeout=0.3)
    r = c.post("/query", json=BODY)
    j = r.json()
    assert r.status_code == status
    assert (j["status"], j["error"]["code"], j["error"]["stage"]) == (body_status, code, stage)
    assert j["reliability"]["semantic_correctness"] == "not_verified" and j["reliability"]["confidence"] is None
    text = r.text
    for leak in (str(tmp_path), "Traceback", "sqlite3", "dbroot"):
        assert leak not in text


def test_model_busy_response_carries_retry_after(tmp_path):
    c, *_ = client_for(tmp_path, FakeRuntime(ModelBusyError("The model is busy. Try again shortly.")))
    r = c.post("/query", json=BODY)
    assert r.status_code == 429 and r.headers["Retry-After"] == "5"


def test_successful_flow_is_unchanged_plus_additive_metadata(tmp_path):
    c, *_ = client_for(tmp_path, FakeRuntime("SELECT COUNT(*) FROM employees"))
    r = c.post("/query", json=BODY)
    j = r.json()
    assert r.status_code == 200 and j["status"] == "ok" and j["error"] is None
    assert j["result"]["rows"] == [[12]] and j["generated_sql"] == "SELECT COUNT(*) FROM employees"
    assert j["reliability"] == {
        "safety": "passed", "preflight": "passed", "execution": "passed",
        "semantic_correctness": "not_verified", "confidence": None, "note": j["reliability"]["note"],
    }  # fmt: skip
    assert {"schema_ms", "model_ms", "safety_ms", "preflight_ms", "execution_ms", "total_ms"} <= set(j["timings"])


# ------------------------------------------------------ health / readiness
def _llama(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    files = {}
    for n in ("llama.exe", "base.gguf", "lora.gguf"):
        p = tmp_path / n
        p.write_bytes(b"x")
        files[n] = p
    s = LlamaSettings(files["llama.exe"], files["base.gguf"], 512, 8, 1, lora=files["lora.gguf"])
    run = lambda *a, **k: GgufRun("ok", "SELECT 1", None, 1.0, {}, None, "n/a", False, [], 0, None)  # noqa: E731
    return LlamaCppRuntime(s, run=run), files


def test_health_reports_readiness_without_paths_or_inference(tmp_path):
    rt, files = _llama(tmp_path / "m")
    calls = []
    rt._run = lambda *a, **k: calls.append(1)  # any inference would be recorded
    c, *_ = client_for(tmp_path, rt)
    j = c.get("/health").json()
    m = j["model_runtime"]
    assert j["status"] == "ok" and m["configured"] is True and m["ready"] is True
    assert m["checks"] == {"executable": True, "base_gguf": True, "lora_gguf": True}
    assert m["availability"]["state"] == "idle" and m["availability"]["max_concurrent"] == 1
    assert str(tmp_path) not in json.dumps(j) and calls == []
    files["base.gguf"].unlink()
    m2 = c.get("/health").json()["model_runtime"]
    assert m2["ready"] is False and m2["checks"]["base_gguf"] is False


def test_ready_endpoint(tmp_path):
    rt, files = _llama(tmp_path / "m")
    c, *_ = client_for(tmp_path, rt)
    ok = c.get("/ready")
    assert ok.status_code == 200 and ok.json()["ready"] is True and ok.json()["reasons"] == []
    files["lora.gguf"].unlink()
    bad = c.get("/ready")
    assert bad.status_code == 503 and bad.json()["reasons"] == ["model_artifacts_missing"]
    c2, *_ = client_for(tmp_path / "x", UnavailableRuntime("nope"))
    r = c2.get("/ready")
    assert r.status_code == 503 and r.json()["reasons"] == ["model_not_configured"]
    assert c2.get("/health").status_code == 200  # liveness independent of readiness


def test_health_without_readiness_support_is_unchanged(tmp_path):
    c, *_ = client_for(tmp_path)
    assert c.get("/health").json() == {"status": "ok", "model_runtime": {"runtime": "fake", "configured": True}}


# ------------------------------------------------------------ cancellation
def test_cancel_endpoint_stops_an_inflight_query(tmp_path):
    rt = BlockingRuntime()
    c, _, service = client_for(tmp_path, rt)
    out = {}
    t = threading.Thread(target=lambda: out.setdefault("r", c.post("/query", json=BODY, headers={"X-Request-ID": "abort-target-01"})))
    t.start()
    assert rt.started.wait(5)
    cancel = c.post("/query/abort-target-01/cancel")
    assert cancel.status_code == 200 and cancel.json() == {"request_id": "abort-target-01", "cancelled": True}
    t.join(5)
    assert not t.is_alive()
    r = out["r"]
    assert r.status_code == 409 and r.json()["status"] == "cancelled" and r.json()["error"]["code"] == "cancelled"
    again = c.post("/query/abort-target-01/cancel")
    assert again.json()["cancelled"] is False  # idempotent, and finished requests are forgotten


@pytest.mark.parametrize("rid", ["unknown-request-1", "bad id", "x", "a" * 200])
def test_cancel_unknown_or_invalid_ids_are_harmless(tmp_path, rid):
    c, *_ = client_for(tmp_path)
    r = c.post(f"/query/{rid}/cancel")
    assert r.status_code in (200, 404) and (r.status_code == 404 or r.json()["cancelled"] is False)
