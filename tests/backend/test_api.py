import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from localsql.backend.api import create_app  # noqa: E402
from localsql.backend.errors import ModelRuntimeError  # noqa: E402
from localsql.backend.runtime import UnavailableRuntime  # noqa: E402
from localsql.backend.service import QueryService  # noqa: E402

from tests.backend.helpers import FakeRuntime, make_service  # noqa: E402


def client_for(tmp_path, runtime=None, **kw):
    service, runtime, spy, registry = make_service(tmp_path, runtime, **kw)
    return TestClient(create_app(service), raise_server_exceptions=False), runtime, spy


def body(**over):
    return {"database_id": "demo", "question": "How many employees are there?", **over}


def test_health_and_databases(tmp_path):
    c, *_ = client_for(tmp_path)
    r = c.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok", "model_runtime": {"runtime": "fake", "configured": True}}
    assert r.headers["X-Request-ID"]
    d = c.get("/databases").json()
    assert d == {"databases": [{"id": "demo", "dialect": "sqlite", "description": "demo db"}]}
    assert "dbroot" not in c.get("/databases").text


def test_query_success_and_request_id_propagation(tmp_path):
    c, runtime, spy = client_for(tmp_path, FakeRuntime("SELECT COUNT(*) FROM employees"))
    r = c.post("/query", json=body(business_context="ctx"), headers={"X-Request-ID": "client-req-123"})
    assert r.status_code == 200
    j = r.json()
    assert j["status"] == "ok" and j["result"]["rows"] == [[12]] and j["result"]["truncated"] is False
    assert j["request_id"] == "client-req-123" == r.headers["X-Request-ID"]
    assert j["generated_sql"] == "SELECT COUNT(*) FROM employees" and j["timings"]["total_ms"] > 0
    assert "BUSINESS CONTEXT:\nctx" in runtime.prompts[0]


def test_invalid_request_id_is_replaced(tmp_path):
    c, *_ = client_for(tmp_path)
    r = c.get("/health", headers={"X-Request-ID": "bad id with spaces\n"})
    assert r.headers["X-Request-ID"] != "bad id with spaces" and len(r.headers["X-Request-ID"]) == 32


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"database_id": "demo"},
        {"database_id": "demo", "question": ""},
        {"database_id": "demo", "question": "   "},
        {"database_id": "demo", "question": "x" * 2001},
        {"database_id": "demo", "question": "q", "business_context": "y" * 4001},
        {"database_id": "demo", "question": "q", "path": "/etc/passwd"},
        {"database_id": "demo", "question": "q", "db_path": "C:\\x.sqlite"},
        {"database_id": 5, "question": "q"},
    ],
)
def test_validation_errors_are_safe_and_never_echo_input(tmp_path, payload):
    c, runtime, _ = client_for(tmp_path)
    r = c.post("/query", json=payload)
    assert r.status_code == 422
    j = r.json()
    assert j["error"]["code"] == "invalid_request" and j["request_id"]
    assert "/etc/passwd" not in r.text and "C:\\\\x.sqlite" not in r.text and "xxxxxxxx" not in r.text
    assert runtime.prompts == []


def test_malformed_json_is_safe(tmp_path):
    c, *_ = client_for(tmp_path)
    r = c.post("/query", content=b"{not json", headers={"Content-Type": "application/json"})
    assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_request"


def test_unknown_database_404_and_path_ids_do_not_traverse(tmp_path):
    c, runtime, _ = client_for(tmp_path)
    for db in ["missing", "../../etc/passwd", "C:\\Windows\\win.ini"]:
        r = c.post("/query", json=body(database_id=db))
        assert r.status_code == 404 and r.json()["error"]["code"] == "unknown_database"
    assert runtime.prompts == []


def test_unavailable_database_503(tmp_path):
    c, *_ = client_for(tmp_path)
    (tmp_path / "dbroot" / "demo.sqlite").unlink()
    r = c.post("/query", json=body())
    assert r.status_code == 503 and r.json()["error"]["code"] == "database_unavailable"
    assert "dbroot" not in r.text


def test_unsafe_sql_is_422_with_structured_safety_and_no_execution(tmp_path):
    c, _, spy = client_for(tmp_path, FakeRuntime("DROP TABLE employees"))
    r = c.post("/query", json=body())
    j = r.json()
    assert r.status_code == 422 and j["status"] == "unsafe_sql" and j["safety"]["allowed"] is False
    assert j["safety"]["reasons"][0]["code"] and j["result"] is None and spy.executed == []


def test_model_error_502_and_not_configured_503(tmp_path):
    c, *_ = client_for(tmp_path, FakeRuntime(ModelRuntimeError("Model generation failed (timeout).")))
    r = c.post("/query", json=body())
    assert r.status_code == 502 and r.json()["error"]["code"] == "model_error"
    c2, *_ = client_for(tmp_path / "b", UnavailableRuntime("Missing environment variable(s): X"))
    r2 = c2.post("/query", json=body())
    assert r2.status_code == 503 and r2.json()["error"]["code"] == "model_not_configured"
    assert c2.get("/health").json()["model_runtime"]["configured"] is False  # health works without a model


def test_execution_error_and_timeout_status_codes(tmp_path):
    c, *_ = client_for(tmp_path, FakeRuntime("SELECT nope FROM employees"))
    assert c.post("/query", json=body()).status_code == 422
    runaway = "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT COUNT(*) FROM c"
    c2, *_ = client_for(tmp_path / "t", FakeRuntime(runaway), timeout=0.3)
    r = c2.post("/query", json=body())
    assert r.status_code == 504 and r.json()["error"]["code"] == "timeout"


def test_unhandled_exception_is_a_generic_500(tmp_path):
    service, *_ = make_service(tmp_path)

    def boom(*a, **k):
        raise RuntimeError("secret internal /var/db/path")

    service.query = boom  # type: ignore[method-assign]
    c = TestClient(create_app(service), raise_server_exceptions=False)
    r = c.post("/query", json=body())
    assert r.status_code == 500 and r.json()["error"] == {"code": "internal_error", "message": "Internal server error."}
    assert "secret" not in r.text


def test_dependency_injection_allows_swapping_the_service(tmp_path):
    from localsql.backend.api import get_query_service

    service, *_ = make_service(tmp_path)
    app = create_app(service)
    other, *_ = make_service(tmp_path / "o", FakeRuntime("SELECT 42"))
    app.dependency_overrides[get_query_service] = lambda: other
    r = TestClient(app).post("/query", json=body())
    assert r.json()["result"]["rows"] == [[42]]


def test_unknown_route_is_safe_404(tmp_path):
    c, *_ = client_for(tmp_path)
    r = c.get("/nope")
    assert r.status_code == 404 and r.json()["error"]["code"] == "not_found"


def test_isinstance_service(tmp_path):
    service, *_ = make_service(tmp_path)
    assert isinstance(service, QueryService)


def test_cors_allows_only_configured_origins_and_exposes_request_id(tmp_path):
    service, *_ = make_service(tmp_path)
    c = TestClient(create_app(service, cors_origins=["http://localhost:5173"]))
    ok = c.get("/health", headers={"Origin": "http://localhost:5173"})
    assert ok.headers["access-control-allow-origin"] == "http://localhost:5173"
    assert "x-request-id" in ok.headers["access-control-expose-headers"].lower()
    assert "access-control-allow-credentials" not in ok.headers
    evil = c.get("/health", headers={"Origin": "http://evil.example"})
    assert "access-control-allow-origin" not in evil.headers
    pre = c.options("/query", headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST",
                                       "Access-Control-Request-Headers": "content-type"})  # fmt: skip
    assert pre.status_code == 200 and "POST" in pre.headers["access-control-allow-methods"]
    err = c.post("/query", json={}, headers={"Origin": "http://localhost:5173"})  # error responses carry CORS too
    assert err.status_code == 422 and err.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_no_cors_headers_by_default(tmp_path):
    service, *_ = make_service(tmp_path)
    r = TestClient(create_app(service)).get("/health", headers={"Origin": "http://localhost:5173"})
    assert "access-control-allow-origin" not in r.headers
