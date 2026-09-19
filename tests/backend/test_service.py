import json
import logging

import pytest

from localsql.backend.errors import DatabaseUnavailableError, UnknownDatabaseError
from localsql.backend.introspection import SQLiteSchemaIntrospector
from localsql.backend.models import QueryRequest
from localsql.data.prompt_builder import build_prompt
from localsql.data.schema_serializer import serialize_schema

from tests.backend.helpers import AllowAllSafety, FakeRuntime, failing_runtime, file_sha, make_registry, make_service


def req(question="How many employees are there?", ctx=None, db="demo"):
    return QueryRequest(database_id=db, question=question, business_context=ctx)


def test_success_path_end_to_end(tmp_path):
    service, runtime, spy, _ = make_service(tmp_path, FakeRuntime("  SELECT COUNT(*) AS n FROM employees;\n"))
    resp = service.query(req(), request_id="abc12345")
    assert resp.status == "ok" and resp.error is None and resp.request_id == "abc12345"
    assert resp.generated_sql == "SELECT COUNT(*) AS n FROM employees;"  # whitespace-trim only
    assert resp.safety.allowed and resp.result.columns == ["n"] and resp.result.rows == [[12]]
    assert resp.result.returned_row_count == 1 and resp.result.truncated is False
    assert spy.executed == ["SELECT COUNT(*) AS n FROM employees;"]
    t = resp.timings
    assert None not in (t.schema_ms, t.model_ms, t.safety_ms, t.execution_ms) and t.total_ms >= t.model_ms
    assert resp.model["runtime"] == "fake" and resp.model["input_tokens"] == 100 and resp.model["output_tokens"] == 7
    assert resp.dialect == "sqlite" and len(resp.prompt_sha256) == 64


def test_prompt_is_the_canonical_prompt_from_introspected_schema(tmp_path):
    service, runtime, _, registry = make_service(tmp_path)
    service.query(req("What is the top salary?", "Salary means annual gross pay."))
    schema = SQLiteSchemaIntrospector().introspect(registry.resolve("demo"))
    expected = build_prompt(serialize_schema(schema), "sqlite", "What is the top salary?", "Salary means annual gross pay.")
    assert runtime.prompts == [expected]  # byte-identical to the training/eval prompt contract
    assert "DIALECT:\nsqlite" in expected and "BUSINESS CONTEXT:\nSalary means annual gross pay." in runtime.prompts[0]
    assert "project_assignments(\n  emp_id INTEGER PK FK->employees.emp_id,\n  project_id INTEGER PK," in runtime.prompts[0]


def test_business_context_omitted_when_absent_or_blank(tmp_path):
    service, runtime, _, _ = make_service(tmp_path)
    service.query(req(ctx=None))
    service.query(req(ctx="   "))
    assert all("BUSINESS CONTEXT" not in p for p in runtime.prompts)
    service.query(req(ctx="hint"))
    assert "BUSINESS CONTEXT:\nhint\n\nQUESTION:" in runtime.prompts[-1]


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM employees",
        "DROP TABLE employees",
        "SELECT 1; DELETE FROM employees",
        "PRAGMA writable_schema=1",
        "ATTACH DATABASE 'x.db' AS x",
        "WITH a AS (DELETE FROM employees RETURNING *) SELECT * FROM a",
        "SELECT load_extension('x')",
        "```sql\nSELECT 1\n```",
        "",
    ],
)
def test_unsafe_sql_never_reaches_execution(tmp_path, sql):
    service, _, spy, registry = make_service(tmp_path, FakeRuntime(sql))
    path = registry.resolve("demo").path
    before = file_sha(path)
    resp = service.query(req())
    assert resp.status == "unsafe_sql" and resp.result is None
    assert resp.safety.allowed is False and resp.safety.reasons and resp.error.stage == "safety"
    assert spy.executed == []  # the executor was never called
    assert file_sha(path) == before


def test_db_cannot_be_mutated_even_if_ast_safety_is_bypassed(tmp_path):
    """Regression: with the safety layer fully bypassed, the executor alone must protect the data."""
    hostile = [
        "DELETE FROM employees",
        "UPDATE employees SET salary = 0",
        "DROP TABLE departments",
        "INSERT INTO departments VALUES (99, 'pwn')",
        "ATTACH DATABASE 'pwn.db' AS pwn",
        "PRAGMA writable_schema = 1",
        "CREATE TABLE pwn (a)",
    ]
    for sql in hostile:
        service, _, spy, registry = make_service(tmp_path, FakeRuntime(sql), safety=AllowAllSafety())
        path = registry.resolve("demo").path
        before = file_sha(path)
        resp = service.query(req())
        assert spy.executed == [sql]  # it DID reach the executor...
        assert resp.status == "execution_error" and resp.result is None  # ...which refused it
        assert resp.error.code in {"authorization_denied", "sql_error"}
        assert file_sha(path) == before, sql  # database bytes are identical
        assert not (path.parent / "pwn.db").exists()


def test_model_failures_are_structured(tmp_path):
    service, _, spy, _ = make_service(tmp_path, failing_runtime())
    resp = service.query(req())
    assert resp.status == "model_error" and resp.error.stage == "model" and resp.error.code == "model_error"
    assert resp.generated_sql is None and spy.executed == []

    service, _, spy, _ = make_service(tmp_path, FakeRuntime(RuntimeError("boom /secret/path")))
    resp = service.query(req())
    assert resp.status == "model_error" and resp.error.code == "internal_error"
    assert "secret" not in resp.model_dump_json() and spy.executed == []


def test_execution_error_is_structured(tmp_path):
    service, _, spy, _ = make_service(tmp_path, FakeRuntime("SELECT missing_col FROM employees"))
    resp = service.query(req())
    assert resp.status == "execution_error" and resp.error.stage == "execution" and resp.error.code == "sql_error"
    assert resp.safety.allowed and resp.generated_sql == "SELECT missing_col FROM employees"
    assert resp.timings.execution_ms is not None


def test_execution_timeout_reported(tmp_path):
    runaway = "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT COUNT(*) FROM c"
    service, *_ = make_service(tmp_path, FakeRuntime(runaway), timeout=0.3)
    resp = service.query(req())
    assert resp.status == "execution_error" and resp.error.code == "timeout"


def test_row_truncation_surfaces_in_response(tmp_path):
    service, *_ = make_service(tmp_path, FakeRuntime("SELECT emp_id FROM employees ORDER BY emp_id"), max_rows=5)
    resp = service.query(req())
    assert resp.status == "ok" and resp.result.returned_row_count == 5 and resp.result.truncated is True


def test_unknown_and_unavailable_database_raise_typed_errors(tmp_path):
    service, runtime, _, _ = make_service(tmp_path)
    with pytest.raises(UnknownDatabaseError):
        service.query(req(db="missing"))
    assert runtime.prompts == []  # the model is never consulted for an unknown database
    (tmp_path / "dbroot" / "demo.sqlite").unlink()
    with pytest.raises(DatabaseUnavailableError):
        service.query(req())


def test_schema_error_when_database_has_no_tables(tmp_path):
    import sqlite3

    service, runtime, _, registry = make_service(tmp_path)
    path = registry.resolve("demo").path
    path.unlink()
    sqlite3.connect(path).close()
    resp = service.query(req())
    assert resp.status == "schema_error" and resp.error.stage == "schema" and runtime.prompts == []


def test_request_validation_model():
    with pytest.raises(Exception):
        QueryRequest(database_id="demo", question="   ")
    with pytest.raises(Exception):
        QueryRequest(database_id="demo", question="q", path="/etc/passwd")  # no path field exists
    assert not any("path" in f for f in QueryRequest.model_fields)


def test_logging_is_structured_and_excludes_data(tmp_path, caplog):
    service, *_ = make_service(tmp_path, FakeRuntime("SELECT name FROM employees WHERE name = 'Ada'"))
    with caplog.at_level(logging.INFO, logger="schemaforge.backend"):
        service.query(req("SECRETQUESTION about Ada", "SECRETCONTEXT"), request_id="req-log-0001")
    text = "\n".join(r.getMessage() for r in caplog.records)
    event = json.loads(caplog.records[-1].getMessage())
    assert event["event"] == "query.completed" and event["request_id"] == "req-log-0001"
    assert event["database_id"] == "demo" and event["status"] == "ok" and "latency_ms" in event
    for leaked in ("SECRETQUESTION", "SECRETCONTEXT", "Ada", "SELECT name"):
        assert leaked not in text


def test_safety_rejection_and_failures_are_logged(tmp_path, caplog):
    service, *_ = make_service(tmp_path, FakeRuntime("DROP TABLE employees"))
    with caplog.at_level(logging.INFO, logger="schemaforge.backend"):
        service.query(req(), request_id="req-log-0002")
    event = json.loads(caplog.records[-1].getMessage())
    assert event["status"] == "unsafe_sql" and event["stage"] == "safety" and event["safety_codes"]
    with caplog.at_level(logging.INFO, logger="schemaforge.backend"):
        with pytest.raises(UnknownDatabaseError):
            service.query(req(db="nope"), request_id="req-log-0003")
    assert any(json.loads(r.getMessage()).get("error_code") == "unknown_database" for r in caplog.records)


def test_service_has_no_fastapi_dependency():
    import localsql.backend.service as svc

    imports = [ln for ln in open(svc.__file__, encoding="utf-8").read().splitlines() if ln.startswith(("import ", "from "))]
    assert not any("fastapi" in ln or "starlette" in ln for ln in imports)
