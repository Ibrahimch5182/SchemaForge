"""Phase 10 reliability regression tests: preflight, malformed output, taxonomy,
cancellation, bounded concurrency, runtime failures, trust metadata, leakage.

No real model is loaded; deterministic layers use real temporary SQLite databases.
"""

import json
import logging
import sys
import threading
import time
from pathlib import Path

import pytest

from localsql.backend.control import CancelToken, InferenceGate
from localsql.backend.errors import (
    ModelBusyError,
    ModelRuntimeError,
    ModelTimeoutError,
    PreflightError,
    RequestCancelledError,
)
from localsql.backend.models import QueryRequest
from localsql.backend.preflight import SQLitePreflight, classify_sqlite_error
from localsql.backend.runtime import LlamaCppRuntime
from localsql.backend.taxonomy import ENVELOPE_HTTP, PIPELINE_CODES, STATUS_HTTP, all_codes, http_status
from localsql.deploy.process import run_captured
from localsql.deploy.runtime import GgufRun, LlamaSettings, run_gguf_once

from tests.backend.helpers import (
    AllowAllSafety,
    BlockingRuntime,
    FakeRuntime,
    file_sha,
    make_registry,
    make_service,
)

RUNAWAY = "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT COUNT(*) FROM c"


def req(question="How many employees are there?", db="demo"):
    return QueryRequest(database_id=db, question=question)


# ------------------------------------------------------------- preflight
@pytest.fixture
def demo(tmp_path):
    return make_registry(tmp_path).resolve("demo")


@pytest.mark.parametrize(
    "sql,code,detail",
    [
        ("SELECT * FROM nope", "unknown_table", "nope"),
        ("SELECT nope FROM employees", "unknown_column", "nope"),
        ("SELECT e.nope FROM employees e", "unknown_column", "e.nope"),
        ("SELECT dept_id FROM employees, departments", "ambiguous_column", "dept_id"),
        ("SELECT frobnicate(name) FROM employees", "unknown_function", "frobnicate"),
        ("SELECT FROM employees", "invalid_syntax", None),
        ("SELECT name FROM employees WHERE", "invalid_syntax", None),
    ],
)
def test_preflight_rejects_invalid_sql_with_structured_codes(demo, sql, code, detail):
    with pytest.raises(PreflightError) as e:
        SQLitePreflight().check(demo, sql)
    assert e.value.code == code and e.value.detail == detail
    assert "near" not in e.value.message and "sqlite3" not in e.value.message.lower() and str(demo.path.parent) not in e.value.message


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT COUNT(*) FROM employees",
        "SELECT e.name, d.name FROM employees e JOIN departments d ON e.dept_id = d.dept_id",
        "WITH t AS (SELECT * FROM employees) SELECT COUNT(*) FROM t",
        "SELECT 1; -- trailing comment",
    ],
)
def test_preflight_accepts_valid_sql(demo, sql):
    SQLitePreflight().check(demo, sql)  # no exception


def test_preflight_does_not_execute_or_mutate_and_refuses_writes(demo):
    before = file_sha(demo.path)
    for sql in ["DELETE FROM employees", "DROP TABLE employees", "INSERT INTO departments VALUES (9,'x')", "ATTACH DATABASE ':memory:' AS m", "PRAGMA writable_schema=1"]:
        with pytest.raises(PreflightError) as e:
            SQLitePreflight().check(demo, sql)
        assert e.value.code == "authorization_denied", sql
    SQLitePreflight().check(demo, RUNAWAY)  # compiled only: returns instantly instead of running forever
    assert file_sha(demo.path) == before


def test_preflight_honors_cancellation(demo):
    token = CancelToken()
    token.cancel()
    with pytest.raises((RequestCancelledError, PreflightError)):
        SQLitePreflight().check(demo, "SELECT * FROM nope", cancel=token)


@pytest.mark.parametrize(
    "message,code",
    [
        ("no such table: t", "unknown_table"),
        ("no such column: c", "unknown_column"),
        ("table t has no column named c", "unknown_column"),
        ("ambiguous column name: x", "ambiguous_column"),
        ("no such function: f", "unknown_function"),
        ('near "FROM": syntax error', "invalid_syntax"),
        ("incomplete input", "invalid_syntax"),
        ("not authorized", "authorization_denied"),
        ("attempt to write a readonly database", "authorization_denied"),
        ("something entirely different", "invalid_sql"),
    ],
)
def test_error_classification(message, code):
    assert classify_sqlite_error(message)[0] == code


def test_classification_never_returns_hostile_detail():
    assert classify_sqlite_error("no such table: " + "x" * 500)[1] is None
    assert classify_sqlite_error("no such table: a\nb; DROP")[1] is None


# ------------------------------------------------ service: trust + failures
def test_success_records_exactly_what_was_verified(tmp_path):
    service, *_ = make_service(tmp_path, FakeRuntime("SELECT COUNT(*) FROM employees"))
    r = service.query(req())
    assert r.status == "ok"
    assert (r.reliability.safety, r.reliability.preflight, r.reliability.execution) == ("passed", "passed", "passed")
    assert r.reliability.semantic_correctness == "not_verified"
    assert r.reliability.confidence is None
    assert "do not guarantee" in r.reliability.note
    dumped = r.model_dump(mode="json")
    assert dumped["reliability"]["confidence"] is None and dumped["timings"]["preflight_ms"] is not None
    assert "confidence" not in json.dumps({k: v for k, v in dumped.items() if k != "reliability"})  # no fake score anywhere else


def test_valid_sql_for_wrong_schema_is_a_validation_error_and_never_executes(tmp_path):
    service, _, spy, registry = make_service(tmp_path, FakeRuntime("SELECT COUNT(*) FROM staff"))
    before = file_sha(registry.resolve("demo").path)
    r = service.query(req())
    assert r.status == "validation_error" and r.result is None
    assert (r.error.stage, r.error.code, r.error.detail) == ("preflight", "unknown_table", "staff")
    assert r.generated_sql == "SELECT COUNT(*) FROM staff" and r.safety.allowed  # safety passed; validity did not
    assert (r.reliability.safety, r.reliability.preflight, r.reliability.execution) == ("passed", "failed", "not_run")
    assert spy.executed == [] and file_sha(registry.resolve("demo").path) == before


def test_unknown_column_and_function_errors_are_structured(tmp_path):
    for sql, code in [("SELECT bonus FROM employees", "unknown_column"), ("SELECT frobnicate(name) FROM employees", "unknown_function")]:
        service, _, spy, _ = make_service(tmp_path, FakeRuntime(sql))
        r = service.query(req())
        assert (r.status, r.error.code) == ("validation_error", code) and spy.executed == []


@pytest.mark.parametrize("raw", ["", "   \n\t ", ": SELECT COUNT(*) FROM employees", "SQL: SELECT 1", "SELEC nothing here", "SELECT 1\x00"])
def test_malformed_model_output_is_its_own_state(tmp_path, raw):
    service, _, spy, _ = make_service(tmp_path, FakeRuntime(raw))
    r = service.query(req())
    assert r.status == "model_error" and r.error.code == "malformed_model_output" and r.error.stage == "model"
    assert r.reliability.safety == "failed" and r.result is None and spy.executed == []


def test_genuinely_unsafe_sql_stays_unsafe_not_malformed(tmp_path):
    service, _, spy, _ = make_service(tmp_path, FakeRuntime("DROP TABLE employees"))
    r = service.query(req())
    assert (r.status, r.error.code) == ("unsafe_sql", "not_read_only_query") and spy.executed == []


def test_safety_bypass_is_still_stopped_by_preflight_and_executor(tmp_path):
    for sql in ["DELETE FROM employees", "DROP TABLE departments", "UPDATE employees SET salary = 0"]:
        service, _, spy, registry = make_service(tmp_path, FakeRuntime(sql), safety=AllowAllSafety())
        before = file_sha(registry.resolve("demo").path)
        r = service.query(req())
        assert (r.status, r.error.code) == ("validation_error", "authorization_denied") and spy.executed == []
        assert file_sha(registry.resolve("demo").path) == before


def test_execution_timeout_is_stable(tmp_path):
    service, *_ = make_service(tmp_path, FakeRuntime(RUNAWAY), timeout=0.3)
    r = service.query(req())
    assert (r.status, r.error.code, r.error.stage) == ("execution_error", "timeout", "execution")
    assert (r.reliability.preflight, r.reliability.execution) == ("passed", "failed")


@pytest.mark.parametrize(
    "exc,status,code",
    [
        (ModelBusyError("The model is busy. Try again shortly."), "model_error", "model_busy"),
        (ModelTimeoutError("Model generation timed out."), "model_error", "model_timeout"),
        (ModelRuntimeError("Model generation failed."), "model_error", "model_error"),
        (RuntimeError("boom /secret/path Traceback"), "model_error", "internal_error"),
    ],
)
def test_model_failures_map_to_stable_codes_without_leaking(tmp_path, exc, status, code):
    service, _, spy, _ = make_service(tmp_path, FakeRuntime(exc))
    r = service.query(req())
    assert (r.status, r.error.code, r.error.stage) == (status, code, "model") and spy.executed == []
    assert "secret" not in r.model_dump_json() and "Traceback" not in r.model_dump_json()


def test_no_path_or_internals_in_any_failure_response(tmp_path):
    cases = [FakeRuntime("SELECT * FROM nope"), FakeRuntime("DROP TABLE x"), FakeRuntime(""), FakeRuntime(RUNAWAY)]
    for rt in cases:
        service, *_ = make_service(tmp_path, rt, timeout=0.3)
        text = service.query(req()).model_dump_json()
        for leak in (str(tmp_path), "Traceback", "sqlite3.", ".py\"", "dbroot"):
            assert leak not in text, (leak, text[:200])


# ---------------------------------------------------------- cancellation
def _run_async(service, request, rid):
    out = {}
    t = threading.Thread(target=lambda: out.setdefault("r", service.query(request, request_id=rid)))
    t.start()
    return t, out


def test_cancelling_during_generation_stops_the_request_promptly(tmp_path):
    rt = BlockingRuntime()
    service, _, spy, _ = make_service(tmp_path, rt)
    t, out = _run_async(service, req(), "cancel-me-0001")
    assert rt.started.wait(5)
    assert service.cancel("cancel-me-0001") is True
    t.join(5)
    assert not t.is_alive()
    r = out["r"]
    assert (r.status, r.error.code, r.error.stage) == ("cancelled", "cancelled", "model")
    assert spy.executed == []
    assert service.cancel("cancel-me-0001") is False  # finished requests are unregistered


def test_cancelling_a_running_sql_statement_interrupts_it(tmp_path):
    service, _, spy, _ = make_service(tmp_path, FakeRuntime(RUNAWAY), timeout=30)
    t, out = _run_async(service, req(), "cancel-sql-0001")
    deadline = time.monotonic() + 5
    while not spy.executed and time.monotonic() < deadline:
        time.sleep(0.01)
    time.sleep(0.2)
    start = time.monotonic()
    assert service.cancel("cancel-sql-0001")
    t.join(5)
    assert not t.is_alive() and time.monotonic() - start < 3  # not the 30 s execution limit
    assert out["r"].status == "cancelled" and out["r"].error.stage == "execution"


def test_pre_cancelled_request_does_no_expensive_work(tmp_path):
    rt = FakeRuntime()
    service, _, spy, _ = make_service(tmp_path, rt)
    token = CancelToken()
    token.cancel()
    r = service.query(req(), cancel=token)
    assert r.status == "cancelled" and rt.prompts == [] and spy.executed == []


def test_cancel_unknown_request_is_false(tmp_path):
    service, *_ = make_service(tmp_path)
    assert service.cancel("does-not-exist") is False


# ------------------------------------------------------ bounded concurrency
def test_gate_limits_running_and_waiting_and_rejects_the_rest():
    gate = InferenceGate(max_concurrent=1, max_waiting=2, wait_timeout_s=5)
    release = threading.Event()
    results: list[str] = []

    def worker(name):
        try:
            with gate.slot():
                results.append(f"start-{name}")
                release.wait(5)
        except ModelBusyError:
            results.append(f"busy-{name}")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(3)]
    threads[0].start()
    while gate.snapshot()["running"] < 1:
        time.sleep(0.005)
    threads[1].start()
    threads[2].start()
    while gate.snapshot()["waiting"] < 2:
        time.sleep(0.005)
    assert gate.snapshot()["state"] == "saturated"
    with pytest.raises(ModelBusyError):  # 4th: 1 running + 2 waiting already
        with gate.slot():
            pass
    release.set()
    for t in threads:
        t.join(5)
    assert sorted(results) == ["start-0", "start-1", "start-2"]
    assert gate.snapshot() == {"state": "idle", "running": 0, "waiting": 0, "max_concurrent": 1, "max_waiting": 2}


def test_gate_never_exceeds_max_concurrent():
    gate = InferenceGate(max_concurrent=2, max_waiting=20, wait_timeout_s=10)
    lock, active, peak = threading.Lock(), [0], [0]

    def worker():
        with gate.slot():
            with lock:
                active[0] += 1
                peak[0] = max(peak[0], active[0])
            time.sleep(0.02)
            with lock:
                active[0] -= 1

    ts = [threading.Thread(target=worker) for _ in range(12)]
    [t.start() for t in ts]
    [t.join(10) for t in ts]
    assert peak[0] == 2 and gate.snapshot()["running"] == 0


def test_gate_wait_timeout_becomes_busy_and_cancel_wins_while_waiting():
    gate = InferenceGate(max_concurrent=1, max_waiting=2, wait_timeout_s=0.15)
    holder = threading.Event()
    t = threading.Thread(target=lambda: (gate.slot().__enter__(), holder.wait(5)))
    ctx = gate.slot()
    ctx.__enter__()
    start = time.monotonic()
    with pytest.raises(ModelBusyError):
        with gate.slot():
            pass
    assert 0.1 < time.monotonic() - start < 2
    token = CancelToken()
    threading.Timer(0.05, token.cancel).start()
    gate2 = InferenceGate(max_concurrent=1, max_waiting=1, wait_timeout_s=5)
    with gate2.slot():
        with pytest.raises(RequestCancelledError):
            with gate2.slot(token):
                pass
    ctx.__exit__(None, None, None)
    assert gate.snapshot()["running"] == 0 and t is not None


def test_gate_releases_the_slot_when_the_body_raises():
    gate = InferenceGate(1, 0, 0)
    with pytest.raises(RuntimeError):
        with gate.slot():
            raise RuntimeError("boom")
    with gate.slot():  # slot free again
        pass
    assert gate.snapshot()["state"] == "idle"


def test_gate_rejects_bad_limits():
    with pytest.raises(ValueError):
        InferenceGate(0, 1, 1)


# --------------------------------------------------- llama.cpp runtime
def _settings(tmp_path, exe=None):
    files = {}
    for n in ("llama.exe", "base.gguf", "lora.gguf"):
        p = tmp_path / n
        p.write_bytes(b"x")
        files[n] = p
    return LlamaSettings(executable=exe or files["llama.exe"], model=files["base.gguf"], lora=files["lora.gguf"],
                         context_size=512, max_new_tokens=8, seed=1, timeout_seconds=20), files  # fmt: skip


def _gguf(status="ok", raw="SELECT 1", error=None, returncode=0):
    return GgufRun(status, raw if status == "ok" else None, None, 5.0, {}, None, "n/a", False, ["cmd"], returncode, error)


def test_runtime_maps_abnormal_exit_timeout_and_cancel(tmp_path, caplog):
    s, _ = _settings(tmp_path)
    with caplog.at_level(logging.ERROR, logger="schemaforge.backend"):
        rt = LlamaCppRuntime(s, run=lambda p, st, **k: _gguf("error", error=f"exit code 3221225477: {tmp_path}\\secret crash", returncode=3221225477))
        with pytest.raises(ModelRuntimeError) as e:
            rt.generate("p")
    assert e.value.code == "model_error" and str(tmp_path) not in e.value.message and "crash" not in e.value.message
    assert any('"returncode": 3221225477' in r.getMessage() for r in caplog.records)  # detail lives in the log

    rt = LlamaCppRuntime(s, run=lambda p, st, **k: _gguf("timeout"))
    with pytest.raises(ModelTimeoutError):
        rt.generate("p")
    rt = LlamaCppRuntime(s, run=lambda p, st, **k: _gguf("cancelled"))
    with pytest.raises(RequestCancelledError):
        rt.generate("p")


def test_runtime_passes_empty_output_through_for_the_service_to_classify(tmp_path):
    s, _ = _settings(tmp_path)
    rt = LlamaCppRuntime(s, run=lambda p, st, **k: _gguf("ok", raw=""))
    assert rt.generate("p").raw_completion == ""


def test_runtime_bounds_concurrent_model_processes(tmp_path):
    s, _ = _settings(tmp_path)
    live, peak, lock = [0], [0], threading.Lock()
    release = threading.Event()

    def slow_run(prompt, settings, **kw):
        with lock:
            live[0] += 1
            peak[0] = max(peak[0], live[0])
        release.wait(5)
        with lock:
            live[0] -= 1
        return _gguf()

    rt = LlamaCppRuntime(s, run=slow_run, gate=InferenceGate(1, 1, 5))
    outcomes: list[str] = []

    def call():
        try:
            rt.generate("p")
            outcomes.append("ok")
        except ModelBusyError:
            outcomes.append("busy")

    first = threading.Thread(target=call)
    first.start()
    while rt.gate.snapshot()["running"] < 1:
        time.sleep(0.005)
    second = threading.Thread(target=call)
    second.start()
    while rt.gate.snapshot()["waiting"] < 1:
        time.sleep(0.005)
    third = threading.Thread(target=call)  # queue (1) is full -> immediate busy, no third process
    third.start()
    third.join(5)
    assert outcomes == ["busy"]
    release.set()
    first.join(5)
    second.join(5)
    assert sorted(outcomes) == ["busy", "ok", "ok"] and peak[0] == 1


def test_runtime_readiness_is_cheap_path_free_and_tracks_artifacts(tmp_path):
    s, files = _settings(tmp_path)
    rt = LlamaCppRuntime(s, run=lambda *a, **k: _gguf())
    r = rt.readiness()
    assert r["ready"] is True and r["checks"] == {"executable": True, "base_gguf": True, "lora_gguf": True}
    assert r["availability"]["state"] == "idle" and str(tmp_path) not in json.dumps(r)
    files["lora.gguf"].unlink()
    r2 = rt.readiness()
    assert r2["ready"] is False and r2["checks"]["lora_gguf"] is False


# ------------------------------------ real subprocess: kill + cleanup paths
SLEEPER = [sys.executable, "-c", "import time; time.sleep(60)"]


def test_run_captured_kills_the_process_on_cancel():
    flag = threading.Event()
    threading.Timer(0.3, flag.set).start()
    start = time.monotonic()
    res = run_captured(SLEEPER, timeout=30, should_cancel=flag.is_set)
    assert res.cancelled and res.returncode is None and not res.timed_out
    assert time.monotonic() - start < 10


def test_run_captured_kills_the_process_on_timeout():
    start = time.monotonic()
    res = run_captured(SLEEPER, timeout=0.5, should_cancel=lambda: False)
    assert res.timed_out and not res.cancelled and time.monotonic() - start < 10


def test_abnormal_exit_of_a_real_process_and_temp_cleanup(tmp_path):
    settings, _ = _settings(tmp_path, exe=Path(sys.executable))  # python rejects llama.cpp's flags -> nonzero exit
    work = tmp_path / "work"
    work.mkdir()
    run = run_gguf_once("SELECT 1", settings, work_dir=work)
    assert run.status == "error" and run.returncode not in (0, None)
    assert list(work.iterdir()) == []  # prompt temp dir removed after failure
    rt = LlamaCppRuntime(settings, run=run_gguf_once)
    with pytest.raises(ModelRuntimeError) as e:
        rt.generate("p")
    assert e.value.code == "model_error" and str(tmp_path) not in e.value.message


# ------------------------------------------------------------ taxonomy
def test_taxonomy_is_total_and_deterministic():
    for status, codes in PIPELINE_CODES.items():
        assert status in STATUS_HTTP
        for c in codes:
            assert isinstance(http_status(status, c), int)
    assert http_status("ok", None) == 200
    assert http_status("model_error", "model_busy") == 429
    assert http_status("model_error", "model_timeout") == 504
    assert http_status("model_error", "model_not_configured") == 503
    assert http_status("model_error", "malformed_model_output") == 502
    assert http_status("execution_error", "timeout") == 504
    assert http_status("execution_error", "sql_error") == 422
    assert http_status("validation_error", "unknown_table") == 422
    assert http_status("unsafe_sql", "not_read_only_query") == 422
    assert http_status("cancelled", "cancelled") == 409
    assert ENVELOPE_HTTP["unknown_database"] == 404


def test_every_safety_policy_code_is_in_the_taxonomy():
    import localsql.backend.safety as safety

    codes = set(safety._DENIED_NODES) | {"denied_function", "not_read_only_query", "multiple_statements"}
    assert codes <= set(PIPELINE_CODES["unsafe_sql"])


def test_every_preflight_code_is_in_the_taxonomy():
    from localsql.backend.preflight import _MESSAGES

    assert set(_MESSAGES) <= set(PIPELINE_CODES["validation_error"])


def test_taxonomy_is_documented():
    doc = (Path(__file__).resolve().parents[2] / "docs" / "PHASE10.md").read_text(encoding="utf-8")
    missing = [c for c in sorted(all_codes()) if f"`{c}`" not in doc]
    assert not missing, f"docs/PHASE10.md does not document: {missing}"


# ------------------------------------------------------------ observability
def _events(caplog):
    out = []
    for r in caplog.records:
        try:
            out.append(json.loads(r.getMessage()))
        except ValueError:
            pass
    return out


@pytest.mark.parametrize(
    "runtime,event",
    [
        (FakeRuntime(ModelBusyError("b")), "query.model_busy"),
        (FakeRuntime(ModelTimeoutError("t")), "query.model_timeout"),
        (FakeRuntime(ModelRuntimeError("e")), "query.model_failure"),
        (FakeRuntime(""), "query.malformed_output"),
        (FakeRuntime("DROP TABLE employees"), "query.safety_rejected"),
        (FakeRuntime("SELECT * FROM nope"), "query.preflight_rejected"),
    ],
)
def test_specific_events_are_logged_with_request_id_and_no_sensitive_data(tmp_path, caplog, runtime, event):
    service, *_ = make_service(tmp_path, runtime)
    with caplog.at_level(logging.INFO, logger="schemaforge.backend"):
        service.query(QueryRequest(database_id="demo", question="SECRETQUESTION about payroll"), request_id="obs-req-000001")
    evs = _events(caplog)
    hit = [e for e in evs if e["event"] == event]
    assert hit and hit[0]["request_id"] == "obs-req-000001" and hit[0]["database_id"] == "demo"
    done = [e for e in evs if e["event"] == "query.completed"][0]
    assert {"schema_ms", "model_ms", "safety_ms", "preflight_ms", "execution_ms", "total_ms"} <= set(done["timings"])
    text = "\n".join(r.getMessage() for r in caplog.records)
    for leaked in ("SECRETQUESTION", "DROP TABLE", "SELECT * FROM nope", "PROMPT", "SYSTEM:"):
        assert leaked not in text


def test_execution_timeout_and_cancellation_events(tmp_path, caplog):
    service, *_ = make_service(tmp_path, FakeRuntime(RUNAWAY), timeout=0.3)
    with caplog.at_level(logging.INFO, logger="schemaforge.backend"):
        service.query(req(), request_id="obs-req-000002")
        token = CancelToken()
        token.cancel()
        service.query(req(), request_id="obs-req-000003", cancel=token)
    names = {e["event"] for e in _events(caplog)}
    assert {"query.execution_timeout", "query.cancelled"} <= names
