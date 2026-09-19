"""PRODUCT RELIABILITY CANARY -- NOT A MODEL BENCHMARK.

One consolidated validation of the Phase 10 hardening against the REAL product
stack (real registry, introspection, safety, preflight, read-only executor,
llama.cpp runtime). It is NOT an accuracy or generalization measurement: it
uses a tiny synthetic database and a handful of generations, and says nothing
about how often the model is right.

Part A (no model): scripted runtimes drive every failure mode through the real
    pipeline -- malformed output, unsafe SQL, invalid-for-database SQL,
    execution timeout, model busy / timeout, cancellation, bounded concurrency,
    a real crashing subprocess -- and check the stable status/code taxonomy,
    that the database file never changes, and that nothing leaks paths.
Part B (real model, 3 generations): representative queries run through the real
    Q4_K_M + LoRA runtime; the pipeline must succeed with honest trust metadata,
    and two easy answers must match their known results. (The third result
    match is reported but not gating: this is a reliability gate, not a
    quality gate.)

Requires SCHEMAFORGE_LLAMA_EXE / _BASE_GGUF / _LORA_GGUF. Report:
.artifacts/phase10/canary_report.json. Exit 0 only if the gate passes.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from localsql.backend.bootstrap import build_query_service  # noqa: E402
from localsql.backend.config import load_backend_config  # noqa: E402
from localsql.backend.control import CancelToken, InferenceGate  # noqa: E402
from localsql.backend.demo import create_demo_database  # noqa: E402
from localsql.backend.errors import ModelBusyError, ModelRuntimeError, ModelTimeoutError, RequestCancelledError  # noqa: E402
from localsql.backend.models import QueryRequest  # noqa: E402
from localsql.backend.runtime import LlamaCppRuntime, ModelGeneration, UnavailableRuntime  # noqa: E402
from localsql.deploy.runtime import LlamaSettings, run_gguf_once  # noqa: E402

BANNER = "PRODUCT RELIABILITY CANARY -- NOT A MODEL BENCHMARK"
ARTIFACT_DIR = REPO_ROOT / ".artifacts" / "phase10"
RUNAWAY = "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT COUNT(*) FROM c"

checks: list[dict] = []


def check(name: str, ok: bool, detail: str = "", gating: bool = True) -> bool:
    checks.append({"check": name, "ok": bool(ok), "gating": gating, "detail": detail})
    tag = "PASS" if ok else ("FAIL" if gating else "NOTE")
    print(f"  [{tag}] {name}" + (f" -- {detail}" if detail else ""))
    return bool(ok)


class ScriptedRuntime:
    """Deterministic stand-in used ONLY in part A; never in part B."""

    def __init__(self, output: Any = "SELECT 1", block: bool = False):
        self.output, self.block = output, block
        self.started = threading.Event()

    def generate(self, prompt: str, cancel: Optional[CancelToken] = None) -> ModelGeneration:
        self.started.set()
        while self.block:
            if cancel is not None and cancel.cancelled:
                raise RequestCancelledError("cancelled")
            time.sleep(0.01)
        if isinstance(self.output, Exception):
            raise self.output
        return ModelGeneration(raw_completion=self.output, latency_ms=1.0)

    def describe(self) -> dict:
        return {"runtime": "scripted", "configured": True}


def db_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def norm_rows(rows: list[list[Any]]) -> list[tuple]:
    def n(v: Any) -> str:
        return f"{float(v):.6f}" if isinstance(v, (int, float)) and not isinstance(v, bool) else str(v)

    return sorted(tuple(sorted(n(v) for v in row)) for row in rows)


def part_a(cfg, db_path: Path) -> None:
    print("Part A -- failure modes through the real pipeline (no model):")
    ask = lambda svc, q="How many employees are there?": svc.query(QueryRequest(database_id="demo", question=q))  # noqa: E731
    before = db_sha(db_path)
    leak_probes = (str(REPO_ROOT), str(db_path.parent), "Traceback", "sqlite3.")

    def clean(resp) -> bool:
        text = resp.model_dump_json()
        return not any(p in text for p in leak_probes)

    def svc_for(output: Any, config=cfg):
        return build_query_service(config, runtime=ScriptedRuntime(output), env={})

    r = ask(svc_for("DROP TABLE employees"))
    check("unsafe SQL -> unsafe_sql, never executed", r.status == "unsafe_sql" and r.result is None and r.reliability.execution == "not_run", f"{r.status}/{r.error.code if r.error else None}")
    r = ask(svc_for("SELECT * FROM staff"))
    check("unknown table -> validation_error/unknown_table", (r.status, r.error.code if r.error else None) == ("validation_error", "unknown_table") and r.result is None, str(r.error.detail if r.error else None))
    r = ask(svc_for("SELECT bonus FROM employees"))
    check("unknown column -> validation_error/unknown_column", (r.status, r.error.code if r.error else None) == ("validation_error", "unknown_column"))
    for label, raw in (("empty", ""), ("label-prefixed", ": SELECT COUNT(*) FROM employees")):
        r = ask(svc_for(raw))
        check(f"{label} model output -> malformed_model_output", (r.status, r.error.code if r.error else None) == ("model_error", "malformed_model_output") and clean(r))
    for exc, code in ((ModelBusyError("busy"), "model_busy"), (ModelTimeoutError("t"), "model_timeout"), (ModelRuntimeError("e"), "model_error")):
        r = ask(svc_for(exc))
        check(f"runtime {code} surfaces as {code}", r.error is not None and r.error.code == code and clean(r))
    fast = cfg.model_copy(update={"execution": cfg.execution.model_copy(update={"timeout_seconds": 0.5})})
    r = ask(svc_for(RUNAWAY, fast))
    check("runaway query -> execution timeout, bounded", (r.status, r.error.code if r.error else None) == ("execution_error", "timeout") and r.timings.total_ms < 5000, f"{r.timings.total_ms:.0f} ms")

    # cancellation of an in-flight generation
    rt = ScriptedRuntime(block=True)
    svc = build_query_service(cfg, runtime=rt, env={})
    out: dict = {}
    t = threading.Thread(target=lambda: out.setdefault("r", svc.query(QueryRequest(database_id="demo", question="q"), request_id="canary-cancel-0001")))
    t.start()
    rt.started.wait(5)
    t0 = time.monotonic()
    svc.cancel("canary-cancel-0001")
    t.join(5)
    check("cancel stops an in-flight request quickly", not t.is_alive() and out["r"].status == "cancelled" and time.monotonic() - t0 < 2, f"{(time.monotonic() - t0) * 1000:.0f} ms")

    # bounded concurrency: 1 running + 2 waiting, the 4th is rejected immediately
    gate = InferenceGate(1, 2, 5)
    release, rejected = threading.Event(), []

    def hold():
        with gate.slot():
            release.wait(5)

    holders = [threading.Thread(target=hold) for _ in range(3)]
    holders[0].start()
    while gate.snapshot()["running"] < 1:
        time.sleep(0.005)
    for h in holders[1:]:
        h.start()
    while gate.snapshot()["waiting"] < 2:
        time.sleep(0.005)
    t0 = time.monotonic()
    try:
        with gate.slot():
            pass
    except ModelBusyError:
        rejected.append(time.monotonic() - t0)
    saturated = gate.snapshot()["state"] == "saturated"
    release.set()
    [h.join(5) for h in holders]
    check("inference gate is bounded (4th request rejected immediately)", bool(rejected) and rejected[0] < 0.5 and saturated and gate.snapshot()["running"] == 0)

    # a REAL subprocess that crashes (python rejects llama.cpp's flags) -> generic model_error, no leak, temp cleaned
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        for n in ("a.gguf", "b.gguf"):
            (tdp / n).write_bytes(b"x")
        settings = LlamaSettings(Path(sys.executable), tdp / "a.gguf", 512, 8, 1, lora=tdp / "b.gguf", timeout_seconds=30)
        svc = build_query_service(cfg, runtime=LlamaCppRuntime(settings, run=run_gguf_once), env={})
        r = ask(svc)
        check("crashing model process -> model_error without leaking internals", (r.status, r.error.code if r.error else None) == ("model_error", "model_error") and clean(r) and td not in r.model_dump_json())

    check("database file unchanged by all failure-mode checks", db_sha(db_path) == before)


def part_b(cfg, service, db_path: Path) -> list[dict]:
    print("Part B -- real Q4_K_M + LoRA runtime (3 generations, reliability gate):")
    rt = service.runtime
    ready = rt.readiness() if hasattr(rt, "readiness") else {}
    check("runtime readiness (no inference)", bool(ready.get("ready")) and ready["availability"]["state"] == "idle", json.dumps({k: v for k, v in ready.items() if k != "availability"}))

    questions = [
        ("count", "How many employees are there?", None, [[12]], True),
        ("join_with_context", "What is the total salary of employees in the Engineering department?", "Engineering is a department name stored in departments.name.", [[625000.0]], True),
        ("group_by", "How many employees are in each department? Show the department name and the number of employees.", None, [["Engineering", 5], ["Sales", 4], ["Support", 3]], False),
    ]
    before = db_sha(db_path)
    outcomes = []
    for qid, question, ctx, expected, gating_match in questions:
        resp = service.query(QueryRequest(database_id="demo", question=question, business_context=ctx))
        ran = resp.status == "ok" and resp.result is not None
        check(f"{qid}: pipeline succeeded", ran, f"status={resp.status}" + (f" code={resp.error.code}" if resp.error else "") + f" total={resp.timings.total_ms:.0f}ms")
        match = bool(ran and norm_rows(resp.result.rows) == norm_rows(expected))
        check(f"{qid}: result matches the known answer", match, str(resp.result.rows if resp.result else None), gating=gating_match)
        trust = resp.reliability
        check(
            f"{qid}: trust metadata is honest",
            trust.semantic_correctness == "not_verified" and trust.confidence is None and (not ran or (trust.safety, trust.preflight, trust.execution) == ("passed", "passed", "passed")),
        )
        outcomes.append({
            "id": qid, "status": resp.status, "error_code": resp.error.code if resp.error else None,
            "generated_sql": resp.generated_sql, "rows": resp.result.rows if resp.result else None,
            "matches_known_answer": match, "match_is_gating": gating_match, "timings_ms": resp.timings.model_dump(),
            "reliability": resp.reliability.model_dump(),
        })  # fmt: skip
    after = rt.readiness()
    check("no leaked inference slots after the run", after["availability"]["running"] == 0 and after["availability"]["waiting"] == 0)
    check("database file unchanged by real queries", db_sha(db_path) == before)
    return outcomes


def main() -> int:
    print(BANNER)
    cfg = load_backend_config()
    root = ARTIFACT_DIR / "databases"
    cfg = cfg.model_copy(update={"databases": cfg.databases.model_copy(update={"root": str(root)})})
    db_path = create_demo_database(root / "demo.sqlite")
    print(f"Fixture database: {db_path}")

    service = build_query_service(cfg)
    if isinstance(service.runtime, UnavailableRuntime):
        print(f"BLOCKER: {service.runtime.describe()['reason']}")
        return 1

    part_a(cfg, db_path)
    outcomes = part_b(cfg, service, db_path)

    gate = all(c["ok"] for c in checks if c["gating"])
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "title": "PRODUCT RELIABILITY CANARY",
        "disclaimer": "NOT A MODEL BENCHMARK. Tiny synthetic fixture, 3 generations; says nothing about accuracy or generalization.",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "runtime": service.runtime.describe(),
        "gate_passed": gate,
        "checks": checks,
        "real_queries": outcomes,
    }
    (ARTIFACT_DIR / "canary_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    failed = [c["check"] for c in checks if c["gating"] and not c["ok"]]
    print(f"\n{BANNER}: {'PASSED' if gate else 'FAILED'}" + (f" (failed: {failed})" if failed else ""))
    print(f"Report: {ARTIFACT_DIR / 'canary_report.json'}")
    return 0 if gate else 1


if __name__ == "__main__":
    sys.exit(main())
