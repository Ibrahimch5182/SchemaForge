"""Phase 8 integrated smoke: the REAL pipeline end to end.

Builds a deterministic demo SQLite DB under .artifacts/phase8/, then runs real
questions through QueryService with the Phase 7 Q4_K_M base + LoRA GGUF runtime
(configured via SCHEMAFORGE_LLAMA_EXE / _BASE_GGUF / _LORA_GGUF).

Exit 0 only if ALL gating checks pass:
  * schema introspection (tables, single + composite PKs, FKs)
  * independent executor write-protection and safety rejection (no model)
  * for each question: model generation -> safety allowed -> execution ok
Answer correctness on the demo questions is reported but NOT gating (this is a
plumbing check, not an accuracy benchmark).
"""

from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from localsql.backend.bootstrap import build_query_service  # noqa: E402
from localsql.backend.config import database_root, load_backend_config  # noqa: E402
from localsql.backend.demo import DEMO_EMPLOYEE_COUNT, DEMO_TABLES, create_demo_database  # noqa: E402
from localsql.backend.errors import BackendError, ExecutionError  # noqa: E402
from localsql.backend.executor import SQLiteReadOnlyExecutor  # noqa: E402
from localsql.backend.introspection import SQLiteSchemaIntrospector  # noqa: E402
from localsql.backend.models import QueryRequest  # noqa: E402
from localsql.backend.registry import DatabaseRegistry  # noqa: E402
from localsql.backend.runtime import UnavailableRuntime  # noqa: E402
from localsql.backend.safety import SQLSafetyPolicy  # noqa: E402

ARTIFACT_DIR = REPO_ROOT / ".artifacts" / "phase8"

QUESTIONS = [
    {
        "id": "count",
        "question": "How many employees are there?",
        "business_context": None,
        "expected_first_cell": DEMO_EMPLOYEE_COUNT,
    },
    {
        "id": "join_with_context",
        "question": "What is the total salary of employees in the Engineering department?",
        "business_context": "Engineering is a department name stored in departments.name.",
        "expected_first_cell": 625000.0,
    },
]

checks: list[dict] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    checks.append({"check": name, "ok": bool(ok), "detail": detail})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    return bool(ok)


def main() -> int:
    cfg = load_backend_config()
    root = database_root(cfg)
    if root.resolve() != (ARTIFACT_DIR / "databases").resolve():
        print(f"BLOCKER: smoke expects databases.root under {ARTIFACT_DIR / 'databases'} (got {root}). Unset SCHEMAFORGE_DB_ROOT.")
        return 1
    db_path = create_demo_database(root / "demo.sqlite")
    print(f"Demo database: {db_path}")

    service = build_query_service(cfg)
    if isinstance(service.runtime, UnavailableRuntime):
        print(f"BLOCKER: {service.runtime.describe()['reason']}")
        return 1
    print(f"Runtime: {json.dumps(service.runtime.describe())}")

    registry = DatabaseRegistry(root, cfg.database_entries())
    db = registry.resolve("demo")

    print("Schema introspection:")
    schema = SQLiteSchemaIntrospector().introspect(db)
    by_name = {t.name: t for t in schema.tables}
    check("tables", [t.name for t in schema.tables] == DEMO_TABLES, str([t.name for t in schema.tables]))
    pa = by_name["project_assignments"]
    check("composite PK", [c.name for c in pa.columns if c.is_primary_key] == ["emp_id", "project_id"])
    emp = {c.name: c for c in by_name["employees"].columns}
    check("FK employees.dept_id->departments.dept_id", emp["dept_id"].foreign_key is not None
          and (emp["dept_id"].foreign_key.table, emp["dept_id"].foreign_key.column) == ("departments", "dept_id"))  # fmt: skip

    print("Defense in depth (no model):")
    executor = SQLiteReadOnlyExecutor(cfg.execution.timeout_seconds, cfg.execution.max_rows)
    try:
        executor.execute(db, "DELETE FROM employees")
        check("executor blocks DELETE", False, "DELETE executed!")
    except ExecutionError as e:
        check("executor blocks DELETE", True, e.code)
    conn = sqlite3.connect(db.path)
    try:
        n = conn.execute("SELECT COUNT(*) FROM employees").fetchone()[0]
    finally:
        conn.close()
    check("database unchanged", n == DEMO_EMPLOYEE_COUNT, f"{n} employees")
    check("safety rejects DROP", not SQLSafetyPolicy().check("DROP TABLE employees").allowed)

    print("Real pipeline (model generation -> safety -> execution):")
    outcomes = []
    for q in QUESTIONS:
        req = QueryRequest(database_id="demo", question=q["question"], business_context=q["business_context"])
        try:
            resp = service.query(req)
        except BackendError as e:
            check(f"{q['id']}: pipeline", False, f"{e.code}: {e.message}")
            continue
        ok = resp.status == "ok" and resp.safety is not None and resp.safety.allowed and resp.result is not None
        check(f"{q['id']}: generation+safety+execution", ok,
              f"status={resp.status} total={resp.timings.total_ms:.0f}ms" + (f" error={resp.error.code}" if resp.error else ""))  # fmt: skip
        first = resp.result.rows[0][0] if ok and resp.result.rows and resp.result.rows[0] else None
        outcomes.append({
            "id": q["id"], "status": resp.status, "generated_sql": resp.generated_sql,
            "rows": resp.result.rows if resp.result else None,
            "matches_expected_INFORMATIONAL": first == q["expected_first_cell"],
            "timings_ms": resp.timings.model_dump(),
        })  # fmt: skip
        print(f"      SQL: {resp.generated_sql}")
        print(f"      rows: {resp.result.rows if resp.result else None}  (expected first cell {q['expected_first_cell']}, informational)")

    passed = all(c["ok"] for c in checks)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / "smoke_report.json").write_text(
        json.dumps({"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "passed": passed,
                    "checks": checks, "questions": outcomes}, indent=2),  # fmt: skip
        encoding="utf-8",
    )
    print(f"\nPHASE 8 SMOKE: {'PASSED' if passed else 'FAILED'}  (report: {ARTIFACT_DIR / 'smoke_report.json'})")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
