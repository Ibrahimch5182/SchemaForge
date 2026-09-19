"""PERSISTENT-SERVING / OPERATIONS CANARY -- NOT A MODEL BENCHMARK.

Verifies the Phase 11 serving path end to end on this machine: the private llama.cpp model
server (Q4_K_M base + hot LoRA) becomes ready, loads the model ONCE, is reused by several
sequential QueryService queries, leaks no inference slot, aborts a cancelled request server-side,
rejects saturation immediately, and stays healthy. It says nothing about accuracy: tiny synthetic
fixture, a handful of generations, no BIRD.

By default the server is started exactly as in production: the `schemaforge-model-server` image
(docker/model-server.Dockerfile + entrypoint, hash verification included) with the GGUFs mounted
read-only on the loopback interface. Pass --server-url to attach to an already running server.

    docker build -f docker/model-server.Dockerfile -t schemaforge-model-server:local .
    uv run python scripts/phase11_model_manifest.py --base B.gguf --lora L.gguf --out-dir MODELDIR   # (files in MODELDIR)
    uv run python scripts/canary_phase11.py --model-dir MODELDIR --base-name base-Q4_K_M.gguf --lora-name lora-1518-f16.gguf

Report: .artifacts/phase11/canary_report.json. Exit 0 only if the gate passes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from localsql.backend import config as bcfg  # noqa: E402
from localsql.backend.bootstrap import build_query_service  # noqa: E402
from localsql.backend.control import CancelToken, InferenceGate  # noqa: E402
from localsql.backend.demo import create_demo_database  # noqa: E402
from localsql.backend.errors import ModelBusyError, RequestCancelledError  # noqa: E402
from localsql.backend.models import QueryRequest  # noqa: E402
from localsql.backend.runtime import UnavailableRuntime  # noqa: E402
from localsql.backend.server_runtime import LlamaServerRuntime  # noqa: E402
from localsql.data.prompt_builder import build_prompt  # noqa: E402
from localsql.data.schema_serializer import serialize_schema  # noqa: E402

BANNER = "PERSISTENT-SERVING CANARY -- NOT A MODEL BENCHMARK"
ARTIFACT_DIR = REPO_ROOT / ".artifacts" / "phase11"

checks: list[dict] = []


def check(name: str, ok: bool, detail: str = "", gating: bool = True) -> bool:
    checks.append({"check": name, "ok": bool(ok), "gating": gating, "detail": detail})
    print(f"  [{'PASS' if ok else ('FAIL' if gating else 'NOTE')}] {name}" + (f" -- {detail}" if detail else ""))
    return bool(ok)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def http_get(url: str, timeout: float = 5.0) -> tuple[int, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")
    except Exception as e:  # noqa: BLE001
        return 0, type(e).__name__


def metrics(base: str) -> dict[str, float]:
    code, text = http_get(base + "/metrics")
    out: dict[str, float] = {}
    for line in text.splitlines():
        m = re.match(r"^(llamacpp:[a-z_]+)\s+([-+0-9.eE]+)$", line)
        if m:
            out[m[1]] = float(m[2])
    return out


def slots_idle(base: str) -> Optional[bool]:
    code, text = http_get(base + "/slots")
    if code != 200:
        return None
    try:
        return not any(s.get("is_processing") for s in json.loads(text))
    except ValueError:
        return None


def docker(*args: str, check_rc: bool = True) -> str:
    r = subprocess.run(["docker", *args], capture_output=True, text=True, encoding="utf-8", errors="replace")
    if check_rc and r.returncode != 0:
        raise RuntimeError(f"docker {args[0]} failed: {(r.stderr or r.stdout)[-400:]}")
    return (r.stdout or "") + (r.stderr or "")


def norm_rows(rows: list[list[Any]]) -> list[tuple]:
    def n(v: Any) -> str:
        return f"{float(v):.6f}" if isinstance(v, (int, float)) and not isinstance(v, bool) else str(v)

    return sorted(tuple(sorted(n(v) for v in row)) for row in rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", type=Path)
    ap.add_argument("--base-name")
    ap.add_argument("--lora-name")
    ap.add_argument("--image", default="schemaforge-model-server:local")
    ap.add_argument("--server-url", help="attach to an already running llama-server instead of starting the container")
    ap.add_argument("--ready-timeout", type=float, default=600.0)
    ap.add_argument("--query-timeout", type=float, default=600.0)
    ap.add_argument("--ngl", type=int, default=0)
    ap.add_argument("--threads", type=int, default=-1, help="CPU threads for llama-server (-1 = all logical CPUs; on shared/SMT hosts set the physical core count -- measured 30x difference)")
    args = ap.parse_args()
    print(BANNER)

    container: Optional[str] = None
    base_url = args.server_url
    started_at = time.time()
    if base_url is None:
        if not (args.model_dir and args.base_name and args.lora_name):
            print("--model-dir/--base-name/--lora-name are required unless --server-url is given", file=sys.stderr)
            return 2
        port = free_port()
        container = f"schemaforge-canary-{int(time.time())}"
        docker("run", "-d", "--name", container, "-p", f"127.0.0.1:{port}:8080", "-v", f"{args.model_dir.resolve()}:/models:ro",
               "-e", f"BASE_GGUF_NAME={args.base_name}", "-e", f"LORA_GGUF_NAME={args.lora_name}", "-e", f"LLAMA_NGL={args.ngl}", "-e", f"LLAMA_THREADS={args.threads}",
               "--cap-drop", "ALL", "--security-opt", "no-new-privileges:true", args.image)  # fmt: skip
        base_url = f"http://127.0.0.1:{port}"
        print(f"Model server container {container} starting (production entrypoint, hash verification on)")

    report: dict[str, Any] = {}
    try:
        cfg = bcfg.load_backend_config(env={})
        root = ARTIFACT_DIR / "databases"
        cfg = cfg.model_copy(update={"databases": cfg.databases.model_copy(update={"root": str(root)})})
        db_path = create_demo_database(root / "demo.sqlite")
        env = {bcfg.ENV_RUNTIME_KIND: "llama_server", bcfg.ENV_SERVER_URL: base_url, bcfg.ENV_MODEL_TIMEOUT: str(args.query_timeout), bcfg.ENV_NGL: str(args.ngl)}
        service = build_query_service(cfg, env=env)
        rt = service.runtime
        if isinstance(rt, UnavailableRuntime) or not isinstance(rt, LlamaServerRuntime):
            print(f"BLOCKER: {rt.describe()}")
            return 1

        # ---- 1. readiness without inference -------------------------------------------------
        print("1. readiness (no inference):")
        t0 = time.monotonic()
        state = "unreachable"
        while time.monotonic() - t0 < args.ready_timeout:
            state = rt.readiness()["serving"]["state"]
            if state in ("ready", "idle"):
                break
            if container and "running" not in docker("inspect", "-f", "{{.State.Status}}", container, check_rc=False):
                print(docker("logs", "--tail", "30", container, check_rc=False))
                break
            time.sleep(2)
        load_s = round(time.monotonic() - t0, 1)
        ready = service.readiness()
        check("model server became ready", ready["ready"] is True, f"{load_s}s from container start; serving={ready.get('serving')}")
        pre = metrics(base_url)
        check("readiness/health triggered no inference", pre.get("llamacpp:prompt_tokens_total", 0) == 0 and pre.get("llamacpp:tokens_predicted_total", 0) == 0,
              f"metrics before first query: {pre}", gating=bool(pre))  # fmt: skip
        h = service.health()
        check("health reports persistent mode", h["model_runtime"].get("runtime_mode") == "persistent_server")

        # ---- 2. sequential queries reuse the same resident server -----------------------------
        print("2. three sequential queries through the full QueryService pipeline:")
        questions = [
            ("count", "How many employees are there?", None, [[12]], True),
            ("join_with_context", "What is the total salary of employees in the Engineering department?", "Engineering is a department name stored in departments.name.", [[625000.0]], True),
            ("group_by", "How many employees are in each department? Show the department name and the number of employees.", None, [["Engineering", 5], ["Sales", 4], ["Support", 3]], False),
        ]  # same fixture questions as the Phase 10 canary
        db_before = hashlib.sha256(db_path.read_bytes()).hexdigest()
        outcomes = []
        for qid, question, ctx, expected, gating_match in questions:
            resp = service.query(QueryRequest(database_id="demo", question=question, business_context=ctx))
            ran = resp.status == "ok" and resp.result is not None
            check(f"{qid}: pipeline succeeded", ran, f"status={resp.status}" + (f" code={resp.error.code}" if resp.error else "") + f" total={resp.timings.total_ms:.0f}ms")
            match = bool(ran and norm_rows(resp.result.rows) == norm_rows(expected))
            check(f"{qid}: result matches the known answer", match, str(resp.result.rows if resp.result else None), gating=gating_match)
            check(f"{qid}: served by the persistent server", resp.model.get("runtime_mode") == "persistent_server")
            outcomes.append({
                "id": qid, "status": resp.status, "error_code": resp.error.code if resp.error else None, "generated_sql": resp.generated_sql,
                "matches_known_answer": match, "timings_ms": resp.timings.model_dump(),
                "input_tokens": resp.model.get("input_tokens"), "output_tokens": resp.model.get("output_tokens"),
                "generated_tokens_per_second": resp.model.get("generated_tokens_per_second"),
                "prompt_tokens_per_second": resp.model.get("prompt_tokens_per_second"),
                "queue_wait_ms": resp.model.get("queue_wait_ms"), "generation_latency_ms": resp.model.get("generation_latency_ms"),
            })  # fmt: skip
        check("database file unchanged by real queries", hashlib.sha256(db_path.read_bytes()).hexdigest() == db_before)

        post = metrics(base_url)
        check("server counters accumulated across queries (one resident process)",
              post.get("llamacpp:prompt_tokens_total", 0) > 0 and post.get("llamacpp:tokens_predicted_total", 0) > 0, str(post), gating=bool(post))  # fmt: skip
        model_loads = None
        if container:
            logs = docker("logs", container, check_rc=False)
            model_loads = len(re.findall(r"load_model: loading model '", logs))
            restarts = docker("inspect", "-f", "{{.RestartCount}}", container).strip()
            check("model loaded exactly once; container never restarted", model_loads == 1 and restarts == "0", f"model loads in log={model_loads}, restarts={restarts}")
        ok_gen = [o for o in outcomes if o["generation_latency_ms"]]
        lat = [o["generation_latency_ms"] for o in ok_gen]
        if len(lat) >= 2:
            check("warm queries do not pay the model-load cost", min(lat[1:]) < load_s * 1000 * 0.5 or load_s < 5,
                  f"load {load_s}s vs per-query generation latencies {[round(x) for x in lat]} ms", gating=False)  # fmt: skip

        # ---- 3. saturation + cancellation against the REAL server ------------------------------
        print("3. saturation and server-side cancellation:")
        db = service._registry.resolve("demo")  # noqa: SLF001 - canary reaches into the composition root
        from localsql.backend.introspection import SQLiteSchemaIntrospector

        prompt = build_prompt(serialize_schema(SQLiteSchemaIntrospector().introspect(db)), "sqlite", "List every employee with their department name and salary, ordered by salary.", None)
        strict = LlamaServerRuntime(rt._settings, gate=InferenceGate(1, 0, 0))  # noqa: SLF001 - one slot, no queue
        token = CancelToken()
        result: dict[str, Any] = {}

        def run_a() -> None:
            try:
                strict.generate(prompt, cancel=token)
                result["a"] = "completed"
            except RequestCancelledError:
                result["a"] = "cancelled"
            except Exception as e:  # noqa: BLE001
                result["a"] = type(e).__name__

        ta = threading.Thread(target=run_a)
        ta.start()
        t_wait = time.monotonic()
        while strict.gate.snapshot()["running"] == 0 and time.monotonic() - t_wait < 20:
            time.sleep(0.05)
        time.sleep(0.3)
        tb = time.monotonic()
        try:
            strict.generate(prompt)
            busy = False
        except ModelBusyError:
            busy = True
        busy_ms = (time.monotonic() - tb) * 1000
        check("saturated runtime rejects immediately (no unbounded queue)", busy and busy_ms < 1000, f"model_busy after {busy_ms:.0f} ms")
        tc = time.monotonic()
        token.cancel()
        ta.join(30)
        cancel_ms = (time.monotonic() - tc) * 1000
        raced = result.get("a") == "completed"
        check("in-flight generation cancelled promptly", result.get("a") == "cancelled" and cancel_ms < 3000, f"outcome={result.get('a')} in {cancel_ms:.0f} ms", gating=not raced)
        idle_deadline = time.monotonic() + 30
        idle = slots_idle(base_url)
        while idle is False and time.monotonic() < idle_deadline:
            time.sleep(0.5)
            idle = slots_idle(base_url)
        check("server freed the slot after cancel (no leaked inference slot)", idle is not False, f"/slots idle={idle}", gating=idle is not None)
        check("backend gate has no leaked slot", strict.gate.snapshot()["running"] == 0 and rt.gate.snapshot()["running"] == 0)

        # ---- 4. still healthy and reusable ---------------------------------------------------
        print("4. recovery:")
        resp = service.query(QueryRequest(database_id="demo", question="How many employees are there?"))
        check("query after cancel/saturation succeeds on the same server", resp.status == "ok", f"status={resp.status}")
        final_ready = service.readiness()
        check("readiness healthy at the end", final_ready["ready"] is True and final_ready["availability"]["state"] == "idle", json.dumps(final_ready))
        if container:
            check("same container the whole time", "true" in docker("inspect", "-f", "{{.State.Running}}", container, check_rc=False))

        tps = [o["generated_tokens_per_second"] for o in outcomes if o["generated_tokens_per_second"]]
        report = {
            "runtime_mode": "persistent_server", "server_start_to_ready_s": load_s, "queries": outcomes,
            "mean_generated_tokens_per_second": round(sum(tps) / len(tps), 2) if tps else None,
            "cancel_latency_ms": round(cancel_ms), "saturation_reject_ms": round(busy_ms),
            "server_metrics_after": post, "model_loads_in_log": model_loads,
            "gpu_layers": args.ngl, "threads": args.threads, "wall_seconds": round(time.time() - started_at, 1),
        }  # fmt: skip
    finally:
        if container:
            log_tail = docker("logs", "--tail", "15", container, check_rc=False)
            docker("rm", "-f", container, check_rc=False)
            print(f"Container {container} removed.")
            report["server_log_tail_line_count"] = len(log_tail.splitlines())

    gate = all(c["ok"] for c in checks if c["gating"])
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "title": "PERSISTENT-SERVING CANARY",
        "disclaimer": "NOT A MODEL BENCHMARK. Tiny synthetic fixture, a handful of generations; says nothing about accuracy.",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "gate_passed": gate, "checks": checks, **report,
    }
    (ARTIFACT_DIR / "canary_report.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    failed = [c["check"] for c in checks if c["gating"] and not c["ok"]]
    print(f"\n{BANNER}: {'PASSED' if gate else 'FAILED'}" + (f" (failed: {failed})" if failed else ""))
    print(f"Report: {ARTIFACT_DIR / 'canary_report.json'}")
    return 0 if gate else 1


if __name__ == "__main__":
    sys.exit(main())
