import json
import random
from pathlib import Path

import pytest

from localsql.deploy import runtime
from localsql.deploy.process import ProcessResult
from localsql.deploy.runtime import LlamaSettings
from localsql.deploy.sanity import (
    compare_generations,
    latency_summary,
    load_generation_manifest,
    rate_summary,
    select_sanity_examples,
)
from localsql.benchmark.models import GenerationExample

from tests.deploy.fixtures import make_example_dict

SETTINGS = LlamaSettings(
    executable=Path("C:/llama cpp/llama-completion.exe"),
    model=Path("D:/my models/q.gguf"),
    context_size=4096,
    max_new_tokens=64,
    seed=42,
)

PERF_STDERR = (
    "llama_perf_context_print:        load time =    900.00 ms\n"
    "llama_perf_context_print: prompt eval time =    500.00 ms /   250 tokens (    2.00 ms per token,   500.00 tokens per second)\n"
    "llama_perf_context_print:        eval time =   1000.00 ms /    40 runs   (   25.00 ms per token,    40.00 tokens per second)\n"
)


def test_chatml_envelope_wraps_canonical_prompt_verbatim():
    p = "SYSTEM:\nx\n\nQUESTION:\nq"
    assert runtime.build_chatml_prompt(p) == f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n"


def test_llama_command_is_greedy_with_spaced_paths_as_single_args():
    cmd = runtime.build_llama_command(SETTINGS, Path("C:/tmp dir/prompt.txt"))
    assert cmd[0] == str(SETTINGS.executable) and cmd[cmd.index("-m") + 1] == str(SETTINGS.model)
    assert cmd[cmd.index("--temp") + 1] == "0" and cmd[cmd.index("--top-k") + 1] == "1"
    assert cmd[cmd.index("--seed") + 1] == "42" and cmd[cmd.index("-ngl") + 1] == "0"
    assert cmd[cmd.index("-f") + 1] == str(Path("C:/tmp dir/prompt.txt")) and "-no-cnv" in cmd
    off = LlamaSettings(**{**SETTINGS.__dict__, "no_conversation_flag": False, "extra_args": ("--verbose-prompt",)})
    c2 = runtime.build_llama_command(off, Path("p"))
    assert "-no-cnv" not in c2 and c2[-1] == "--verbose-prompt"


def test_parse_perf_present_and_absent():
    perf = runtime.parse_perf(PERF_STDERR)
    assert perf["prompt_tokens"] == 250 and perf["prompt_tokens_per_second"] == 500.0
    assert perf["generated_tokens"] == 40 and perf["generated_tokens_per_second"] == 40.0
    none = runtime.parse_perf("nothing useful")
    assert none["generated_tokens_per_second"] is None and none["availability"].startswith("not_available")


def test_clean_completion_only_strips_end_marker():
    assert runtime.clean_completion("SELECT 1 [end of text]\n") == ("SELECT 1 ", True)
    assert runtime.clean_completion("SELECT 1\n") == ("SELECT 1\n", False)


def _patch_process(monkeypatch, result):
    seen = {}

    def fake(cmd, timeout=None, **kw):
        seen["cmd"], seen["timeout"] = cmd, timeout
        seen["prompt"] = Path(cmd[cmd.index("-f") + 1]).read_bytes()
        return result

    monkeypatch.setattr(runtime, "run_captured", fake)
    return seen


def test_run_gguf_once_ok_uses_whitespace_only_normalization(monkeypatch):
    seen = _patch_process(monkeypatch, ProcessResult(0, "  SELECT 1;\n", PERF_STDERR, 1500.0, False, 123, "peak"))
    run = runtime.run_gguf_once("PROMPT\r\nX", SETTINGS)
    assert run.status == "ok" and run.predicted_sql == "SELECT 1;" and run.raw_completion == "  SELECT 1;\n"
    assert run.perf["generated_tokens_per_second"] == 40.0 and run.peak_rss_bytes == 123
    assert seen["prompt"] == b"<|im_start|>user\nPROMPT\r\nX<|im_end|>\n<|im_start|>assistant\n"  # bytes exact
    assert seen["timeout"] == SETTINGS.timeout_seconds


def test_run_gguf_once_failure_and_timeout(monkeypatch):
    _patch_process(monkeypatch, ProcessResult(2, "", "bad model", 10.0, False, None, "n/a"))
    run = runtime.run_gguf_once("p", SETTINGS)
    assert run.status == "error" and "exit code 2" in run.error and run.predicted_sql is None
    _patch_process(monkeypatch, ProcessResult(None, "", "", 10.0, True, None, "n/a"))
    assert runtime.run_gguf_once("p", SETTINGS).status == "timeout"


def test_latency_summary_p50_p95():
    s = latency_summary([100, 200, 300, 400, 500])
    assert s["p50_ms"] == 300 and s["p95_ms"] == 480.0 and s["mean_ms"] == 300 and s["count"] == 5
    assert latency_summary([]) ["p50_ms"] is None
    assert latency_summary([7])["p95_ms"] == 7


def test_rate_summary_null_with_explanation():
    r = rate_summary([None, None], "generation tokens/sec")
    assert r["mean"] is None and "not_available" in r["availability"]
    assert rate_summary([10.0, 20.0, None], "x")["mean"] == 15.0


def _examples(n=50):
    return [GenerationExample.model_validate(make_example_dict(i)) for i in range(n)]


def test_sanity_selection_deterministic_and_order_independent():
    ex = _examples()
    a = select_sanity_examples(ex, 10, "salt")
    shuffled = ex[:]
    random.Random(1).shuffle(shuffled)
    b = select_sanity_examples(shuffled, 10, "salt")
    assert [e.example_id for e in a] == [e.example_id for e in b] and len(a) == 10
    assert [e.example_id for e in select_sanity_examples(ex, 10, "other")] != [e.example_id for e in a]


def test_real_manifest_selection_is_gold_free_and_stable():
    path = Path(__file__).resolve().parents[2] / "data/benchmarks/bird_mini_dev/generation/manifest.jsonl"
    if not path.exists():
        pytest.skip("generation manifest not present (gitignored)")
    ex = load_generation_manifest(path)
    sel = select_sanity_examples(ex, 10, "schemaforge-phase7-quant-sanity-v1")
    assert len(sel) == 10 and all(not hasattr(e, "sql") for e in sel)


def test_manifest_with_gold_fields_rejected(tmp_path):
    bad = make_example_dict(1) | {"sql": "SELECT 1"}
    p = tmp_path / "m.jsonl"
    p.write_text(json.dumps(bad) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="gold-free"):
        load_generation_manifest(p)


def _doc(sha, sqls):
    return {"model": {"sha256": sha}, "results": [{"example_id": f"e{i}", "predicted_sql": s} for i, s in enumerate(sqls)]}


def test_compare_generations_agreement_and_changes():
    ref = _doc("A", ["SELECT 1", "SELECT 2", "SELECT 3"])
    cand = _doc("B", ["SELECT 1", "SELECT 9", None])
    rep = compare_generations(ref, cand)
    assert rep["n"] == 3 and rep["exact_normalized_agreement"] == 1
    assert [c["example_id"] for c in rep["changed_examples"]] == ["e1", "e2"]
    assert rep["reference_shape_counts"]["parseable"] == 3 and rep["candidate_shape_counts"]["parseable"] == 2
    assert rep["kind"] == "quantization_regression_sanity_check"
    with pytest.raises(ValueError):
        compare_generations(ref, _doc("B", ["SELECT 1"]))


LORA = Path("D:/my models/lora 1518.gguf")


def test_runtime_lora_command_construction():
    s = LlamaSettings(**{**SETTINGS.__dict__, "lora": LORA})
    cmd = runtime.build_llama_command(s, Path("p"))
    i = cmd.index("--lora")
    assert cmd[i + 1] == str(LORA) and cmd.count("--lora") == 1
    assert runtime.settings_dict(s)["deployment_mode"] == "hot_lora"
    assert "--lora" not in runtime.build_llama_command(SETTINGS, Path("p"))
    assert runtime.settings_dict(SETTINGS)["deployment_mode"] == "single_gguf"


def test_deployment_info_records_both_hashes_and_combined_size(tmp_path):
    import hashlib

    base, lora = tmp_path / "base-Q4_K_M.gguf", tmp_path / "lora.gguf"
    base.write_bytes(b"b" * 1000)
    lora.write_bytes(b"l" * 250)
    info = runtime.deployment_info(LlamaSettings(**{**SETTINGS.__dict__, "model": base, "lora": lora}))
    assert info["mode"] == "hot_lora"
    assert info["base"]["sha256"] == hashlib.sha256(b"b" * 1000).hexdigest()
    assert info["lora"]["sha256"] == hashlib.sha256(b"l" * 250).hexdigest()
    assert (info["base"]["size_bytes"], info["lora"]["size_bytes"], info["combined_size_bytes"]) == (1000, 250, 1250)
    single = runtime.deployment_info(LlamaSettings(**{**SETTINGS.__dict__, "model": base}))
    assert single["lora"] is None and single["combined_size_bytes"] == 1000


def _load(name):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parents[2] / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_benchmark_runs_base_plus_lora_and_records_deployment(tmp_path, monkeypatch):
    import sys

    bench = _load("benchmark_gguf")
    exe, base, lora = tmp_path / "llama-completion.exe", tmp_path / "base.gguf", tmp_path / "lora.gguf"
    exe.write_bytes(b"x")
    base.write_bytes(b"b" * 100)
    lora.write_bytes(b"l" * 10)
    manifest = tmp_path / "m.jsonl"
    manifest.write_text("\n".join(json.dumps(make_example_dict(i)) for i in range(12)), encoding="utf-8")
    commands = []

    def fake(cmd, timeout=None, **kw):
        commands.append(cmd)
        return ProcessResult(0, "SELECT 1", PERF_STDERR, 100.0, False, 555, "peak")

    monkeypatch.setattr(runtime, "run_captured", fake)
    out = tmp_path / "bench.json"
    monkeypatch.setattr(sys, "argv", ["b", "--llama-exe", str(exe), "--model", str(base), "--lora", str(lora),
                                      "--manifest", str(manifest), "--output", str(out),
                                      "--num-prompts", "2", "--repeats", "2", "--warmup-runs", "1"])  # fmt: skip
    assert bench.main() == 0
    doc = json.loads(out.read_text())
    assert doc["deployment"]["mode"] == "hot_lora" and doc["deployment"]["combined_size_bytes"] == 110
    assert doc["model"]["sha256"] and doc["lora"]["sha256"] and doc["model"]["sha256"] != doc["lora"]["sha256"]
    assert len(commands) == 1 + 2 * 2 and all(str(lora) in c and "--lora" in c for c in commands)
    assert doc["summary"]["deterministic_across_repeats"] is True
    assert doc["protocol"]["warmup_runs"] == 1 and len(doc["measured_runs"]) == 4  # warm-up excluded


def test_benchmark_requires_lora_unless_opted_out(tmp_path, monkeypatch):
    import sys

    bench = _load("benchmark_gguf")
    monkeypatch.setattr(sys, "argv", ["b", "--llama-exe", "e", "--model", "m", "--output", str(tmp_path / "o.json"), "--dry-run"])
    with pytest.raises(SystemExit) as e:
        bench.main()
    assert "--lora" in str(e.value)


def _doc2(lora_sha, sqls):
    d = _doc("BASE", sqls)
    d["lora"] = {"sha256": lora_sha}
    return d


def test_sanity_comparison_requires_same_lora_for_f16_and_q4():
    rep = compare_generations(_doc2("L", ["SELECT 1"]), _doc2("L", ["SELECT 1"]))
    assert rep["shared_lora_sha256"] == "L" and rep["exact_normalized_agreement"] == 1
    with pytest.raises(ValueError, match="SAME LoRA"):
        compare_generations(_doc2("L1", ["SELECT 1"]), _doc2("L2", ["SELECT 1"]))
    with pytest.raises(ValueError, match="SAME LoRA"):
        compare_generations(_doc2("L", ["SELECT 1"]), _doc("BASE", ["SELECT 1"]))
