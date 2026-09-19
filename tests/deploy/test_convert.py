import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

from localsql.deploy import convert, process
from localsql.deploy.process import ProcessResult

from tests.deploy.fixtures import REPO_ROOT

SPACED = "C:\\Program Files\\llama cpp\\bin"


def _fake_run_factory(calls, returncode=0, write_output=True, stderr=""):
    def fake_run(command, timeout=None, cwd=None, poll_interval_s=0.05):
        calls.append(list(command))
        if write_output and returncode == 0:
            # tool writes to whatever path follows --outfile / -o, or the last-but-one positional
            out = next(c for c in command if str(c).endswith(".partial.gguf"))
            Path(out).write_bytes(b"gguf-bytes")
        return ProcessResult(returncode, "out", stderr, 12.0, False, None, "n/a")

    return fake_run


def _quantize_plan(tmp_path):
    exe = tmp_path / "llama-quantize.exe"
    exe.write_bytes(b"x")
    src = tmp_path / "merged.gguf"
    src.write_bytes(b"merged")
    return convert.plan_quantize(exe, src, tmp_path / "out" / "q.gguf", "Q4_K_M"), src


def test_command_construction_all_stages():
    base = convert.plan_base_gguf("py", Path("c.py"), Path("hf"), Path("b.gguf"), "f16", ["--x"])
    assert base.command == ["py", "c.py", "hf", "--outfile", "b.gguf", "--outtype", "f16", "--x"]
    lora = convert.plan_lora_gguf("py", Path("l.py"), Path("ad"), Path("hf"), Path("l.gguf"), "f16")
    assert lora.command == ["py", "l.py", "ad", "--base", "hf", "--outfile", "l.gguf", "--outtype", "f16"]
    merge = convert.plan_merge(Path("exp"), Path("b.gguf"), Path("l.gguf"), Path("m.gguf"))
    assert merge.command == ["exp", "-m", "b.gguf", "--lora", "l.gguf", "-o", "m.gguf"]
    quant = convert.plan_quantize(Path("q"), Path("m.gguf"), Path("o.gguf"), "Q4_K_M")
    assert quant.command == ["q", "m.gguf", "o.gguf", "Q4_K_M"]


def test_windows_paths_with_spaces_stay_single_argv_items():
    plan = convert.plan_merge(
        Path(SPACED) / "llama-export-lora.exe", Path("D:/my models/base.gguf"), Path("D:/my models/l.gguf"),
        Path("D:/my models/out dir/m.gguf"),
    )  # fmt: skip
    assert all(" " not in a or a in plan.command for a in plan.command)  # no splitting into pieces
    assert str(Path("D:/my models/base.gguf")) in plan.command
    assert "my models" in convert.dry_run_report(plan)["command_display"]
    assert '"' in convert.dry_run_report(plan)["command_display"]  # display quoting


def test_dry_run_plans_but_touches_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(convert, "run_captured", lambda *a, **k: pytest.fail("must not execute"))
    plan = convert.plan_quantize(tmp_path / "missing.exe", tmp_path / "missing.gguf", tmp_path / "o" / "q.gguf", "Q4_K_M")
    report = convert.dry_run_report(plan)
    assert report["dry_run"] and len(report["preflight_problems"]) == 2
    assert list(tmp_path.iterdir()) == []


def test_successful_stage_records_hash_size_and_command(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(convert, "run_captured", _fake_run_factory(calls))
    plan, src = _quantize_plan(tmp_path)
    rec = convert.execute_stage(plan, tmp_path / "manifests")
    assert rec["status"] == "completed" and rec["output"]["size_bytes"] == len(b"gguf-bytes")
    assert len(rec["output"]["sha256"]) == 64 and rec["inputs"]["input_gguf"]["sha256"]
    assert rec["command"] == plan.command and rec["quant_type"] == "Q4_K_M"
    assert calls[0][-2].endswith(".partial.gguf")  # wrote to partial, then renamed
    assert plan.output.exists() and not convert.partial_path(plan.output).exists()
    saved = json.loads(convert.record_path(tmp_path / "manifests", plan).read_text())
    assert saved["output"] == rec["output"]


def test_rerun_identical_is_noop_but_different_is_refused(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(convert, "run_captured", _fake_run_factory(calls))
    plan, src = _quantize_plan(tmp_path)
    convert.execute_stage(plan, tmp_path / "manifests")
    again = convert.execute_stage(plan, tmp_path / "manifests")
    assert again["status"] == "already_complete" and len(calls) == 1

    src.write_bytes(b"different merged model")  # input changed -> must not silently replace
    with pytest.raises(convert.StageError, match="Refusing to overwrite"):
        convert.execute_stage(plan, tmp_path / "manifests")
    assert len(calls) == 1


def test_existing_output_without_record_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(convert, "run_captured", lambda *a, **k: pytest.fail("must not execute"))
    plan, _ = _quantize_plan(tmp_path)
    plan.output.parent.mkdir(parents=True)
    plan.output.write_bytes(b"someone else's artifact")
    with pytest.raises(convert.StageError, match="without a stage record"):
        convert.execute_stage(plan, tmp_path / "manifests")
    assert plan.output.read_bytes() == b"someone else's artifact"


def test_nonzero_exit_fails_immediately_and_leaves_no_output(tmp_path, monkeypatch):
    monkeypatch.setattr(convert, "run_captured", _fake_run_factory([], returncode=3, stderr="boom"))
    plan, _ = _quantize_plan(tmp_path)
    with pytest.raises(convert.StageError, match="exit code 3"):
        convert.execute_stage(plan, tmp_path / "manifests")
    assert not plan.output.exists()
    assert not convert.record_path(tmp_path / "manifests", plan).exists()


def test_exit_zero_without_output_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(convert, "run_captured", _fake_run_factory([], write_output=False))
    plan, _ = _quantize_plan(tmp_path)
    with pytest.raises(convert.StageError, match="produced no output"):
        convert.execute_stage(plan, tmp_path / "manifests")


def test_missing_tool_or_input_fails_clearly(tmp_path):
    plan = convert.plan_quantize(tmp_path / "nope.exe", tmp_path / "nope.gguf", tmp_path / "q.gguf", "Q4_K_M")
    with pytest.raises(convert.StageError, match="preflight failed"):
        convert.execute_stage(plan, tmp_path / "manifests")


def test_run_captured_never_uses_shell(monkeypatch):
    seen = {}

    class FakePopen:
        returncode = 0

        def __init__(self, argv, **kwargs):
            seen["argv"], seen["kwargs"] = argv, kwargs

        def communicate(self, timeout=None):
            return b"o", b"e"

    monkeypatch.setattr(process.subprocess, "Popen", FakePopen)
    res = process.run_captured(["C:\\a b\\tool.exe", "arg with space"])
    assert seen["kwargs"]["shell"] is False and isinstance(seen["argv"], list)
    assert seen["argv"] == ["C:\\a b\\tool.exe", "arg with space"]
    assert res.stdout == "o" and res.peak_rss_bytes is None and "not_available" in res.peak_rss_note


def test_run_captured_timeout_kills(monkeypatch):
    class FakePopen:
        returncode = None
        killed = False

        def __init__(self, argv, **kw):
            pass

        def communicate(self, timeout=None):
            if not FakePopen.killed:
                raise subprocess.TimeoutExpired("x", timeout)
            return b"", b""

        def kill(self):
            FakePopen.killed = True

    monkeypatch.setattr(process.subprocess, "Popen", FakePopen)
    res = process.run_captured(["x"], timeout=1)
    assert res.timed_out and res.returncode is None


def test_no_shell_true_anywhere_in_phase7_source():
    files = list((REPO_ROOT / "src" / "localsql" / "deploy").glob("*.py"))
    files += [REPO_ROOT / "scripts" / n for n in (
        "prepare_phase7.py", "convert_phase7.py", "run_local_gguf.py", "benchmark_gguf.py", "compare_gguf_outputs.py")]  # fmt: skip
    for f in files:
        text = f.read_text(encoding="utf-8")
        assert "shell=True" not in text, f
        assert "os.system" not in text, f


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_convert_cli_dry_run_writes_nothing(tmp_path, capsys, monkeypatch):
    mod = _load_script("convert_phase7")
    monkeypatch.setattr(convert, "run_captured", lambda *a, **k: pytest.fail("must not execute"))
    before = set((REPO_ROOT / ".artifacts").rglob("*")) if (REPO_ROOT / ".artifacts").exists() else set()
    rc = mod.main(["quantize", "--dry-run", "--quantize-exe", str(tmp_path / "q dir" / "llama-quantize.exe")])
    out = capsys.readouterr().out
    assert rc == 0 and "Q4_K_M" in out and "preflight_problems" in out
    assert "base-f16.gguf" in out and "base-Q4_K_M.gguf" in out and "merged" not in out
    after = set((REPO_ROOT / ".artifacts").rglob("*")) if (REPO_ROOT / ".artifacts").exists() else set()
    assert before == after


def test_primary_path_dry_run_needs_no_export_lora(tmp_path, capsys, monkeypatch):
    """base -> lora -> quantize-base dry-runs work with only convert scripts + llama-quantize."""
    mod = _load_script("convert_phase7")
    monkeypatch.setattr(convert, "run_captured", lambda *a, **k: pytest.fail("must not execute"))
    assert mod.main(["base", "--dry-run", "--convert-script", "c.py", "--hf-dir", str(tmp_path)]) == 0
    assert mod.main(["lora", "--dry-run", "--convert-script", "l.py", "--hf-dir", str(tmp_path)]) == 0
    assert mod.main(["quantize", "--dry-run", "--quantize-exe", "llama-quantize.exe"]) == 0
    out = capsys.readouterr().out
    assert "llama-export-lora" not in out


def test_quantize_defaults_to_base_f16_and_merge_is_optional_subcommand():
    mod = _load_script("convert_phase7")
    from localsql.deploy.config import load_phase7_config
    from localsql.deploy.manifest import default_artifact_paths

    cfg = load_phase7_config(REPO_ROOT / "configs" / "phase7.yaml")
    paths = default_artifact_paths(REPO_ROOT / ".artifacts" / "phase7", cfg)
    args = mod.build_parser().parse_args(["quantize", "--quantize-exe", "q.exe"])
    plan = mod.make_plan(args, cfg, paths)
    assert plan.command[-3:] == [paths["base_f16_gguf"], paths["base_q4_k_m_gguf"], "Q4_K_M"]
    assert "merged" not in paths["base_q4_k_m_gguf"] and paths["optional_merged_f16_gguf"] != paths["base_q4_k_m_gguf"]
    assert cfg.deployment.mode == "hot_lora" and cfg.deployment.merged_export_optional
    # merge still exists, but nothing in the primary path depends on it
    merge_args = mod.build_parser().parse_args(["merge", "--export-lora-exe", "x.exe"])
    assert mod.make_plan(merge_args, cfg, paths).name == "merge"
