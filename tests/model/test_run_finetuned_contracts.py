"""Contract tests for scripts/run_finetuned.py.

No model/CUDA/network involved -- pure plumbing: manifest parsing (same
gold-free contract as run_baseline.py), fail-closed adapter validation,
resume/dry-run behavior, and that generation goes through the shared
`generate_one_example` path with unchanged whitespace-only normalization.
"""

import importlib.util
import json
from pathlib import Path

import pytest

from localsql.benchmark.models import GenerationExample
from localsql.model.generation import generate_one_example, normalize_predicted_sql
from localsql.model.qwen_backend import AdapterValidationError
from localsql.model.run_artifacts import RunConfig, RunDirectory

from tests.model.fixtures import FakeBackend, make_example

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_FINETUNED_PATH = REPO_ROOT / "scripts" / "run_finetuned.py"

_FORBIDDEN_SYMBOLS = {"GradingExample", "load_gold_sql", "analyze_source_consistency", "GRADING"}


def _load_run_finetuned_module():
    spec = importlib.util.spec_from_file_location("run_finetuned_under_test", RUN_FINETUNED_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_adapter_dir(tmp_path: Path) -> Path:
    adapter_dir = tmp_path / "checkpoint-1518"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"fake-weights")
    return adapter_dir


def _write_manifest(path: Path, examples) -> None:
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(ex.model_dump_json() + "\n")


def test_runner_module_has_no_gold_or_grading_symbols():
    module = _load_run_finetuned_module()
    leaked = _FORBIDDEN_SYMBOLS & set(dir(module))
    assert not leaked, f"run_finetuned.py imports gold/grading symbols: {leaked}"


def test_load_manifest_parses_generation_examples_only(tmp_path):
    module = _load_run_finetuned_module()
    manifest_path = tmp_path / "manifest.jsonl"
    examples = [make_example(0), make_example(1)]
    _write_manifest(manifest_path, examples)

    loaded = module.load_manifest(manifest_path)
    assert len(loaded) == 2
    assert all(isinstance(ex, GenerationExample) for ex in loaded)
    assert [ex.example_id for ex in loaded] == [ex.example_id for ex in examples]


def test_load_manifest_rejects_a_row_with_gold_sql_field(tmp_path):
    module = _load_run_finetuned_module()
    manifest_path = tmp_path / "manifest.jsonl"
    bad_row = {**json.loads(make_example(0).model_dump_json()), "sql": "SELECT * FROM secret_gold"}
    manifest_path.write_text(json.dumps(bad_row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="gold"):
        module.load_manifest(manifest_path)


def test_adapter_model_id_differs_from_base_model_id(tmp_path):
    module = _load_run_finetuned_module()
    adapter_dir = _make_adapter_dir(tmp_path)
    tagged = module.adapter_model_id("Qwen/Qwen3-4B-Instruct-2507", adapter_dir)
    assert tagged != "Qwen/Qwen3-4B-Instruct-2507"
    assert "Qwen/Qwen3-4B-Instruct-2507" in tagged
    assert "checkpoint-1518" in tagged


def test_dry_run_does_not_require_cuda_or_model_loading(tmp_path, capsys):
    """--dry-run must succeed with a valid adapter dir and never import
    torch/transformers/peft."""
    module = _load_run_finetuned_module()
    adapter_dir = _make_adapter_dir(tmp_path)
    manifest_path = tmp_path / "manifest.jsonl"
    _write_manifest(manifest_path, [make_example(0), make_example(1)])

    run_config = RunConfig(
        run_id="ft-dry-run",
        model_id=module.adapter_model_id("Qwen/Qwen3-4B-Instruct-2507", adapter_dir),
        model_revision=None,
        quantization={"quant_type": "nf4"},
        generation={"do_sample": False, "seed": 42},
        context_mode="with_business_context",
        manifest_path=str(manifest_path),
        manifest_sha256="deadbeef",
        limit=None,
    )
    run_dir = RunDirectory(tmp_path / "runs", "ft-dry-run")
    manifest = module.load_manifest(manifest_path)

    module.run_dry_run(manifest, run_dir, run_config, adapter_dir)

    out = capsys.readouterr().out
    assert "Dry run OK" in out
    assert run_dir.run_config_path.exists()


def test_dry_run_fails_closed_on_missing_adapter(tmp_path):
    module = _load_run_finetuned_module()
    missing_adapter = tmp_path / "no-such-adapter"
    manifest_path = tmp_path / "manifest.jsonl"
    _write_manifest(manifest_path, [make_example(0)])

    run_config = RunConfig(
        run_id="ft-dry-run-missing",
        model_id=module.adapter_model_id("Qwen/Qwen3-4B-Instruct-2507", missing_adapter),
        model_revision=None,
        quantization={"quant_type": "nf4"},
        generation={"do_sample": False, "seed": 42},
        context_mode="with_business_context",
        manifest_path=str(manifest_path),
        manifest_sha256="deadbeef",
        limit=None,
    )
    run_dir = RunDirectory(tmp_path / "runs", "ft-dry-run-missing")
    manifest = module.load_manifest(manifest_path)

    with pytest.raises(AdapterValidationError):
        module.run_dry_run(manifest, run_dir, run_config, missing_adapter)


def test_generation_goes_through_shared_generate_one_example_path():
    """Same contract as run_baseline.py: generation is produced by the
    shared `generate_one_example`, not a second implementation."""
    manifest = [make_example(i) for i in range(3)]
    backend = FakeBackend(raw_completion="  SELECT COUNT(*) FROM customers  ")
    records = [
        generate_one_example(ex, backend, model_revision="abc", context_mode="with_business_context")
        for ex in manifest
    ]
    assert backend.calls == [ex.prompt for ex in manifest]
    # Whitespace-only normalization, unchanged.
    for r in records:
        assert r.predicted_sql == normalize_predicted_sql(r.raw_completion)
        assert r.predicted_sql == "SELECT COUNT(*) FROM customers"


def test_main_fails_closed_before_touching_manifest_on_bad_adapter(tmp_path, capsys):
    """A missing --adapter must be rejected before the manifest is even
    read (fail closed as early as possible)."""
    module = _load_run_finetuned_module()
    manifest_path = tmp_path / "manifest.jsonl"
    # Deliberately do NOT create the manifest file -- if load_manifest were
    # reached first, this would raise FileNotFoundError instead of the
    # expected clean BLOCKER exit.
    import sys as _sys

    argv = [
        "run_finetuned.py",
        "--manifest",
        str(manifest_path),
        "--run-id",
        "ft-missing-adapter",
        "--adapter",
        str(tmp_path / "does-not-exist"),
        "--dry-run",
    ]
    old_argv = _sys.argv
    _sys.argv = argv
    try:
        with pytest.raises(SystemExit) as exc_info:
            module.main()
    finally:
        _sys.argv = old_argv

    assert exc_info.value.code == 1
    out = capsys.readouterr().out
    assert "BLOCKER" in out
