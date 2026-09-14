import json

import pytest

from localsql.model.run_artifacts import (
    Provenance,
    ResumeConflictError,
    RunConfig,
    RunDirectory,
    numeric_stats,
    percentile,
)


def _config(**overrides) -> RunConfig:
    defaults = dict(
        run_id="test-run",
        model_id="Qwen/Qwen3-4B-Instruct-2507",
        model_revision="abc123",
        quantization={"quant_type": "nf4"},
        generation={"do_sample": False, "seed": 42},
        context_mode="with_business_context",
        manifest_path="manifest.jsonl",
        manifest_sha256="deadbeef",
        limit=None,
    )
    defaults.update(overrides)
    return RunConfig(**defaults)


def test_run_config_matches_ignores_run_id():
    a = _config(run_id="run-a")
    b = _config(run_id="run-b")
    assert a.matches(b)


def test_run_config_mismatch_on_revision():
    a = _config(model_revision="abc123")
    b = _config(model_revision="def456")
    assert not a.matches(b)


def test_fresh_run_directory_prepare_succeeds(tmp_path):
    run_dir = RunDirectory(tmp_path, "run-1")
    run_dir.prepare(_config())
    assert run_dir.run_config_path.exists()


def test_resume_with_matching_config_succeeds(tmp_path):
    run_dir = RunDirectory(tmp_path, "run-1")
    run_dir.prepare(_config())
    run_dir.prepare(_config())  # second call, same config -- must not raise


def test_resume_with_different_config_raises(tmp_path):
    run_dir = RunDirectory(tmp_path, "run-1")
    run_dir.prepare(_config(limit=5))
    with pytest.raises(ResumeConflictError):
        run_dir.prepare(_config(limit=10))


def test_completed_ids_and_resume_do_not_duplicate(tmp_path):
    run_dir = RunDirectory(tmp_path, "run-1")
    run_dir.prepare(_config())
    run_dir.append_generation({"example_id": "ex-0", "db_id": "d", "status": "ok", "predicted_sql": "SELECT 1", "latency_ms": 1, "input_tokens": 1, "output_tokens": 1, "context_mode": "with_business_context"})
    run_dir.append_generation({"example_id": "ex-1", "db_id": "d", "status": "error"})

    completed = run_dir.completed_example_ids()
    assert completed == {"ex-0"}  # the error record is NOT counted as completed

    # Simulate a resumed run re-processing only the remaining example.
    run_dir.append_generation({"example_id": "ex-1", "db_id": "d", "status": "ok", "predicted_sql": "SELECT 2", "latency_ms": 1, "input_tokens": 1, "output_tokens": 1, "context_mode": "with_business_context"})
    completed = run_dir.completed_example_ids()
    assert completed == {"ex-0", "ex-1"}


def test_rewrite_predictions_deduplicates_by_latest_record(tmp_path):
    run_dir = RunDirectory(tmp_path, "run-1")
    run_dir.prepare(_config())
    run_dir.append_generation({"example_id": "ex-0", "db_id": "d", "status": "error"})
    run_dir.append_generation({"example_id": "ex-0", "db_id": "d", "status": "ok", "predicted_sql": "SELECT 1", "latency_ms": 1, "input_tokens": 1, "output_tokens": 1, "context_mode": "with_business_context"})

    count = run_dir.rewrite_predictions(model_id="Qwen/Qwen3-4B-Instruct-2507")
    assert count == 1
    lines = run_dir.predictions_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    pred = json.loads(lines[0])
    assert pred["predicted_sql"] == "SELECT 1"
    assert pred["model_id"] == "Qwen/Qwen3-4B-Instruct-2507"


def test_provenance_dry_run_has_no_gpu_fields():
    prov = Provenance.collect(backend_info=None)
    assert prov.torch_version is None
    d = prov.to_dict()
    assert "python_version" in d


def test_run_config_round_trips_through_json():
    config = _config()
    restored = RunConfig.from_dict(json.loads(json.dumps(config.to_dict())))
    assert restored == config


def test_numeric_stats_and_percentile():
    values = [float(v) for v in range(1, 101)]  # 1..100
    stats = numeric_stats(values)
    assert stats["min"] == 1.0
    assert stats["max"] == 100.0
    assert stats["count"] == 100
    assert abs(percentile(sorted(values), 50) - 50.5) < 1e-6


def test_numeric_stats_empty():
    assert numeric_stats([])["count"] == 0
