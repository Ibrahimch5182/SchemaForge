"""Contract tests for scripts/run_qlora_full_training.py (Phase 5C).

CPU/offline, no model/CUDA/network. Proves the session-boundary,
resume-compatibility, and canonical-horizon-recording behavior without
ever touching a GPU -- all of these paths execute (and BLOCKER, where
applicable) before `QLoraBackend.load_for_training()` is ever called.
"""

import importlib.util
import json
from pathlib import Path

import pytest

from localsql.train.config import load_train_config
from localsql.train.full_training import compute_optimizer_step_schedule

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = REPO_ROOT / "scripts" / "run_qlora_full_training.py"
TRAIN_CONFIG_PATH = REPO_ROOT / "configs" / "train.yaml"


def _load_runner_module():
    spec = importlib.util.spec_from_file_location("run_qlora_full_training_under_test", RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _example(idx: int, example_id: str | None = None):
    from tests.train.fixtures import make_prepared_example

    return make_prepared_example(idx) if example_id is None else make_prepared_example(idx, db_id="shop_db")


def test_runner_module_never_imports_benchmark_package():
    source = RUNNER_PATH.read_text(encoding="utf-8")
    assert "localsql.benchmark" not in source
    assert "bird_mini_dev" not in source


def test_runner_defaults_to_phase5_candidate_dataset():
    source = RUNNER_PATH.read_text(encoding="utf-8")
    assert 'CANONICAL_TRAIN_FILE = REPO_ROOT / "data" / "processed_phase5_candidate" / "train.jsonl"' in source
    assert 'CANONICAL_VALIDATION_FILE = REPO_ROOT / "data" / "processed_phase5_candidate" / "validation.jsonl"' in source


def test_expected_canonical_schedule_matches_real_dataset_size():
    """Reported numbers requirement: 6,067 examples, batch=1, grad-accum=8,
    2 epochs -- 759 steps/epoch, 1,518 total."""
    schedule = compute_optimizer_step_schedule(6067, 1, 8, 2)
    assert schedule["steps_per_epoch"] == 759
    assert schedule["total_optimizer_steps"] == 1518


def test_run_dry_run_reports_expected_schedule(tmp_path, capsys):
    module = _load_runner_module()
    cfg = load_train_config(TRAIN_CONFIG_PATH)
    examples = [_example(i) for i in range(6067)]  # simulate the real candidate size

    module.run_dry_run(cfg, examples, num_train_epochs=2, run_dir=tmp_path / "run", train_path=Path("fake.jsonl"))

    out = capsys.readouterr().out
    assert "759" in out
    assert "1518" in out


def test_run_full_training_refuses_to_overwrite_completed_run(tmp_path):
    module = _load_runner_module()
    cfg = load_train_config(TRAIN_CONFIG_PATH)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "summary.json").write_text("{}", encoding="utf-8")
    train_path = tmp_path / "train.jsonl"
    train_path.write_text("{}", encoding="utf-8")

    with pytest.raises(SystemExit):
        module.run_full_training(
            cfg, [_example(0)], run_dir, train_path, 2, None, True, None, None, None, None
        )


def test_run_full_training_rejects_nonexistent_resume_checkpoint_path(tmp_path):
    module = _load_runner_module()
    cfg = load_train_config(TRAIN_CONFIG_PATH)
    run_dir = tmp_path / "run"
    train_path = tmp_path / "train.jsonl"
    train_path.write_text("{}", encoding="utf-8")

    with pytest.raises(SystemExit):
        module.run_full_training(
            cfg,
            [_example(0)],
            run_dir,
            train_path,
            2,
            None,
            True,
            None,
            None,
            str(tmp_path / "does_not_exist" / "checkpoint-2"),
            None,
        )


def test_run_full_training_rejects_stop_after_global_step_beyond_canonical_horizon(tmp_path, capsys):
    """stop_after_global_step must never exceed the canonical total
    optimizer-step horizon -- this is a config error, not a valid partial
    session, and must BLOCKER before any GPU work."""
    module = _load_runner_module()
    cfg = load_train_config(TRAIN_CONFIG_PATH)
    run_dir = tmp_path / "run"
    train_path = tmp_path / "train.jsonl"
    train_path.write_text("{}", encoding="utf-8")
    examples = [_example(i) for i in range(10)]  # small -> tiny canonical horizon

    schedule = compute_optimizer_step_schedule(
        len(examples), cfg.optimization.per_device_train_batch_size, cfg.optimization.gradient_accumulation_steps, 2
    )

    with pytest.raises(SystemExit):
        module.run_full_training(
            cfg, examples, run_dir, train_path, 2, schedule["total_optimizer_steps"] + 1, True, None, None, None, None
        )

    assert "exceeds the canonical total" in capsys.readouterr().out


def test_run_full_training_rejects_nonpositive_stop_after_global_step(tmp_path):
    module = _load_runner_module()
    cfg = load_train_config(TRAIN_CONFIG_PATH)
    run_dir = tmp_path / "run"
    train_path = tmp_path / "train.jsonl"
    train_path.write_text("{}", encoding="utf-8")

    with pytest.raises(SystemExit):
        module.run_full_training(cfg, [_example(0)], run_dir, train_path, 2, 0, True, None, None, None, None)


def test_run_full_training_stop_after_global_step_does_not_change_reported_canonical_horizon(tmp_path, capsys, monkeypatch):
    """Requirement: stop-after-global-step must not modify the canonical
    total training horizon. Two otherwise-identical calls, one with a
    (valid, in-range) --stop-after-global-step and one without, must
    report the identical canonical schedule -- verified by intercepting
    right before any GPU work via a fake backend, rather than relying on
    an unrelated BLOCKER's ordering."""
    module = _load_runner_module()
    cfg = load_train_config(TRAIN_CONFIG_PATH)
    examples = [_example(i) for i in range(6067)]

    class _FakeBackend:
        def __init__(self, cfg):
            pass

        def load_for_training(self):
            raise RuntimeError("stopped intentionally right after the schedule was computed/printed")

    import localsql.train.qlora_backend as qlora_backend_module

    monkeypatch.setattr(qlora_backend_module, "QLoraBackend", _FakeBackend)

    for stop_step in (None, 400):
        run_dir = tmp_path / f"run-{stop_step}"
        train_path = tmp_path / "train.jsonl"
        train_path.write_text("{}", encoding="utf-8")
        with pytest.raises(RuntimeError, match="stopped intentionally right after the schedule"):
            module.run_full_training(
                cfg, examples, run_dir, train_path, 2, stop_step, True, None, None, None, None
            )
        out = capsys.readouterr().out
        assert "759 steps/epoch x 2 epochs = 1518 total optimizer steps" in out


def test_run_full_training_resume_compatibility_blocks_on_max_seq_length_mismatch(tmp_path, capsys):
    """Conflicting resume/canonical settings must fail loudly where
    validation is practical -- here, the checkpoint's source run_config.json
    recorded a different max_seq_length than the current config."""
    module = _load_runner_module()
    cfg = load_train_config(TRAIN_CONFIG_PATH)

    source_run_dir = tmp_path / "runs" / "phase5c-session-1"
    checkpoint_dir = source_run_dir / "checkpoint" / "checkpoint-400"
    checkpoint_dir.mkdir(parents=True)
    source_run_config = {
        "model_id": cfg.model.id,
        "max_seq_length": 2048,  # deliberately different from cfg.sequence.max_seq_length (4096)
        "quantization": {"load_in_4bit": True, "quant_type": cfg.runtime.quantization, "double_quant": True, "compute_dtype": "float16"},
        "lora": {"r": cfg.lora.r, "alpha": cfg.lora.alpha, "dropout": cfg.lora.dropout, "target_modules": list(cfg.lora.target_modules), "task_type": cfg.lora.task_type},
        "canonical_num_train_epochs": 2,
        "canonical_total_optimizer_steps": 1518,
        "train_file_sha256": "irrelevant-for-this-test",
    }
    (source_run_dir / "run_config.json").write_text(json.dumps(source_run_config), encoding="utf-8")

    run_dir = tmp_path / "runs" / "phase5c-session-2"
    train_path = tmp_path / "train.jsonl"
    train_path.write_text("{}", encoding="utf-8")
    examples = [_example(i) for i in range(6067)]

    with pytest.raises(SystemExit):
        module.run_full_training(
            cfg, examples, run_dir, train_path, 2, None, True, None, None, str(checkpoint_dir), None
        )

    out = capsys.readouterr().out
    assert "BLOCKER: resume settings do not match" in out
    assert "max_seq_length" in out


def test_run_full_training_resume_compatibility_passes_when_settings_match(tmp_path, capsys, monkeypatch):
    """When the checkpoint's source run_config.json matches the current
    canonical settings, resume-compatibility validation must PASS and the
    run must proceed past it (to the next, GPU-requiring step, which we
    intercept by monkeypatching QLoraBackend to avoid needing torch)."""
    module = _load_runner_module()
    cfg = load_train_config(TRAIN_CONFIG_PATH)
    examples = [_example(i) for i in range(6067)]
    schedule = compute_optimizer_step_schedule(
        len(examples), cfg.optimization.per_device_train_batch_size, cfg.optimization.gradient_accumulation_steps, 2
    )

    source_run_dir = tmp_path / "runs" / "phase5c-session-1"
    checkpoint_dir = source_run_dir / "checkpoint" / "checkpoint-400"
    checkpoint_dir.mkdir(parents=True)

    train_path = tmp_path / "train.jsonl"
    train_path.write_text("{}", encoding="utf-8")
    train_file_sha256 = module.file_sha256(train_path)

    source_run_config = {
        "model_id": cfg.model.id,
        "max_seq_length": cfg.sequence.max_seq_length,
        "quantization": {
            "load_in_4bit": cfg.runtime.load_in_4bit,
            "quant_type": cfg.runtime.quantization,
            "double_quant": cfg.runtime.double_quant,
            "compute_dtype": cfg.runtime.compute_dtype,
        },
        "lora": {
            "r": cfg.lora.r,
            "alpha": cfg.lora.alpha,
            "dropout": cfg.lora.dropout,
            "target_modules": list(cfg.lora.target_modules),
            "task_type": cfg.lora.task_type,
        },
        "canonical_num_train_epochs": 2,
        "canonical_total_optimizer_steps": schedule["total_optimizer_steps"],
        "train_file_sha256": train_file_sha256,
    }
    (source_run_dir / "run_config.json").write_text(json.dumps(source_run_config), encoding="utf-8")

    run_dir = tmp_path / "runs" / "phase5c-session-2"

    class _FakeBackend:
        def __init__(self, cfg):
            pass

        def load_for_training(self):
            raise RuntimeError("stopped intentionally right after the compatibility check passed")

    # run_full_training imports QLoraBackend lazily inside itself
    # (`from localsql.train.qlora_backend import ... QLoraBackend`), so
    # patch the class actually used: localsql.train.qlora_backend.QLoraBackend.
    import localsql.train.qlora_backend as qlora_backend_module

    monkeypatch.setattr(qlora_backend_module, "QLoraBackend", _FakeBackend)

    # The fake backend deliberately raises past the compatibility check --
    # calling run_full_training directly (not through main()'s BLOCKER
    # wrapper) means that RuntimeError propagates as-is, which is exactly
    # what proves the compatibility check itself did not block.
    with pytest.raises(RuntimeError, match="stopped intentionally right after the compatibility check passed"):
        module.run_full_training(
            cfg, examples, run_dir, train_path, 2, None, True, None, None, str(checkpoint_dir), None
        )

    out = capsys.readouterr().out
    assert "Resume-compatibility check passed" in out
    assert "BLOCKER: resume settings do not match" not in out


def test_run_full_training_missing_source_run_config_proceeds_with_note(tmp_path, capsys):
    """Requirement: validate 'where validation is practical' -- a missing
    source run_config.json (e.g. a foreign/older checkpoint) must not
    itself be a hard BLOCKER; it should proceed with a note, since there
    is genuinely nothing to validate against."""
    module = _load_runner_module()
    cfg = load_train_config(TRAIN_CONFIG_PATH)
    checkpoint_dir = tmp_path / "runs" / "some-other-run" / "checkpoint" / "checkpoint-1"
    checkpoint_dir.mkdir(parents=True)  # no run_config.json alongside it

    run_dir = tmp_path / "runs" / "phase5c-session-2"
    train_path = tmp_path / "train.jsonl"
    train_path.write_text("{}", encoding="utf-8")
    examples = [_example(i) for i in range(6067)]

    class _FakeBackend:
        def __init__(self, cfg):
            pass

        def load_for_training(self):
            raise RuntimeError("stopped intentionally after the missing-run_config note")

    import localsql.train.qlora_backend as qlora_backend_module

    monkeypatch_target = qlora_backend_module.QLoraBackend
    qlora_backend_module.QLoraBackend = _FakeBackend
    try:
        with pytest.raises(RuntimeError, match="stopped intentionally after the missing-run_config note"):
            module.run_full_training(
                cfg, examples, run_dir, train_path, 2, None, True, None, None, str(checkpoint_dir), None
            )
    finally:
        qlora_backend_module.QLoraBackend = monkeypatch_target

    out = capsys.readouterr().out
    assert "not found -- cannot validate resume compatibility" in out


def test_run_full_training_records_train_file_sha256_and_provenance_fields_in_source():
    """Structural guard: run_config/summary construction must include
    train_file/train_file_sha256/source_revision/source_revision_origin --
    the same fields scripts/export_checkpoint.py reads -- so full-training
    runs remain exportable without any export_checkpoint.py changes."""
    source = RUNNER_PATH.read_text(encoding="utf-8")
    for expected in (
        '"train_file": str(train_path)',
        '"train_file_sha256": train_file_sha256',
        '"source_revision": source_revision_info["source_revision"]',
        '"source_revision_origin": source_revision_info["source_revision_origin"]',
        '"canonical_num_train_epochs": num_train_epochs',
        '"canonical_total_optimizer_steps": schedule["total_optimizer_steps"]',
        '"starting_global_step": train_result["starting_global_step"]',
        '"final_global_step": train_result["global_step"]',
        '"session_end_reason": session_end_reason',
        '"resumed_from_checkpoint": train_result["resumed_from_checkpoint"]',
    ):
        assert expected in source, f"missing: {expected}"


def test_run_full_training_never_passes_max_steps_to_backend():
    """Structural guard for the CRITICAL requirement: this runner must
    call backend.train_smoke with num_train_epochs (never max_steps),
    otherwise TrainingArguments' max_steps would silently override
    num_train_epochs and break the fixed canonical horizon."""
    source = RUNNER_PATH.read_text(encoding="utf-8")
    call_block = source.split("train_result = backend.train_smoke(", 1)[1].split(")\n", 1)[0]
    assert "num_train_epochs=num_train_epochs" in call_block
    assert "max_steps" not in call_block


def test_main_parser_exposes_stop_after_global_step_and_num_train_epochs_flags():
    source = RUNNER_PATH.read_text(encoding="utf-8")
    assert '"--stop-after-global-step"' in source
    assert '"--num-train-epochs"' in source
    assert '"--resume-from-checkpoint"' in source
    assert '"--source-revision"' in source
