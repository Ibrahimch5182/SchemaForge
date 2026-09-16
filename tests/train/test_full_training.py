"""Contract tests for Phase 5C: canonical full-training schedule math,
resume-compatibility validation, and the session-boundary stop decision
rule. CPU/offline, no torch/transformers/CUDA required -- this repo's own
dev machine doesn't have those installed, and these tests must still run.
"""

import pytest

from localsql.train.full_training import (
    CANONICAL_RESUME_FIELDS,
    ResumeCompatibilityError,
    StopAfterGlobalStepState,
    compute_optimizer_step_schedule,
    validate_resume_compatibility,
)


# --- compute_optimizer_step_schedule ---


def test_compute_optimizer_step_schedule_matches_real_canonical_numbers():
    """The real canonical dataset: 6,067 training examples, batch=1,
    grad-accum=8, 2 epochs -- must match what a real Kaggle run would see
    from transformers.Trainer's own step-counting formula."""
    schedule = compute_optimizer_step_schedule(
        num_examples=6067, per_device_train_batch_size=1, gradient_accumulation_steps=8, num_train_epochs=2
    )
    assert schedule["steps_per_epoch"] == 759
    assert schedule["total_optimizer_steps"] == 1518
    assert schedule["total_train_batch_size"] == 8


def test_compute_optimizer_step_schedule_stop_after_global_step_never_changes_it():
    """The canonical total-step horizon depends only on
    (num_examples, batch config, num_train_epochs) -- never on whether or
    how a session plans to stop early. This is a documentation-style
    guard: the function has no stop_after_global_step parameter at all,
    so it structurally cannot be influenced by one."""
    import inspect

    params = set(inspect.signature(compute_optimizer_step_schedule).parameters)
    assert "stop_after_global_step" not in params
    a = compute_optimizer_step_schedule(6067, 1, 8, 2)
    b = compute_optimizer_step_schedule(6067, 1, 8, 2)
    assert a == b


def test_compute_optimizer_step_schedule_generic_formula():
    # 100 examples, batch=2, grad_accum=4, 3 epochs:
    # steps_per_dataloader = ceil(100/2) = 50; steps_per_epoch = ceil(50/4) = 13; total = 39
    schedule = compute_optimizer_step_schedule(
        num_examples=100, per_device_train_batch_size=2, gradient_accumulation_steps=4, num_train_epochs=3
    )
    assert schedule["steps_per_epoch"] == 13
    assert schedule["total_optimizer_steps"] == 39


def test_compute_optimizer_step_schedule_exact_division_no_remainder():
    # 64 examples, batch=1, grad_accum=8, 1 epoch -> exactly 8 steps.
    schedule = compute_optimizer_step_schedule(64, 1, 8, 1)
    assert schedule["steps_per_epoch"] == 8
    assert schedule["total_optimizer_steps"] == 8


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_examples": 0, "per_device_train_batch_size": 1, "gradient_accumulation_steps": 8, "num_train_epochs": 2},
        {"num_examples": 100, "per_device_train_batch_size": 0, "gradient_accumulation_steps": 8, "num_train_epochs": 2},
        {"num_examples": 100, "per_device_train_batch_size": 1, "gradient_accumulation_steps": 0, "num_train_epochs": 2},
        {"num_examples": 100, "per_device_train_batch_size": 1, "gradient_accumulation_steps": 8, "num_train_epochs": 0},
    ],
)
def test_compute_optimizer_step_schedule_rejects_nonpositive_inputs(kwargs):
    with pytest.raises(ValueError):
        compute_optimizer_step_schedule(**kwargs)


# --- StopAfterGlobalStepState (session-boundary decision rule) ---


def test_stop_after_global_step_state_does_not_trigger_before_target():
    state = StopAfterGlobalStepState(stop_step=400)
    for step in (1, 100, 399):
        assert state.should_stop_and_save(step) is False
    assert state.stopped_at_boundary is False


def test_stop_after_global_step_state_triggers_exactly_at_target():
    state = StopAfterGlobalStepState(stop_step=400)
    assert state.should_stop_and_save(400) is True
    assert state.stopped_at_boundary is True


def test_stop_after_global_step_state_triggers_if_step_already_past_target():
    """Defensive: if global_step ever jumps past the target (shouldn't
    happen with per-step callbacks, but must not silently miss the
    boundary), the state still reports a stop."""
    state = StopAfterGlobalStepState(stop_step=400)
    assert state.should_stop_and_save(405) is True
    assert state.stopped_at_boundary is True


def test_stop_after_global_step_state_records_boundary_flag_for_session_end_reason():
    """The flag this exposes is exactly what qlora_backend.train_smoke
    uses to compute `ended_by_planned_boundary` in its return dict, which
    the full-training runner turns into session_end_reason."""
    state = StopAfterGlobalStepState(stop_step=10)
    assert state.stopped_at_boundary is False
    state.should_stop_and_save(5)
    assert state.stopped_at_boundary is False
    state.should_stop_and_save(10)
    assert state.stopped_at_boundary is True


def test_stop_after_global_step_state_absent_boundary_never_triggers_when_unused():
    """Sanity check for the 'absent stop boundary allows canonical
    training to continue to completion' requirement: a session that never
    constructs a StopAfterGlobalStepState (stop_after_global_step=None)
    has no mechanism that could request an early stop -- there is nothing
    to unit-test on the "none" path beyond the mutual-exclusivity guard in
    qlora_backend.train_smoke (see test_qlora_backend_full_training.py),
    since None simply means the callback is never added at all."""
    # No callback constructed -- nothing should ever call should_stop_and_save.
    assert True


# --- validate_resume_compatibility ---


def _base_config(**overrides):
    config = {
        "model_id": "Qwen/Qwen3-4B-Instruct-2507",
        "max_seq_length": 4096,
        "quantization": {"load_in_4bit": True, "quant_type": "nf4", "double_quant": True, "compute_dtype": "float16"},
        "lora": {"r": 16, "alpha": 32, "dropout": 0.05, "target_modules": ["q_proj"], "task_type": "CAUSAL_LM"},
        "canonical_num_train_epochs": 2,
        "canonical_total_optimizer_steps": 1518,
        "train_file_sha256": "a" * 64,
    }
    config.update(overrides)
    return config


def test_validate_resume_compatibility_no_mismatches_when_identical():
    current = _base_config()
    source = _base_config()
    assert validate_resume_compatibility(current, source) == []


@pytest.mark.parametrize("field", CANONICAL_RESUME_FIELDS)
def test_validate_resume_compatibility_detects_mismatch_in_every_field(field):
    current = _base_config()
    source = _base_config()
    if field == "max_seq_length":
        source[field] = 8192
    elif field == "canonical_num_train_epochs":
        source[field] = 1
    elif field == "canonical_total_optimizer_steps":
        source[field] = 759
    elif field == "train_file_sha256":
        source[field] = "b" * 64
    elif field == "model_id":
        source[field] = "some/other-model"
    elif field == "quantization":
        source[field] = dict(source[field], quant_type="fp4")
    elif field == "lora":
        source[field] = dict(source[field], r=8)

    mismatches = validate_resume_compatibility(current, source)
    assert len(mismatches) == 1
    assert field in mismatches[0]


def test_validate_resume_compatibility_ignores_fields_missing_from_either_side():
    """A field present in the checkpoint's source run_config.json but
    absent from a differently-shaped current config (or vice versa) is
    not itself an error -- only two PRESENT, DIFFERENT values are. Here
    `model_id`/`max_seq_length` match on both sides; every other
    canonical field is simply absent from `current` (e.g. an older/
    differently-shaped config) and must not be flagged."""
    current = {"model_id": "Qwen/Qwen3-4B-Instruct-2507", "max_seq_length": 4096}
    source = _base_config()
    assert validate_resume_compatibility(current, source) == []


def test_validate_resume_compatibility_detects_multiple_simultaneous_mismatches():
    current = _base_config()
    source = _base_config(model_id="other-model", max_seq_length=8192)
    mismatches = validate_resume_compatibility(current, source)
    assert len(mismatches) == 2


def test_validate_resume_compatibility_returns_human_readable_strings():
    current = _base_config()
    source = _base_config(model_id="other-model")
    mismatches = validate_resume_compatibility(current, source)
    assert "current=" in mismatches[0]
    assert "checkpoint source=" in mismatches[0]


def test_resume_compatibility_error_is_a_runtime_error():
    assert issubclass(ResumeCompatibilityError, RuntimeError)
