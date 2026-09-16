"""Phase 5C: canonical full-training schedule math and resume-compatibility
validation. Pure Python -- no torch/transformers import, so this stays
testable offline without CUDA.
"""

from __future__ import annotations

import math


def compute_optimizer_step_schedule(
    num_examples: int,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    num_train_epochs: int,
    num_devices: int = 1,
) -> dict:
    """Expected optimizer-step counts for a fixed training horizon, using
    the same arithmetic `transformers.Trainer` uses internally for a
    map-style dataset with `drop_last=False` (the default):

        steps_per_dataloader = ceil(num_examples / (per_device_train_batch_size * num_devices))
        steps_per_epoch      = ceil(steps_per_dataloader / gradient_accumulation_steps)
        total_optimizer_steps = steps_per_epoch * num_train_epochs

    This is an ESTIMATE mirroring Trainer's own step-counting formula, not
    a substitute for the real `trainer_state.json` a Kaggle run produces
    -- used here to (a) report the canonical expected horizon before any
    GPU run, and (b) as the canonical fixed horizon that
    `stop_after_global_step` and multi-session resume are validated
    against.
    """
    if num_examples <= 0:
        raise ValueError(f"num_examples must be positive, got {num_examples}")
    if per_device_train_batch_size <= 0 or gradient_accumulation_steps <= 0 or num_devices <= 0:
        raise ValueError("per_device_train_batch_size, gradient_accumulation_steps, and num_devices must be positive")
    if num_train_epochs <= 0:
        raise ValueError(f"num_train_epochs must be positive, got {num_train_epochs}")

    steps_per_dataloader = math.ceil(num_examples / (per_device_train_batch_size * num_devices))
    steps_per_epoch = math.ceil(steps_per_dataloader / gradient_accumulation_steps)
    total_optimizer_steps = steps_per_epoch * num_train_epochs

    return {
        "num_examples": num_examples,
        "per_device_train_batch_size": per_device_train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "num_devices": num_devices,
        "total_train_batch_size": per_device_train_batch_size * gradient_accumulation_steps * num_devices,
        "num_train_epochs": num_train_epochs,
        "steps_per_epoch": steps_per_epoch,
        "total_optimizer_steps": total_optimizer_steps,
    }


class StopAfterGlobalStepState:
    """Framework-independent decision rule for a graceful, checkpointed
    early stop at a specific optimizer step -- extracted from
    `qlora_backend.QLoraBackend.train_smoke`'s actual `TrainerCallback` so
    the "stop exactly at the requested step, and request a forced
    checkpoint save on that step" logic is unit-testable without a CUDA/
    transformers environment. The real callback delegates to
    `should_stop_and_save` for its decision; it does not duplicate this
    comparison.
    """

    def __init__(self, stop_step: int):
        self.stop_step = stop_step
        self.stopped_at_boundary = False

    def should_stop_and_save(self, global_step: int) -> bool:
        """True once `global_step` has reached `stop_step` -- the caller
        (the real TrainerCallback) is expected to set BOTH
        `control.should_training_stop = True` and
        `control.should_save = True` when this returns True, so a
        resumable checkpoint always exists at the boundary even if it
        doesn't land on a `save_steps` multiple."""
        if global_step >= self.stop_step:
            self.stopped_at_boundary = True
            return True
        return False


class ResumeCompatibilityError(RuntimeError):
    """Raised when a resume's canonical settings don't match the source
    checkpoint/run's recorded settings -- refuses to silently continue
    training under different settings than the checkpoint was produced
    under."""


# Fields compared between the current run's config and the checkpoint
# source run's recorded `run_config.json`, wherever both are available
# (i.e. "where validation is practical" -- an older/foreign run_config.json
# missing a field is not itself an error; only a PRESENT, DIFFERING value
# is).
CANONICAL_RESUME_FIELDS = (
    "model_id",
    "max_seq_length",
    "quantization",
    "lora",
    "canonical_num_train_epochs",
    "canonical_total_optimizer_steps",
    "train_file_sha256",
)


def validate_resume_compatibility(current_config: dict, source_run_config: dict) -> list[str]:
    """Compare `current_config` (the run about to start/resume) against
    `source_run_config` (the run_config.json of the run that produced the
    checkpoint being resumed from) across `CANONICAL_RESUME_FIELDS`.

    Returns a list of human-readable mismatch descriptions (empty if
    compatible). A field present in `source_run_config` but absent from
    `current_config` (or vice versa) is not itself flagged as a mismatch
    -- only two present-and-different values are, since not every field
    is guaranteed to exist on every historical run_config.json.
    """
    mismatches = []
    for field in CANONICAL_RESUME_FIELDS:
        if field in current_config and field in source_run_config:
            if current_config[field] != source_run_config[field]:
                mismatches.append(
                    f"{field}: current={current_config[field]!r} != checkpoint source={source_run_config[field]!r}"
                )
    return mismatches
