"""Phase 5B Task 6: durable checkpoint export (reliability-hardened pass).

Packages a full-state HF `Trainer` checkpoint (LoRA adapter + optimizer +
LR scheduler + RNG + `trainer_state.json`) plus the exact config/dataset/
provenance needed to resume it in a fresh environment, into one
self-contained directory with a SHA256 manifest. Never assumes any
particular path (e.g. `/kaggle/working`) is permanent -- the export is
meant to be copied somewhere durable (Kaggle Datasets, Drive, local disk,
...) and later extracted into any compatible environment.

This export exists specifically so an interrupted multi-hour run cannot
force a restart from step zero -- so it FAILS CLOSED rather than
packaging something that merely looks resumable:

1. `validate_checkpoint_completeness` requires every HF Trainer/PEFT
   resumable-state category (model/adapter, optimizer, LR scheduler,
   trainer_state/global step, RNG state, training arguments) to actually
   be present, tolerating the filename variants our pinned stack can
   produce (e.g. `adapter_model.safetensors` or the legacy
   `adapter_model.bin`) -- `IncompleteCheckpointError` if any category is
   missing, and `build_export` never creates a package on that failure.
2. `verify_training_data_snapshot` confirms the exact training JSONL a
   run used is packaged byte-for-byte and its SHA256 matches what that
   run recorded at training time (`run_config.json`'s
   `train_file_sha256`) -- `TrainingDataMismatchError` on a mismatch,
   since resuming Trainer state against a different/reordered dataset is
   unsafe.

Explicitly excludes full base-model weights. This project's checkpoints
are PEFT/LoRA-only -- `Trainer.save_model()` on a `PeftModel` saves only
the adapter, never the 4-bit base weights -- but this module additionally
allow-lists expected checkpoint filenames and refuses anything oversized
or base-model-shaped as a defensive safety net, rather than trusting that
assumption silently.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

# Expected files inside a full-state HF Trainer checkpoint of a PEFT/LoRA
# model.
ALLOWED_CHECKPOINT_FILENAMES = {
    "adapter_model.safetensors",
    "adapter_model.bin",
    "adapter_config.json",
    "optimizer.pt",
    "optimizer.bin",
    "scheduler.pt",
    "rng_state.pth",
    "rng_state_0.pth",
    "trainer_state.json",
    "training_args.bin",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "scaler.pt",
    "chat_template.jinja",
    "README.md",
}

# Filename prefixes that would indicate full (non-adapter) base model
# weights leaking into the checkpoint -- refused outright even if present,
# rather than silently including multi-GB base weights in the export.
DISALLOWED_BASE_MODEL_PREFIXES = ("pytorch_model", "model-", "model.safetensors")

MAX_SINGLE_FILE_MB = 500.0  # defensive cap; a LoRA r16 adapter is a few MB

# Required HF Trainer/PEFT resumable-state categories -- every one must be
# present, tolerating the filename variants our pinned stack can produce,
# or the checkpoint is not actually resumable and export must fail closed.
REQUIRED_CHECKPOINT_STATE_CATEGORIES: dict[str, tuple[str, ...]] = {
    "model_adapter_state": ("adapter_model.safetensors", "adapter_model.bin"),
    "optimizer_state": ("optimizer.pt", "optimizer.bin"),
    "lr_scheduler_state": ("scheduler.pt",),
    "trainer_state": ("trainer_state.json",),
    "rng_state": ("rng_state.pth", "rng_state_0.pth"),
    "training_arguments": ("training_args.bin",),
}


class IncompleteCheckpointError(RuntimeError):
    """Raised when a checkpoint is missing one or more required
    resumable-state categories -- fail closed, never package a checkpoint
    advertised as resumable when it isn't."""


class TrainingDataMismatchError(RuntimeError):
    """Raised when the training JSONL being packaged does not match the
    SHA256 the source run recorded at training time -- resuming Trainer
    state against a different/reordered dataset is unsafe."""


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_latest_checkpoint(checkpoint_root: Path) -> Path | None:
    """Highest-numbered `checkpoint-<N>` directory under `checkpoint_root`,
    or None if none exist. Compared numerically, not lexicographically, so
    `checkpoint-9` never sorts after `checkpoint-10`.
    """
    if not checkpoint_root.exists():
        return None
    candidates = []
    for child in checkpoint_root.iterdir():
        if child.is_dir() and child.name.startswith("checkpoint-"):
            suffix = child.name[len("checkpoint-"):]
            if suffix.isdigit():
                candidates.append((int(suffix), child))
    if not candidates:
        return None
    return max(candidates, key=lambda pair: pair[0])[1]


def validate_checkpoint_completeness(checkpoint_dir: Path) -> dict[str, str]:
    """Verify `checkpoint_dir` contains every required HF Trainer/PEFT
    resumable-state category (see `REQUIRED_CHECKPOINT_STATE_CATEGORIES`).

    Returns `{category: matched_filename}`. Raises
    `IncompleteCheckpointError` naming every missing category (and which
    filename variants were searched for) if any are absent.
    """
    if not checkpoint_dir.exists():
        raise IncompleteCheckpointError(f"checkpoint directory does not exist: {checkpoint_dir}")
    present_files = {p.name for p in checkpoint_dir.iterdir() if p.is_file()}
    found: dict[str, str] = {}
    missing: dict[str, tuple[str, ...]] = {}
    for category, candidates in REQUIRED_CHECKPOINT_STATE_CATEGORIES.items():
        match = next((c for c in candidates if c in present_files), None)
        if match:
            found[category] = match
        else:
            missing[category] = candidates
    if missing:
        raise IncompleteCheckpointError(
            f"checkpoint at {checkpoint_dir} is missing required resumable-state categories: "
            f"{missing} -- refusing to export a package that would be advertised as resumable but isn't. "
            "The empirical Kaggle resume test will confirm the exact filenames our pinned stack produces; "
            "if a new HF/PEFT version uses a different filename, add it to REQUIRED_CHECKPOINT_STATE_CATEGORIES."
        )
    return found


def verify_training_data_snapshot(train_file: Path, expected_sha256: str | None) -> str:
    """Verify the exact training JSONL a run used is available and (when
    `expected_sha256` -- typically `run_config.json`'s `train_file_sha256`
    -- is known) matches it byte-for-byte. Returns the file's actual
    SHA256. Raises `TrainingDataMismatchError` on a mismatch -- resuming
    Trainer state against a changed/reordered dataset is unsafe, so this
    fails closed rather than silently packaging a different file.
    """
    if not train_file.exists():
        raise TrainingDataMismatchError(f"training data file not found: {train_file}")
    actual_sha256 = file_sha256(train_file)
    if expected_sha256 and actual_sha256 != expected_sha256:
        raise TrainingDataMismatchError(
            f"training data file {train_file} has SHA256 {actual_sha256}, but the source run recorded "
            f"{expected_sha256} at training time -- the file has changed since training (or the wrong "
            "file was passed). Refusing to package a training-data snapshot that would not reproduce "
            "the run's actual training data."
        )
    return actual_sha256


def classify_checkpoint_files(checkpoint_dir: Path) -> tuple[list[Path], list[dict]]:
    """Split a checkpoint directory's files into (to_copy, excluded_report).

    `excluded_report` entries describe anything skipped and why -- nothing
    is ever silently dropped without a recorded reason.
    """
    to_copy: list[Path] = []
    excluded: list[dict] = []
    for path in sorted(checkpoint_dir.rglob("*")):
        if path.is_dir():
            continue
        rel = path.relative_to(checkpoint_dir)
        name = path.name
        size_mb = path.stat().st_size / (1024 * 1024)
        if any(name.startswith(p) for p in DISALLOWED_BASE_MODEL_PREFIXES):
            excluded.append(
                {"path": str(rel), "reason": "looks like full base model weights -- refused", "size_mb": round(size_mb, 2)}
            )
            continue
        if size_mb > MAX_SINGLE_FILE_MB:
            excluded.append(
                {"path": str(rel), "reason": f"exceeds {MAX_SINGLE_FILE_MB}MB safety cap", "size_mb": round(size_mb, 2)}
            )
            continue
        if name not in ALLOWED_CHECKPOINT_FILENAMES:
            excluded.append(
                {"path": str(rel), "reason": "not on the expected-checkpoint-file allowlist", "size_mb": round(size_mb, 2)}
            )
            continue
        to_copy.append(path)
    return to_copy, excluded


def build_export(
    checkpoint_dir: Path,
    export_dir: Path,
    extra_files: dict[str, Path],
    resume_instructions: str,
    provenance: dict | None = None,
) -> dict:
    """Copy an allow-listed, completeness-VALIDATED checkpoint's files
    plus `extra_files` (published relative name -> source path) into
    `export_dir`, and write a SHA256 manifest + resume instructions.
    Returns the manifest dict.

    Fails closed via `validate_checkpoint_completeness` BEFORE creating
    `export_dir` or copying anything -- an incomplete checkpoint produces
    no partial export directory at all, not just a warning.

    Refuses to overwrite an existing `export_dir` -- never silently
    clobbers a prior export. `provenance`, if given, is embedded verbatim
    into `MANIFEST.json` (environment/source-revision fields, Task 4).
    """
    validate_checkpoint_completeness(checkpoint_dir)  # raises before any I/O below

    if export_dir.exists():
        raise FileExistsError(f"export directory already exists, refusing to overwrite: {export_dir}")
    checkpoint_out = export_dir / "checkpoint"
    checkpoint_out.mkdir(parents=True)

    to_copy, excluded = classify_checkpoint_files(checkpoint_dir)
    if not to_copy:
        raise AssertionError(f"no checkpoint files passed the allowlist in {checkpoint_dir} -- nothing to export")

    manifest_files: dict[str, str] = {}
    for src in to_copy:
        rel = src.relative_to(checkpoint_dir)
        dst = checkpoint_out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        manifest_files[f"checkpoint/{rel.as_posix()}"] = file_sha256(dst)

    for published_name, src in extra_files.items():
        dst = export_dir / published_name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        manifest_files[published_name] = file_sha256(dst)

    resume_path = export_dir / "RESUME_INSTRUCTIONS.md"
    resume_path.write_text(resume_instructions, encoding="utf-8")
    manifest_files["RESUME_INSTRUCTIONS.md"] = file_sha256(resume_path)

    manifest = {
        "excluded_checkpoint_files": excluded,
        "provenance": provenance or {},
        "sha256": manifest_files,
    }
    manifest_path = export_dir / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def render_resume_instructions(
    *,
    source_run_id: str,
    checkpoint_name: str,
    global_step: int | None,
    model_id: str,
    resolved_revision: str | None,
    train_file: str | None,
    max_seq_length: int | None,
    save_steps: int | None,
    save_total_limit: int | None,
    source_revision: str | None,
    source_revision_origin: str | None = None,
) -> str:
    revision_note = (
        f"`{source_revision}` (resolved via {source_revision_origin})"
        if source_revision
        else "unavailable (not a git repository, git not found, and no explicit --source-revision/SOURCE_REVISION "
        "file at export time)"
    )
    return f"""# LocalSQL durable checkpoint export

Source run: `{source_run_id}` / `{checkpoint_name}`
Trainer global_step at export time: {global_step if global_step is not None else "unknown"}
Model: `{model_id}` (resolved revision: `{resolved_revision or "unknown"}`)
Candidate max_seq_length: {max_seq_length if max_seq_length is not None else "unknown"}
Source repo revision: {revision_note}

## What is in this export

- `checkpoint/` -- the VALIDATED resumable Trainer checkpoint (LoRA
  adapter, optimizer, LR scheduler, RNG state, `trainer_state.json`,
  training arguments -- every category confirmed present before this
  package was built; see `MANIFEST.json`). Full base model weights are
  deliberately NOT included -- re-download/load the base model
  (`{model_id}`, revision `{resolved_revision or "resolve at load time"}`)
  in 4-bit as usual; only the adapter and training state are restored.
- `training_data.jsonl` -- the EXACT training JSONL this run used,
  byte-for-byte, SHA256-verified against what the run recorded at
  training time (see `MANIFEST.json`). Resuming Trainer state against a
  different or reordered dataset is unsafe, so this is packaged
  explicitly rather than assumed to still exist at its original path.
- `train_config.yaml` -- the exact QLoRA config this checkpoint was
  produced under.
- `candidate_policy.json` -- the candidate training-data policy and
  provenance (adaptive per-database schema compaction decision, DB lists,
  SHA256 of original/candidate files) this checkpoint's training data was
  drawn from.
- `run_config.json` -- the source run's exact configuration snapshot
  (train file path + SHA256, LoRA/quantization/optimization settings,
  `max_seq_length`, prior `save_steps`/`save_total_limit`/
  `resume_from_checkpoint`).
- `MANIFEST.json` -- SHA256 of every file in this export, environment/
  source-revision provenance, and any checkpoint files that were found
  but excluded (and why).

## How to resume in a fresh environment

1. Extract this export anywhere -- durable storage is not required to
   already exist at a specific path -- point `--resume-from-checkpoint`
   at wherever you extracted `checkpoint/` to, and `--input` at wherever
   you extracted `training_data.jsonl` to.
2. Ensure the environment has `uv sync --group model --group train` and
   the same repo checked out at revision {revision_note}.
3. Run, with a NEW `--run-id` (never reuse `{source_run_id}` -- this
   runner refuses to overwrite a run directory that already has a
   `summary.json`):

```
uv run python scripts/run_qlora_smoke.py --run-id <new-run-id> \\
    --input <path-to-extracted>/training_data.jsonl \\
    --max-steps <higher-than-{global_step if global_step is not None else "prior"}-global-step> \\
    --save-steps {save_steps if save_steps is not None else "<N>"} --save-total-limit {save_total_limit if save_total_limit is not None else "<N>"} \\
    --resume-from-checkpoint <path-to-extracted>/checkpoint
```

(Original `--input` path for reference, if extracting `training_data.jsonl`
elsewhere: `{train_file or "see run_config.json's train_file"}`.)

The resumed run's `summary.json` will record `starting_global_step`,
`resumed_from_checkpoint`, and the final `global_step` -- confirming the
resume actually advanced training rather than restarting it.
"""
