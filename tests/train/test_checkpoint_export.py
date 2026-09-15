"""Contract tests for Phase 5B Task 6: durable checkpoint export
(reliability-hardened pass).

CPU/offline, synthetic checkpoint directories only -- no model/CUDA.
Covers the fail-closed guarantees added in the hardening pass: a
checkpoint missing any required resumable-state category must be
rejected before any export directory is created, and the exact training
data snapshot must be verified byte-for-byte against the source run's
recorded SHA256.
"""

import json

import pytest

from localsql.train.checkpoint_export import (
    IncompleteCheckpointError,
    REQUIRED_CHECKPOINT_STATE_CATEGORIES,
    TrainingDataMismatchError,
    build_export,
    classify_checkpoint_files,
    file_sha256,
    find_latest_checkpoint,
    render_resume_instructions,
    validate_checkpoint_completeness,
    verify_training_data_snapshot,
)


def _write(path, content: bytes = b"x"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _make_fake_checkpoint(root, name="checkpoint-2", extra_files=None, big_file_mb=None, omit=()):
    ckpt = root / "checkpoint" / name
    full = {
        "adapter_model.safetensors": b"lora-weights",
        "adapter_config.json": b"{}",
        "optimizer.pt": b"opt",
        "scheduler.pt": b"sched",
        "rng_state.pth": b"rng",
        "trainer_state.json": json.dumps({"global_step": 2}).encode(),
        "training_args.bin": b"args",
    }
    for name_ in omit:
        full.pop(name_, None)
    for name_, content in full.items():
        _write(ckpt / name_, content)
    for name_, content in (extra_files or {}).items():
        _write(ckpt / name_, content)
    if big_file_mb:
        _write(ckpt / "pytorch_model.bin", b"0" * (big_file_mb * 1024 * 1024))
    return ckpt


# --- find_latest_checkpoint ---


def test_find_latest_checkpoint_picks_highest_numeric_step(tmp_path):
    root = tmp_path / "checkpoint"
    (root / "checkpoint-2").mkdir(parents=True)
    (root / "checkpoint-10").mkdir(parents=True)  # must not sort after checkpoint-2 lexicographically
    (root / "checkpoint-4").mkdir(parents=True)

    latest = find_latest_checkpoint(root)

    assert latest.name == "checkpoint-10"


def test_find_latest_checkpoint_returns_none_when_absent(tmp_path):
    assert find_latest_checkpoint(tmp_path / "does_not_exist") is None


def test_find_latest_checkpoint_ignores_non_checkpoint_dirs(tmp_path):
    root = tmp_path / "checkpoint"
    (root / "checkpoint-1").mkdir(parents=True)
    (root / "scratch").mkdir(parents=True)

    latest = find_latest_checkpoint(root)

    assert latest.name == "checkpoint-1"


# --- classify_checkpoint_files ---


def test_classify_checkpoint_files_accepts_expected_files(tmp_path):
    ckpt = _make_fake_checkpoint(tmp_path)
    to_copy, excluded = classify_checkpoint_files(ckpt)
    names = {p.name for p in to_copy}
    assert names == {
        "adapter_model.safetensors",
        "adapter_config.json",
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
        "trainer_state.json",
        "training_args.bin",
    }
    assert excluded == []


def test_classify_checkpoint_files_refuses_base_model_weight_shaped_files(tmp_path):
    ckpt = _make_fake_checkpoint(tmp_path, extra_files={"pytorch_model-00001-of-00002.bin": b"weights"})
    to_copy, excluded = classify_checkpoint_files(ckpt)
    assert all(p.name != "pytorch_model-00001-of-00002.bin" for p in to_copy)
    reasons = {e["path"]: e["reason"] for e in excluded}
    assert "refused" in reasons["pytorch_model-00001-of-00002.bin"]


def test_classify_checkpoint_files_rejects_oversized_files(tmp_path):
    ckpt = _make_fake_checkpoint(tmp_path, big_file_mb=1)
    from localsql.train import checkpoint_export as mod

    original_cap = mod.MAX_SINGLE_FILE_MB
    mod.MAX_SINGLE_FILE_MB = 0.5
    try:
        to_copy, excluded = classify_checkpoint_files(ckpt)
    finally:
        mod.MAX_SINGLE_FILE_MB = original_cap
    assert all("pytorch_model.bin" != p.name for p in to_copy)


def test_classify_checkpoint_files_excludes_unexpected_filenames(tmp_path):
    ckpt = _make_fake_checkpoint(tmp_path, extra_files={"mystery.bin": b"?"})
    to_copy, excluded = classify_checkpoint_files(ckpt)
    assert all(p.name != "mystery.bin" for p in to_copy)
    assert any(e["path"] == "mystery.bin" for e in excluded)


def test_classify_checkpoint_files_accepts_legacy_adapter_bin_and_optimizer_bin_variants(tmp_path):
    """Support the actual filenames our pinned/current stack can produce,
    not just one extension per category."""
    ckpt = tmp_path / "checkpoint" / "checkpoint-1"
    _write(ckpt / "adapter_model.bin", b"legacy-lora-weights")
    _write(ckpt / "optimizer.bin", b"legacy-opt")
    to_copy, excluded = classify_checkpoint_files(ckpt)
    names = {p.name for p in to_copy}
    assert "adapter_model.bin" in names
    assert "optimizer.bin" in names


# --- validate_checkpoint_completeness (fail-closed) ---


def test_validate_checkpoint_completeness_accepts_full_checkpoint(tmp_path):
    ckpt = _make_fake_checkpoint(tmp_path)
    found = validate_checkpoint_completeness(ckpt)
    assert set(found.keys()) == set(REQUIRED_CHECKPOINT_STATE_CATEGORIES.keys())


def test_validate_checkpoint_completeness_rejects_missing_model_adapter_state(tmp_path):
    ckpt = _make_fake_checkpoint(tmp_path, omit=("adapter_model.safetensors",))
    with pytest.raises(IncompleteCheckpointError):
        validate_checkpoint_completeness(ckpt)


def test_validate_checkpoint_completeness_rejects_missing_optimizer_state(tmp_path):
    ckpt = _make_fake_checkpoint(tmp_path, omit=("optimizer.pt",))
    with pytest.raises(IncompleteCheckpointError):
        validate_checkpoint_completeness(ckpt)


def test_validate_checkpoint_completeness_rejects_missing_scheduler_state(tmp_path):
    ckpt = _make_fake_checkpoint(tmp_path, omit=("scheduler.pt",))
    with pytest.raises(IncompleteCheckpointError):
        validate_checkpoint_completeness(ckpt)


def test_validate_checkpoint_completeness_rejects_missing_trainer_state(tmp_path):
    ckpt = _make_fake_checkpoint(tmp_path, omit=("trainer_state.json",))
    with pytest.raises(IncompleteCheckpointError):
        validate_checkpoint_completeness(ckpt)


def test_validate_checkpoint_completeness_rejects_missing_rng_state(tmp_path):
    ckpt = _make_fake_checkpoint(tmp_path, omit=("rng_state.pth",))
    with pytest.raises(IncompleteCheckpointError):
        validate_checkpoint_completeness(ckpt)


def test_validate_checkpoint_completeness_rejects_missing_training_arguments(tmp_path):
    ckpt = _make_fake_checkpoint(tmp_path, omit=("training_args.bin",))
    with pytest.raises(IncompleteCheckpointError):
        validate_checkpoint_completeness(ckpt)


def test_validate_checkpoint_completeness_accepts_legacy_filename_variants(tmp_path):
    ckpt = _make_fake_checkpoint(tmp_path, omit=("adapter_model.safetensors", "optimizer.pt"))
    _write(ckpt / "adapter_model.bin", b"legacy")
    _write(ckpt / "optimizer.bin", b"legacy")
    found = validate_checkpoint_completeness(ckpt)
    assert found["model_adapter_state"] == "adapter_model.bin"
    assert found["optimizer_state"] == "optimizer.bin"


def test_validate_checkpoint_completeness_error_names_every_missing_category(tmp_path):
    ckpt = _make_fake_checkpoint(tmp_path, omit=("optimizer.pt", "scheduler.pt"))
    with pytest.raises(IncompleteCheckpointError) as exc_info:
        validate_checkpoint_completeness(ckpt)
    message = str(exc_info.value)
    assert "optimizer_state" in message
    assert "lr_scheduler_state" in message


# --- verify_training_data_snapshot ---


def test_verify_training_data_snapshot_returns_sha256_when_no_expected_given(tmp_path):
    train_file = tmp_path / "train.jsonl"
    train_file.write_bytes(b'{"example_id": "a"}\n')
    sha = verify_training_data_snapshot(train_file, expected_sha256=None)
    assert sha == file_sha256(train_file)


def test_verify_training_data_snapshot_passes_when_sha256_matches(tmp_path):
    train_file = tmp_path / "train.jsonl"
    train_file.write_bytes(b'{"example_id": "a"}\n')
    expected = file_sha256(train_file)
    assert verify_training_data_snapshot(train_file, expected_sha256=expected) == expected


def test_verify_training_data_snapshot_raises_on_mismatch(tmp_path):
    train_file = tmp_path / "train.jsonl"
    train_file.write_bytes(b'{"example_id": "a"}\n')
    with pytest.raises(TrainingDataMismatchError):
        verify_training_data_snapshot(train_file, expected_sha256="0" * 64)


def test_verify_training_data_snapshot_raises_when_file_missing(tmp_path):
    with pytest.raises(TrainingDataMismatchError):
        verify_training_data_snapshot(tmp_path / "does_not_exist.jsonl", expected_sha256=None)


# --- build_export ---


def test_build_export_copies_allowed_files_and_writes_manifest(tmp_path):
    ckpt = _make_fake_checkpoint(tmp_path)
    export_dir = tmp_path / "export"
    train_config = tmp_path / "train.yaml"
    _write(train_config, b"model: {}")

    manifest = build_export(ckpt, export_dir, {"train_config.yaml": train_config}, "resume instructions text")

    assert (export_dir / "checkpoint" / "adapter_model.safetensors").exists()
    assert (export_dir / "train_config.yaml").exists()
    assert (export_dir / "RESUME_INSTRUCTIONS.md").read_text(encoding="utf-8") == "resume instructions text"
    manifest_on_disk = json.loads((export_dir / "MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest_on_disk == manifest
    assert "checkpoint/adapter_model.safetensors" in manifest["sha256"]
    assert "train_config.yaml" in manifest["sha256"]


def test_build_export_manifest_sha256_matches_copied_file_content(tmp_path):
    ckpt = _make_fake_checkpoint(tmp_path)
    export_dir = tmp_path / "export"

    manifest = build_export(ckpt, export_dir, {}, "resume")

    copied = export_dir / "checkpoint" / "adapter_model.safetensors"
    assert manifest["sha256"]["checkpoint/adapter_model.safetensors"] == file_sha256(copied)


def test_build_export_excludes_base_model_weights_and_records_them(tmp_path):
    ckpt = _make_fake_checkpoint(tmp_path, extra_files={"pytorch_model-00001-of-00002.bin": b"weights"})
    export_dir = tmp_path / "export"

    manifest = build_export(ckpt, export_dir, {}, "resume")

    assert not (export_dir / "checkpoint" / "pytorch_model-00001-of-00002.bin").exists()
    assert any(e["path"] == "pytorch_model-00001-of-00002.bin" for e in manifest["excluded_checkpoint_files"])


def test_build_export_refuses_to_overwrite_existing_export_dir(tmp_path):
    ckpt = _make_fake_checkpoint(tmp_path)
    export_dir = tmp_path / "export"
    build_export(ckpt, export_dir, {}, "resume")

    with pytest.raises(FileExistsError):
        build_export(ckpt, export_dir, {}, "resume again")


def test_build_export_raises_if_nothing_passes_allowlist(tmp_path):
    ckpt = tmp_path / "checkpoint" / "checkpoint-1"
    _write(ckpt / "mystery.bin", b"?")
    export_dir = tmp_path / "export"

    with pytest.raises(IncompleteCheckpointError):
        build_export(ckpt, export_dir, {}, "resume")


def test_build_export_does_not_mutate_source_checkpoint(tmp_path):
    ckpt = _make_fake_checkpoint(tmp_path)
    before = {p.name: p.read_bytes() for p in ckpt.iterdir()}
    export_dir = tmp_path / "export"

    build_export(ckpt, export_dir, {}, "resume")

    after = {p.name: p.read_bytes() for p in ckpt.iterdir()}
    assert before == after


def test_build_export_embeds_provenance_dict_in_manifest(tmp_path):
    ckpt = _make_fake_checkpoint(tmp_path)
    export_dir = tmp_path / "export"
    provenance = {
        "training_time": {"torch_version": "2.4.0", "accelerate_version": "1.0.0"},
        "export_time": {"source_revision": "abc123", "source_revision_origin": "explicit_arg"},
    }

    manifest = build_export(ckpt, export_dir, {}, "resume", provenance=provenance)

    assert manifest["provenance"] == provenance
    on_disk = json.loads((export_dir / "MANIFEST.json").read_text(encoding="utf-8"))
    assert on_disk["provenance"] == provenance


def test_build_export_fails_closed_on_incomplete_checkpoint_before_creating_export_dir(tmp_path):
    """FAIL CLOSED requirement: no export directory (partial or otherwise)
    is ever created when the checkpoint is missing required state."""
    ckpt = _make_fake_checkpoint(tmp_path, omit=("optimizer.pt",))
    export_dir = tmp_path / "export"

    with pytest.raises(IncompleteCheckpointError):
        build_export(ckpt, export_dir, {}, "resume")

    assert not export_dir.exists()


def test_build_export_packages_exact_training_data_and_sha256_matches_source(tmp_path):
    """Exact-dataset snapshot packaging: the training JSONL is packaged
    byte-for-byte and its manifest SHA256 matches the source file."""
    ckpt = _make_fake_checkpoint(tmp_path)
    export_dir = tmp_path / "export"
    train_file = tmp_path / "training_source" / "train.jsonl"
    _write(train_file, b'{"example_id": "a", "prompt": "p", "completion": "c"}\n')

    manifest = build_export(ckpt, export_dir, {"training_data.jsonl": train_file}, "resume")

    packaged = export_dir / "training_data.jsonl"
    assert packaged.read_bytes() == train_file.read_bytes()
    assert manifest["sha256"]["training_data.jsonl"] == file_sha256(train_file)


# --- render_resume_instructions ---


def test_render_resume_instructions_includes_key_fields():
    text = render_resume_instructions(
        source_run_id="phase5b-resume-a",
        checkpoint_name="checkpoint-2",
        global_step=2,
        model_id="Qwen/Qwen3-4B-Instruct-2507",
        resolved_revision="cdbee75f17c01a7cc42f958dc650907174af0554",
        train_file="data/certification/longest_16.jsonl",
        max_seq_length=4096,
        save_steps=1,
        save_total_limit=2,
        source_revision="abc123",
        source_revision_origin="git_rev_parse",
    )
    assert "phase5b-resume-a" in text
    assert "checkpoint-2" in text
    assert "abc123" in text
    assert "git_rev_parse" in text
    assert "--resume-from-checkpoint" in text
    assert "training_data.jsonl" in text
    assert "Qwen/Qwen3-4B-Instruct-2507" in text


def test_render_resume_instructions_handles_missing_source_revision():
    text = render_resume_instructions(
        source_run_id="r",
        checkpoint_name="checkpoint-1",
        global_step=1,
        model_id="m",
        resolved_revision=None,
        train_file=None,
        max_seq_length=None,
        save_steps=None,
        save_total_limit=None,
        source_revision=None,
        source_revision_origin="unavailable",
    )
    assert "unavailable" in text
