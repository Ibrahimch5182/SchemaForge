import hashlib
import json

import pytest

from localsql.deploy.manifest import ManifestConflictError, build_manifest, write_manifest
from localsql.deploy.provenance import (
    ProvenanceError,
    artifact_record,
    validate_adapter,
    validate_base_snapshot,
)

from tests.deploy.fixtures import FAKE_WEIGHTS, fake_config, make_adapter_dir, real_config


def test_frozen_constants_in_config():
    cfg = real_config()
    assert cfg.adapter.weights_sha256 == "f7b78b3cb012219bdc9ef48ee2cf5a9105a9d395033da4f9c8a1af2f10ff34cc"
    assert cfg.model.revision == "cdbee75f17c01a7cc42f958dc650907174af0554"
    assert (cfg.adapter.expected_r, cfg.adapter.expected_lora_alpha) == (16, 32)


def test_exact_sha_accepted(tmp_path):
    prov = validate_adapter(make_adapter_dir(tmp_path), fake_config())
    assert prov.weights_sha256 == hashlib.sha256(FAKE_WEIGHTS).hexdigest()
    assert prov.r == 16 and prov.lora_alpha == 32


def test_wrong_sha_rejected(tmp_path):
    d = make_adapter_dir(tmp_path, weights=b"tampered")
    with pytest.raises(ProvenanceError, match="SHA256 mismatch"):
        validate_adapter(d, fake_config())


@pytest.mark.parametrize(
    "override,match",
    [
        ({"base_model_name_or_path": "Qwen/Other"}, "base_model_name_or_path"),
        ({"r": 8}, "r is"),
        ({"lora_alpha": 16}, "lora_alpha"),
    ],
)
def test_wrong_adapter_metadata_rejected(tmp_path, override, match):
    d = make_adapter_dir(tmp_path, **override)
    with pytest.raises(ProvenanceError, match=match):
        validate_adapter(d, fake_config())


def test_missing_artifacts_rejected(tmp_path):
    with pytest.raises(ProvenanceError, match="not found"):
        validate_adapter(tmp_path / "nope", fake_config())
    d = make_adapter_dir(tmp_path)
    (d / "adapter_model.safetensors").unlink()
    with pytest.raises(ProvenanceError, match="missing"):
        validate_adapter(d, fake_config())


def test_base_snapshot_revision_check(tmp_path):
    cfg = fake_config()
    good = tmp_path / cfg.model.revision
    good.mkdir()
    (good / "config.json").write_text("{}")
    assert validate_base_snapshot(good, cfg)["revision_evidence"] == "directory_name_matches_frozen_revision"

    bad = tmp_path / ("a" * 40)
    bad.mkdir()
    (bad / "config.json").write_text("{}")
    with pytest.raises(ProvenanceError, match="expected frozen"):
        validate_base_snapshot(bad, cfg)

    plain = tmp_path / "qwen"
    plain.mkdir()
    (plain / "config.json").write_text("{}")
    assert validate_base_snapshot(plain, cfg)["revision_evidence"] == "unverified_by_path"

    with pytest.raises(ProvenanceError, match="config.json"):
        validate_base_snapshot(tmp_path, cfg)


def test_artifact_record_hash_and_size(tmp_path):
    f = tmp_path / "m.gguf"
    f.write_bytes(b"abc" * 1000)
    rec = artifact_record(f)
    assert rec == {"path": str(f), "size_bytes": 3000, "sha256": hashlib.sha256(b"abc" * 1000).hexdigest()}
    with pytest.raises(ProvenanceError):
        artifact_record(tmp_path / "missing.gguf")


def test_manifest_written_idempotent_and_conflict_guarded(tmp_path):
    cfg = fake_config()
    prov = validate_adapter(make_adapter_dir(tmp_path), cfg)
    root = tmp_path / "phase7"
    m = build_manifest(cfg, prov, root)
    assert m["quality_claim"].startswith("none")
    assert m["intended_quantization"]["target"] == "Q4_K_M"
    path = write_manifest(root, cfg, m)
    assert {p.name for p in root.iterdir()} == set(cfg.paths.subdirs)
    first = path.read_text()
    write_manifest(root, cfg, build_manifest(cfg, prov, root))  # same content: kept as-is
    assert path.read_text() == first

    changed = json.loads(json.dumps(m))
    changed["base_model"]["revision"] = "0" * 40
    with pytest.raises(ManifestConflictError):
        write_manifest(root, cfg, changed)
