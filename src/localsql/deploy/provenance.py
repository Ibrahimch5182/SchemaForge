"""Frozen-input validation and artifact hashing for Phase 7. Fail closed."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from localsql.deploy.config import Phase7Config

ADAPTER_CONFIG_FILE = "adapter_config.json"
ADAPTER_WEIGHTS_FILE = "adapter_model.safetensors"
_HEX40 = re.compile(r"^[0-9a-f]{40}$")


class ProvenanceError(RuntimeError):
    """A frozen input does not match the Phase 7 contract."""


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Streaming SHA256 -- GGUFs are multi-GB, never read whole into RAM."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def artifact_record(path: Path) -> dict:
    """{path, size_bytes, sha256} for an existing file."""
    path = Path(path)
    if not path.is_file():
        raise ProvenanceError(f"Artifact not found: {path}")
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}


def dir_fingerprint(path: Path) -> dict:
    """Cheap identity of a directory input (relative path + size per file),
    used for HF snapshot dirs where hashing every weight shard is wasteful."""
    path = Path(path)
    entries = sorted(
        (p.relative_to(path).as_posix(), p.stat().st_size) for p in path.rglob("*") if p.is_file()
    )
    digest = hashlib.sha256(json.dumps(entries).encode("utf-8")).hexdigest()
    return {"path": str(path), "file_count": len(entries), "listing_sha256": digest}


@dataclass(frozen=True)
class AdapterProvenance:
    adapter_dir: str
    checkpoint: str
    weights_sha256: str
    weights_size_bytes: int
    base_model_name_or_path: str
    r: int
    lora_alpha: int

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def validate_adapter(adapter_dir: Path, config: Phase7Config) -> AdapterProvenance:
    """Validate the frozen checkpoint-1518 adapter. Raises ProvenanceError
    on any mismatch (missing file, wrong SHA, wrong base model, wrong r/alpha)."""
    adapter_dir = Path(adapter_dir)
    if not adapter_dir.is_dir():
        raise ProvenanceError(f"Adapter directory not found: {adapter_dir}")
    cfg_path = adapter_dir / ADAPTER_CONFIG_FILE
    weights_path = adapter_dir / ADAPTER_WEIGHTS_FILE
    missing = [p.name for p in (cfg_path, weights_path) if not p.is_file()]
    if missing:
        raise ProvenanceError(f"Adapter directory {adapter_dir} is missing: {', '.join(missing)}")

    sha = sha256_file(weights_path)
    if sha != config.adapter.weights_sha256:
        raise ProvenanceError(
            f"Adapter weights SHA256 mismatch: got {sha}, expected {config.adapter.weights_sha256} "
            f"({config.adapter.checkpoint})"
        )

    try:
        raw = json.loads(cfg_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ProvenanceError(f"{cfg_path} is not valid JSON: {e}") from e

    base = raw.get("base_model_name_or_path")
    if base != config.model.id:
        raise ProvenanceError(f"Adapter base_model_name_or_path is {base!r}, expected {config.model.id!r}")
    if raw.get("r") != config.adapter.expected_r:
        raise ProvenanceError(f"Adapter r is {raw.get('r')!r}, expected {config.adapter.expected_r}")
    if raw.get("lora_alpha") != config.adapter.expected_lora_alpha:
        raise ProvenanceError(
            f"Adapter lora_alpha is {raw.get('lora_alpha')!r}, expected {config.adapter.expected_lora_alpha}"
        )

    return AdapterProvenance(
        adapter_dir=str(adapter_dir),
        checkpoint=config.adapter.checkpoint,
        weights_sha256=sha,
        weights_size_bytes=weights_path.stat().st_size,
        base_model_name_or_path=base,
        r=raw["r"],
        lora_alpha=raw["lora_alpha"],
    )


def validate_base_snapshot(hf_dir: Path, config: Phase7Config) -> dict:
    """Check a local HF base-model directory as far as the filesystem allows.

    A directory cannot prove its own revision. We require config.json, and if
    the directory name is a 40-hex commit SHA (HF cache `snapshots/<sha>`
    layout) it must equal the frozen revision. Otherwise the revision is
    recorded as `unverified_by_path` so the manifest never overclaims.
    """
    hf_dir = Path(hf_dir)
    if not hf_dir.is_dir():
        raise ProvenanceError(f"Base model directory not found: {hf_dir}")
    if not (hf_dir / "config.json").is_file():
        raise ProvenanceError(f"Base model directory {hf_dir} has no config.json")
    name = hf_dir.name
    if _HEX40.match(name):
        if name != config.model.revision:
            raise ProvenanceError(
                f"Base snapshot directory is revision {name}, expected frozen {config.model.revision}"
            )
        evidence = "directory_name_matches_frozen_revision"
    else:
        evidence = "unverified_by_path"
    return {
        "model_id": config.model.id,
        "expected_revision": config.model.revision,
        "revision_evidence": evidence,
        "fingerprint": dir_fingerprint(hf_dir),
    }
