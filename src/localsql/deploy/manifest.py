"""Phase 7 preparation: output layout + machine-readable phase7_manifest.json."""

from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path

from localsql.deploy.config import Phase7Config
from localsql.deploy.provenance import AdapterProvenance

MANIFEST_NAME = "phase7_manifest.json"
MANIFEST_SCHEMA_VERSION = 1


class ManifestConflictError(RuntimeError):
    """An existing manifest differs from the one that would be written."""


def phase7_root(repo_root: Path, cfg: Phase7Config) -> Path:
    return Path(repo_root) / cfg.paths.root


def default_artifact_paths(root: Path, cfg: Phase7Config) -> dict:
    stem = cfg.adapter.checkpoint.replace("checkpoint-", "")
    quant = cfg.quantization.target
    outtype = cfg.quantization.intermediate_outtype
    return {
        "base_f16_gguf": str(root / "base-gguf" / f"base-{outtype}.gguf"),
        "lora_f16_gguf": str(root / "lora-gguf" / f"lora-{stem}-{outtype}.gguf"),
        "base_q4_k_m_gguf": str(root / quant.lower() / f"base-{quant}.gguf"),
        # OPTIONAL packaging only (llama-export-lora); not part of the primary path.
        "optional_merged_f16_gguf": str(root / "merged-gguf" / f"merged-{stem}-{outtype}.gguf"),
    }


def build_manifest(cfg: Phase7Config, adapter: AdapterProvenance, root: Path, now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "phase": 7,
        "created_utc": now.isoformat(timespec="seconds"),
        "base_model": {"id": cfg.model.id, "revision": cfg.model.revision},
        "adapter": adapter.to_dict(),
        "deployment_mode": cfg.deployment.mode,
        "pipeline": [
            f"HF base @ {cfg.model.revision[:12]} -> base {cfg.quantization.intermediate_outtype.upper()} GGUF",
            f"PEFT LoRA -> LoRA {cfg.quantization.intermediate_outtype.upper()} GGUF",
            f"base {cfg.quantization.intermediate_outtype.upper()} GGUF -> base {cfg.quantization.target} GGUF",
            f"base {cfg.quantization.target} + LoRA GGUF (runtime --lora) -> fine-tuned model",
        ],
        "optional_packaging": "merged GGUF via llama-export-lora is optional and not required",
        "intended_quantization": {
            "deployment_format": "GGUF",
            "quantized_component": "base model only; LoRA GGUF is held constant",
            "intermediate_outtype": cfg.quantization.intermediate_outtype,
            "target": cfg.quantization.target,
            "note": "Deployment-time llama.cpp quantization; distinct from training-time bitsandbytes NF4.",
        },
        "artifact_paths": default_artifact_paths(root, cfg),
        "quality_claim": "none: Q4_K_M quality is not asserted until measured",
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
    }


def _comparable(manifest: dict) -> dict:
    return {k: v for k, v in manifest.items() if k not in ("created_utc", "environment")}


def write_manifest(root: Path, cfg: Phase7Config, manifest: dict) -> Path:
    """Create the layout and write the manifest. Idempotent: an existing
    manifest with the same content (ignoring timestamp/environment) is kept;
    a different one is never silently overwritten."""
    for sub in cfg.paths.subdirs:
        (root / sub).mkdir(parents=True, exist_ok=True)
    path = root / "manifests" / MANIFEST_NAME
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if _comparable(existing) != _comparable(manifest):
            raise ManifestConflictError(
                f"{path} already exists with different content; refusing to overwrite. Move it deliberately first."
            )
        return path
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path
