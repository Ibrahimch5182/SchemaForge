"""Shared synthetic fixtures for Phase 7 tests (no model, llama.cpp, CUDA, or network)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from localsql.deploy.config import Phase7Config, load_phase7_config

REPO_ROOT = Path(__file__).resolve().parents[2]
FAKE_WEIGHTS = b"fake-adapter-weights"


def real_config() -> Phase7Config:
    return load_phase7_config(REPO_ROOT / "configs" / "phase7.yaml")


def fake_config(weights: bytes = FAKE_WEIGHTS) -> Phase7Config:
    """Real config, but pinned to the SHA of synthetic fake weights."""
    cfg = real_config()
    adapter = cfg.adapter.model_copy(update={"weights_sha256": hashlib.sha256(weights).hexdigest()})
    return cfg.model_copy(update={"adapter": adapter})


def make_adapter_dir(tmp_path: Path, weights: bytes = FAKE_WEIGHTS, **overrides) -> Path:
    d = tmp_path / "adapter"
    d.mkdir()
    cfg = {"base_model_name_or_path": "Qwen/Qwen3-4B-Instruct-2507", "r": 16, "lora_alpha": 32}
    cfg.update(overrides)
    (d / "adapter_config.json").write_text(json.dumps(cfg), encoding="utf-8")
    (d / "adapter_model.safetensors").write_bytes(weights)
    return d


def make_example_dict(i: int) -> dict:
    return {
        "example_id": f"ex-{i:03d}",
        "db_id": "db",
        "dialect": "sqlite",
        "question": f"q{i}",
        "business_context": None,
        "serialized_schema": "t(\n  a INTEGER\n)",
        "prompt": f"PROMPT {i}",
        "difficulty": None,
    }
