"""Typed parsing of `configs/phase7.yaml` (pydantic + pyyaml only)."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True)


class ModelSpec(_Frozen):
    id: str
    revision: str


class AdapterSpec(_Frozen):
    checkpoint: str
    default_dir: str
    weights_sha256: str
    expected_r: int
    expected_lora_alpha: int


class DeploymentSpec(_Frozen):
    mode: str
    merged_export_optional: bool


class QuantizationSpec(_Frozen):
    intermediate_outtype: str
    target: str


class PathsSpec(_Frozen):
    root: str
    subdirs: list[str]


class InferenceSpec(_Frozen):
    context_size: int
    max_new_tokens: int
    seed: int
    n_gpu_layers: int
    timeout_seconds: int
    prompt_manifest: str


class QualitySanitySpec(_Frozen):
    sample_size: int
    selection_salt: str


class BenchmarkSpec(_Frozen):
    num_prompts: int
    warmup_runs: int
    measured_repeats: int


class Phase7Config(_Frozen):
    model: ModelSpec
    adapter: AdapterSpec
    deployment: DeploymentSpec
    quantization: QuantizationSpec
    paths: PathsSpec
    inference: InferenceSpec
    quality_sanity: QualitySanitySpec
    benchmark: BenchmarkSpec


def load_phase7_config(path: Path) -> Phase7Config:
    return Phase7Config.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
