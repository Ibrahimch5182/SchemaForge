"""Typed parsing of `configs/model.yaml`. Pydantic + PyYAML only -- no torch/
transformers/bitsandbytes import here, so this stays usable in dry-run mode.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict

ContextMode = Literal["with_business_context", "without_business_context"]


class ModelSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    revision: Optional[str] = None


class RuntimeConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    load_in_4bit: bool
    quantization: str
    double_quant: bool
    compute_dtype: str
    device: str
    batch_size: int


class GenerationConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    do_sample: bool
    max_new_tokens: int
    seed: int


class PromptConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    envelope: str
    context_mode: ContextMode


class TokenProfileConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    warn_threshold: int
    hard_limit: int


class ModelConfig(BaseModel):
    """Root config, mirroring `configs/model.yaml`."""

    model_config = ConfigDict(frozen=True)

    model: ModelSpec
    runtime: RuntimeConfig
    generation: GenerationConfig
    prompt: PromptConfig
    token_profile: TokenProfileConfig
    paths: dict[str, str]


def load_model_config(path: Path) -> ModelConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return ModelConfig.model_validate(raw)


def quantization_summary(config: ModelConfig) -> dict:
    """Compact dict of quantization settings for run provenance/reports."""
    return {
        "load_in_4bit": config.runtime.load_in_4bit,
        "quant_type": config.runtime.quantization,
        "double_quant": config.runtime.double_quant,
        "compute_dtype": config.runtime.compute_dtype,
    }


def generation_summary(config: ModelConfig) -> dict:
    return {
        "do_sample": config.generation.do_sample,
        "max_new_tokens": config.generation.max_new_tokens,
        "seed": config.generation.seed,
    }
