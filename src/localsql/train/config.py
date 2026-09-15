"""Typed parsing of `configs/train.yaml`. Pydantic + PyYAML only -- no
torch/transformers/peft import here, so this stays usable without CUDA or
the optional "train" dependency group.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, ConfigDict


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


class LoraConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    r: int
    alpha: int
    dropout: float
    target_modules: list[str]
    task_type: str


class OptimizationConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    planned_full_experiment_epochs: int
    warmup_ratio: float
    gradient_checkpointing: bool
    optim: str
    seed: int


class SequenceConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    max_seq_length: int


class LossConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    completion_only: bool


class DataConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    train_file: str
    validation_file: str


class TokenProfileConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    warn_threshold: int
    hard_limit: int


class TrainConfig(BaseModel):
    """Root config, mirroring `configs/train.yaml`."""

    model_config = ConfigDict(frozen=True)

    model: ModelSpec
    runtime: RuntimeConfig
    lora: LoraConfig
    optimization: OptimizationConfig
    sequence: SequenceConfig
    loss: LossConfig
    data: DataConfig
    token_profile: TokenProfileConfig
    paths: dict[str, str]


def load_train_config(path: Path) -> TrainConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return TrainConfig.model_validate(raw)


def quantization_summary(config: TrainConfig) -> dict:
    return {
        "load_in_4bit": config.runtime.load_in_4bit,
        "quant_type": config.runtime.quantization,
        "double_quant": config.runtime.double_quant,
        "compute_dtype": config.runtime.compute_dtype,
    }


def lora_summary(config: TrainConfig) -> dict:
    return {
        "r": config.lora.r,
        "alpha": config.lora.alpha,
        "dropout": config.lora.dropout,
        "target_modules": list(config.lora.target_modules),
        "task_type": config.lora.task_type,
    }


def optimization_summary(config: TrainConfig) -> dict:
    return {
        "per_device_train_batch_size": config.optimization.per_device_train_batch_size,
        "gradient_accumulation_steps": config.optimization.gradient_accumulation_steps,
        "learning_rate": config.optimization.learning_rate,
        "warmup_ratio": config.optimization.warmup_ratio,
        "gradient_checkpointing": config.optimization.gradient_checkpointing,
        "optim": config.optimization.optim,
        "seed": config.optimization.seed,
    }
