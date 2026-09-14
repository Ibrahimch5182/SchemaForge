"""Qwen3 4-bit NF4 inference backend.

torch/transformers/bitsandbytes/huggingface_hub are imported lazily inside
methods (never at module level) so this module -- and the rest of
`localsql.model` -- stays importable on a machine without the optional
"model" dependency group (e.g. for `--dry-run`). Only `load()` and
`generate_one()` require them, and only `load()` requires CUDA.

One Qwen backend only -- not a generic multi-provider framework.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from localsql.model.config import ModelConfig
from localsql.model.generation import BackendGenerationResult, build_model_inputs


class CudaNotAvailableError(RuntimeError):
    """Real Qwen inference requires CUDA. Use --dry-run for infra checks
    without a GPU, or run this on a Linux CUDA cloud/Kaggle instance."""


@dataclass(frozen=True)
class LoadedModelInfo:
    model_id: str
    resolved_revision: str
    tokenizer_revision: str
    device: str
    quantization: dict
    torch_version: str
    transformers_version: str
    bitsandbytes_version: str
    cuda_version: Optional[str]
    gpu_name: Optional[str]
    gpu_total_memory_mb: Optional[float]


class QwenBackend:
    def __init__(self, config: ModelConfig):
        self.config = config
        self._model = None
        self._tokenizer = None
        self.info: Optional[LoadedModelInfo] = None

    def load(self) -> LoadedModelInfo:
        import torch
        import transformers
        import bitsandbytes
        from huggingface_hub import HfApi
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        if not torch.cuda.is_available():
            raise CudaNotAvailableError(
                "No usable CUDA device found. Real Qwen baseline inference must run on a "
                "Linux CUDA cloud/Kaggle GPU. Use --dry-run to validate infrastructure "
                "without a model/GPU on this machine."
            )

        cfg = self.config
        resolved_revision = HfApi().model_info(cfg.model.id, revision=cfg.model.revision).sha

        tokenizer = AutoTokenizer.from_pretrained(cfg.model.id, revision=resolved_revision)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=cfg.runtime.load_in_4bit,
            bnb_4bit_quant_type=cfg.runtime.quantization,
            bnb_4bit_use_double_quant=cfg.runtime.double_quant,
            bnb_4bit_compute_dtype=getattr(torch, cfg.runtime.compute_dtype),
        )
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model.id,
            revision=resolved_revision,
            quantization_config=bnb_config,
            device_map=cfg.runtime.device,
        )
        model.eval()

        torch.cuda.reset_peak_memory_stats()

        self._model = model
        self._tokenizer = tokenizer
        self.info = LoadedModelInfo(
            model_id=cfg.model.id,
            resolved_revision=resolved_revision,
            tokenizer_revision=resolved_revision,  # tokenizer ships in the same repo/revision
            device=cfg.runtime.device,
            quantization={
                "load_in_4bit": cfg.runtime.load_in_4bit,
                "quant_type": cfg.runtime.quantization,
                "double_quant": cfg.runtime.double_quant,
                "compute_dtype": cfg.runtime.compute_dtype,
            },
            torch_version=torch.__version__,
            transformers_version=transformers.__version__,
            bitsandbytes_version=bitsandbytes.__version__,
            cuda_version=torch.version.cuda,
            gpu_name=torch.cuda.get_device_name(0),
            gpu_total_memory_mb=round(torch.cuda.get_device_properties(0).total_memory / (1024**2), 1),
        )
        return self.info

    def load_tokenizer_only(self) -> str:
        """Lightweight load for token-profile mode -- no model weights,
        no bitsandbytes, no CUDA requirement. Returns the resolved revision.
        """
        from huggingface_hub import HfApi
        from transformers import AutoTokenizer

        cfg = self.config
        resolved_revision = HfApi().model_info(cfg.model.id, revision=cfg.model.revision).sha
        self._tokenizer = AutoTokenizer.from_pretrained(cfg.model.id, revision=resolved_revision)
        return resolved_revision

    def count_prompt_tokens(self, canonical_prompt: str) -> int:
        """Token count of the fully chat-templated input (what the model
        actually sees), for token-profile mode."""
        input_ids = build_model_inputs(self._tokenizer, canonical_prompt)
        return len(input_ids)

    def generate_one(self, prompt: str) -> BackendGenerationResult:
        import torch

        cfg = self.config
        device = next(self._model.parameters()).device
        input_ids = build_model_inputs(self._tokenizer, prompt, return_tensors="pt").to(device)
        input_len = input_ids.shape[-1]

        start = time.perf_counter()
        with torch.inference_mode():
            output_ids = self._model.generate(
                input_ids,
                max_new_tokens=cfg.generation.max_new_tokens,
                do_sample=cfg.generation.do_sample,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        latency_ms = (time.perf_counter() - start) * 1000

        new_tokens = output_ids[0][input_len:]
        raw_completion = self._tokenizer.decode(new_tokens, skip_special_tokens=True)

        return BackendGenerationResult(
            raw_completion=raw_completion,
            input_tokens=int(input_len),
            output_tokens=int(new_tokens.shape[-1]),
            latency_ms=latency_ms,
        )

    def peak_memory_mb(self) -> Optional[float]:
        import torch

        if not torch.cuda.is_available():
            return None
        return round(torch.cuda.max_memory_allocated() / (1024**2), 1)
