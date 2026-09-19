"""Model runtime abstraction.

`QueryService` depends only on the `ModelRuntime` protocol: a runtime takes the
canonical prompt and returns the RAW completion. Normalization is the
service's job (`normalize_predicted_sql`), so every runtime -- local llama.cpp
today, a persistent/cloud server later -- is normalized identically.

`LlamaCppRuntime` reuses the Phase 7 hot-LoRA deployment (Q4_K_M base +
runtime `--lora`) and its `run_gguf_once` subprocess helper unchanged.
Limitation (inherited from Phase 7): each request starts a llama.cpp process
and reloads the model; a persistent server runtime would replace this class.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Protocol

from localsql.backend.errors import ModelRuntimeError, RuntimeNotConfiguredError
from localsql.deploy.runtime import GgufRun, LlamaSettings, run_gguf_once


@dataclass(frozen=True)
class ModelGeneration:
    raw_completion: str
    latency_ms: float
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ModelRuntime(Protocol):
    def generate(self, prompt: str) -> ModelGeneration:
        """Return the raw completion for a canonical prompt, or raise ModelRuntimeError."""
        ...

    def describe(self) -> dict[str, Any]:
        """Static, non-sensitive description (no filesystem paths)."""
        ...


class UnavailableRuntime:
    """Placeholder when the model is not configured, so /health and /databases
    still work and /query fails with a clear, safe error."""

    def __init__(self, reason: str):
        self._reason = reason

    def generate(self, prompt: str) -> ModelGeneration:
        raise RuntimeNotConfiguredError(f"Model runtime is not configured: {self._reason}")

    def describe(self) -> dict[str, Any]:
        return {"runtime": "unavailable", "configured": False, "reason": self._reason}


class LlamaCppRuntime:
    def __init__(
        self,
        settings: LlamaSettings,
        *,
        run: Callable[..., GgufRun] = run_gguf_once,
        check_files: bool = True,
    ):
        if settings.lora is None:
            raise ModelRuntimeError("Phase 8 requires the hot-LoRA deployment (Q4_K_M base + LoRA GGUF).")
        if check_files:
            for label, p in (("llama executable", settings.executable), ("base GGUF", settings.model), ("LoRA GGUF", settings.lora)):
                if not Path(p).is_file():
                    raise ModelRuntimeError(f"{label} not found.")
        self._settings = settings
        self._run = run
        # One llama.cpp process at a time: each holds a multi-GB model.
        self._lock = threading.Lock()

    def generate(self, prompt: str) -> ModelGeneration:
        with self._lock:
            result = self._run(prompt, self._settings)
        if result.status != "ok" or result.raw_completion is None:
            # result.error may embed process stderr (paths); keep the public message generic.
            raise ModelRuntimeError(f"Model generation failed ({result.status}).")
        perf = result.perf or {}
        return ModelGeneration(
            raw_completion=result.raw_completion,
            latency_ms=result.latency_ms,
            input_tokens=perf.get("prompt_tokens"),
            output_tokens=perf.get("generated_tokens"),
            metadata={
                "generated_tokens_per_second": perf.get("generated_tokens_per_second"),
                "prompt_tokens_per_second": perf.get("prompt_tokens_per_second"),
                "peak_rss_bytes": result.peak_rss_bytes,
                "end_of_text_marker_stripped": result.end_of_text_marker_stripped,
            },
        )

    def describe(self) -> dict[str, Any]:
        s = self._settings
        return {
            "runtime": "llama_cpp",
            "configured": True,
            "deployment_mode": "hot_lora",
            "base_gguf": Path(s.model).name,
            "lora_gguf": Path(s.lora).name if s.lora else None,
            "context_size": s.context_size,
            "max_new_tokens": s.max_new_tokens,
            "seed": s.seed,
            "n_gpu_layers": s.n_gpu_layers,
            "sampling": "greedy",
        }
