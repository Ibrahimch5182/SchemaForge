"""Model runtime abstraction.

`QueryService` depends only on the `ModelRuntime` protocol: a runtime takes the
canonical prompt and returns the RAW completion. Normalization is the
service's job (`normalize_predicted_sql`), so every runtime -- local llama.cpp
today, a persistent/cloud server later -- is normalized identically.

`LlamaCppRuntime` reuses the Phase 7 hot-LoRA deployment (Q4_K_M base +
runtime `--lora`) and its `run_gguf_once` subprocess helper. Phase 10 makes it
predictable: a bounded `InferenceGate` (no unbounded pile-up of multi-GB
processes), cooperative cancellation that kills the process, stable error
codes, and a cheap `readiness()` that never loads the model.
Limitation (Phase 11): each request still starts a process and reloads the model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Protocol

from localsql.backend.control import CancelToken, InferenceGate
from localsql.backend.errors import (
    ModelRuntimeError,
    ModelTimeoutError,
    RequestCancelledError,
    RuntimeNotConfiguredError,
)
from localsql.backend.observability import get_logger, log_event
from localsql.deploy.runtime import GgufRun, LlamaSettings, run_gguf_once


@dataclass(frozen=True)
class ModelGeneration:
    raw_completion: str
    latency_ms: float
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ModelRuntime(Protocol):
    def generate(self, prompt: str, cancel: Optional[CancelToken] = None) -> ModelGeneration:
        """Return the raw completion for a canonical prompt, or raise ModelRuntimeError
        (ModelBusyError / ModelTimeoutError / ...) or RequestCancelledError."""
        ...

    def describe(self) -> dict[str, Any]:
        """Static, non-sensitive description (no filesystem paths)."""
        ...


class UnavailableRuntime:
    """Placeholder when the model is not configured, so /health and /databases
    still work and /query fails with a clear, safe error."""

    def __init__(self, reason: str):
        self._reason = reason

    def generate(self, prompt: str, cancel: Optional[CancelToken] = None) -> ModelGeneration:
        raise RuntimeNotConfiguredError(f"Model runtime is not configured: {self._reason}")

    def describe(self) -> dict[str, Any]:
        return {"runtime": "unavailable", "configured": False, "reason": self._reason}

    def readiness(self) -> dict[str, Any]:
        return {"ready": False, "checks": {}, "availability": None}


class LlamaCppRuntime:
    def __init__(
        self,
        settings: LlamaSettings,
        *,
        run: Callable[..., GgufRun] = run_gguf_once,
        check_files: bool = True,
        gate: Optional[InferenceGate] = None,
        logger: Optional[logging.Logger] = None,
    ):
        if settings.lora is None:
            raise ModelRuntimeError("Phase 8 requires the hot-LoRA deployment (Q4_K_M base + LoRA GGUF).")
        if check_files:
            for label, p in (("llama executable", settings.executable), ("base GGUF", settings.model), ("LoRA GGUF", settings.lora)):
                if not Path(p).is_file():
                    raise ModelRuntimeError(f"{label} not found.")
        self._settings = settings
        self._run = run
        self._gate = gate or InferenceGate()
        self._log = logger or get_logger()

    @property
    def gate(self) -> InferenceGate:
        return self._gate

    def generate(self, prompt: str, cancel: Optional[CancelToken] = None) -> ModelGeneration:
        with self._gate.slot(cancel):  # raises ModelBusyError / RequestCancelledError
            if cancel is not None and cancel.cancelled:
                raise RequestCancelledError("The request was cancelled.")
            result = self._run(prompt, self._settings, should_cancel=(lambda: cancel.cancelled) if cancel else None)
        if result.status == "cancelled":
            raise RequestCancelledError("The request was cancelled.")
        if result.status == "timeout":
            log_event(self._log, "runtime.timeout", level=logging.WARNING, timeout_s=self._settings.timeout_seconds)
            raise ModelTimeoutError("Model generation timed out.")
        if result.status != "ok" or result.raw_completion is None:
            # Internal detail (exit code, stderr tail) goes to the log only; the API message stays generic.
            log_event(
                self._log,
                "runtime.failure",
                level=logging.ERROR,
                status=result.status,
                returncode=result.returncode,
                detail=(result.error or "")[-300:],
            )
            raise ModelRuntimeError("Model generation failed.")
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

    def readiness(self) -> dict[str, Any]:
        """Cheap and side-effect free: stats three files and reads gate counters.
        Never loads the model and never hashes the multi-GB artifacts."""
        s = self._settings
        checks = {
            "executable": Path(s.executable).is_file(),
            "base_gguf": Path(s.model).is_file(),
            "lora_gguf": bool(s.lora) and Path(s.lora).is_file(),
        }
        return {"ready": all(checks.values()), "checks": checks, "availability": self._gate.snapshot()}

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
