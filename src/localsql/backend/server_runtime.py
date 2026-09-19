"""Persistent llama.cpp HTTP serving runtime (Phase 11).

`llama-server` is started once (Q4_K_M base + hot LoRA, model resident in
memory); this runtime only speaks HTTP to it. It implements the same
`ModelRuntime` protocol as the Phase 8 subprocess `LlamaCppRuntime`, so
`QueryService` is runtime-agnostic and normalization/safety/preflight/execution
are byte-for-byte unchanged.

Prompt contract: the canonical prompt is wrapped by `build_chatml_prompt` and
sent to `/completion` verbatim. The envelope ends `assistant<newline>`; the
server tokenizes the string exactly (unlike `llama-completion -f`, which strips
one trailing newline -- the Phase 8 drift the subprocess runtime compensates for
with `guard_prompt_trailing_newline`). `tests/backend/test_phase11_*` pins that
both runtimes feed the model the same effective prompt.

Cancellation/timeout: the completion is streamed on a worker thread and polled;
on cancel or deadline the HTTP connection is closed, which makes llama-server
abort the generation and free the slot. Generation is greedy with a fixed seed.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

import httpx

from localsql.backend.control import CancelToken, InferenceGate
from localsql.backend.errors import (
    ModelBusyError,
    ModelRuntimeError,
    ModelTimeoutError,
    RequestCancelledError,
)
from localsql.backend.observability import get_logger, log_event
from localsql.backend.runtime import ModelGeneration
from localsql.deploy.runtime import build_chatml_prompt

RUNTIME_MODE = "persistent_server"
_POLL_S = 0.1


@dataclass(frozen=True)
class ServerSettings:
    base_url: str
    context_size: int
    max_new_tokens: int
    seed: int
    request_timeout_seconds: float = 120.0
    connect_timeout_seconds: float = 3.0
    readiness_timeout_seconds: float = 2.0
    readiness_cache_seconds: float = 1.0
    # Prompt-prefix KV reuse is faster on repeated schemas but can perturb numerics slightly
    # versus a cold prefill; default off so serving stays deterministic and comparable.
    cache_prompt: bool = False
    n_gpu_layers: int = 0  # informational: offload is a server launch flag, not a request field
    parallel_slots: int = 1  # informational: must match the server's --parallel


def build_completion_payload(canonical_prompt: str, settings: ServerSettings) -> dict[str, Any]:
    """The exact `/completion` body: ChatML envelope, greedy decoding, fixed seed, streaming."""
    return {
        "prompt": build_chatml_prompt(canonical_prompt),
        "n_predict": settings.max_new_tokens,
        "temperature": 0,
        "top_k": 1,
        "seed": settings.seed,
        "cache_prompt": settings.cache_prompt,
        "stream": True,
    }


def _default_client(settings: ServerSettings) -> httpx.Client:
    t = settings.request_timeout_seconds
    return httpx.Client(
        base_url=settings.base_url.rstrip("/"),
        timeout=httpx.Timeout(connect=settings.connect_timeout_seconds, read=t, write=10.0, pool=5.0),
        # Never route the private model-server hop through an ambient proxy.
        trust_env=False,
    )


class LlamaServerRuntime:
    def __init__(
        self,
        settings: ServerSettings,
        *,
        gate: Optional[InferenceGate] = None,
        logger: Optional[logging.Logger] = None,
        client_factory: Optional[Callable[[ServerSettings], httpx.Client]] = None,
    ):
        self._settings = settings
        self._gate = gate or InferenceGate(settings.parallel_slots)
        self._log = logger or get_logger()
        self._client_factory = client_factory or _default_client
        self._ready_lock = threading.Lock()
        self._ready_cache: Optional[tuple[float, dict[str, Any]]] = None
        self._last_state: Optional[str] = None

    @property
    def gate(self) -> InferenceGate:
        return self._gate

    # ------------------------------------------------------------ generation
    def generate(self, prompt: str, cancel: Optional[CancelToken] = None) -> ModelGeneration:
        t_enter = time.perf_counter()
        try:
            with self._gate.slot(cancel):  # raises ModelBusyError / RequestCancelledError
                queue_wait_ms = round((time.perf_counter() - t_enter) * 1000, 3)
                if cancel is not None and cancel.cancelled:
                    raise RequestCancelledError("The request was cancelled.")
                return self._complete(prompt, cancel, queue_wait_ms)
        except ModelBusyError:
            log_event(self._log, "runtime.saturated", level=logging.WARNING, mode=RUNTIME_MODE, gate=self._gate.snapshot())
            raise

    def _fail(self, reason: str, message: str = "Model generation failed.", **fields: Any) -> ModelRuntimeError:
        # Internal detail goes to the log only; the API message stays generic and never names the server.
        log_event(self._log, "runtime.failure", level=logging.ERROR, mode=RUNTIME_MODE, reason=reason, **fields)
        return ModelRuntimeError(message)

    def _complete(self, prompt: str, cancel: Optional[CancelToken], queue_wait_ms: float) -> ModelGeneration:
        s = self._settings
        payload = build_completion_payload(prompt, s)
        start = time.perf_counter()
        deadline = time.monotonic() + s.request_timeout_seconds
        client = self._client_factory(s)
        items: "queue.Queue[tuple]" = queue.Queue()
        live: list[httpx.Response] = []  # the in-flight response, so an abort can close it directly

        def reader() -> None:
            try:
                with client.stream("POST", "/completion", json=payload) as resp:
                    live.append(resp)
                    if resp.status_code != 200:
                        items.put(("http", resp.status_code))
                        return
                    for line in resp.iter_lines():
                        items.put(("line", line))
                items.put(("eof",))
            except Exception as e:  # noqa: BLE001 - classified by the consumer
                items.put(("exc", e))

        thread = threading.Thread(target=reader, name="llama-server-stream", daemon=True)
        thread.start()
        parts: list[str] = []
        final: Optional[dict[str, Any]] = None
        try:
            while True:
                try:
                    item = items.get(timeout=_POLL_S)
                except queue.Empty:
                    item = None
                if cancel is not None and cancel.cancelled:
                    log_event(self._log, "runtime.cancelled", mode=RUNTIME_MODE, queue_wait_ms=queue_wait_ms)
                    raise RequestCancelledError("The request was cancelled.")
                if time.monotonic() >= deadline:
                    log_event(self._log, "runtime.timeout", level=logging.WARNING, mode=RUNTIME_MODE, timeout_s=s.request_timeout_seconds)
                    raise ModelTimeoutError("Model generation timed out.")
                if item is None:
                    continue
                kind = item[0]
                if kind == "exc":
                    exc = item[1]
                    if isinstance(exc, httpx.TimeoutException):
                        if isinstance(exc, httpx.ConnectTimeout):
                            raise self._fail("connect_timeout") from exc
                        log_event(self._log, "runtime.timeout", level=logging.WARNING, mode=RUNTIME_MODE, timeout_s=s.request_timeout_seconds)
                        raise ModelTimeoutError("Model generation timed out.") from exc
                    raise self._fail("server_unreachable" if isinstance(exc, httpx.TransportError) else "client_error", exception=type(exc).__name__) from exc
                if kind == "http":
                    raise self._fail("http_status", http_status=item[1])
                if kind == "eof":
                    break
                chunk = self._parse_line(item[1])
                if chunk is None:
                    continue
                if "error" in chunk:
                    code = chunk["error"].get("code") if isinstance(chunk["error"], dict) else None
                    if code == 503:
                        raise ModelBusyError("The model is busy. Try again shortly.")
                    raise self._fail("server_error_event", error_code=code)
                content = chunk.get("content", "")
                if not isinstance(content, str):
                    raise self._fail("malformed_response", detail="content_not_string")
                parts.append(content)
                if chunk.get("stop") is True:
                    final = chunk
                    break
        finally:
            # Closing the connection is what aborts an in-flight generation server-side.
            try:
                for resp in list(live):
                    resp.close()
                client.close()
            except Exception:  # noqa: BLE001
                pass
            thread.join(timeout=1.0)

        if final is None:
            raise self._fail("incomplete_stream")
        latency_ms = (time.perf_counter() - start) * 1000
        timings = final.get("timings") if isinstance(final.get("timings"), dict) else {}
        raw = "".join(parts)
        gen_tps = timings.get("predicted_per_second")
        result = ModelGeneration(
            raw_completion=raw,
            latency_ms=latency_ms,
            input_tokens=_int_or_none(final.get("tokens_evaluated")),
            output_tokens=_int_or_none(final.get("tokens_predicted")),
            metadata={
                "runtime_mode": RUNTIME_MODE,
                "generated_tokens_per_second": gen_tps,
                "prompt_tokens_per_second": timings.get("prompt_per_second"),
                "server_prompt_ms": timings.get("prompt_ms"),
                "server_generation_ms": timings.get("predicted_ms"),
                "queue_wait_ms": queue_wait_ms,
                "stop_type": final.get("stop_type"),
            },
        )
        log_event(
            self._log, "runtime.generated", mode=RUNTIME_MODE, queue_wait_ms=queue_wait_ms,
            generation_latency_ms=round(latency_ms, 3), input_tokens=result.input_tokens,
            output_tokens=result.output_tokens, generated_tokens_per_second=gen_tps,
            stop_type=final.get("stop_type"),
        )  # fmt: skip
        return result

    def _parse_line(self, line: str) -> Optional[dict[str, Any]]:
        line = line.strip()
        if not line or line.startswith(":"):
            return None
        if line.startswith("data:"):
            line = line[5:].strip()
        if not line or line == "[DONE]":
            return None
        try:
            obj = json.loads(line)
        except ValueError as e:
            raise self._fail("malformed_response", detail="invalid_json") from e
        if not isinstance(obj, dict):
            raise self._fail("malformed_response", detail="not_an_object")
        return obj

    # ------------------------------------------------------------- readiness
    def readiness(self) -> dict[str, Any]:
        """GET the server's /health (never /completion): loading vs ready vs unreachable.
        Cached for `readiness_cache_seconds` so polling cannot hammer the model server."""
        now = time.monotonic()
        with self._ready_lock:
            cached = self._ready_cache
            if cached is not None and now - cached[0] < self._settings.readiness_cache_seconds:
                state = cached[1]
            else:
                state = self._probe()
                self._ready_cache = (now, state)
        availability = self._gate.snapshot()
        name = state["state"]
        serving_state = name if name != "ready" else ("ready" if availability["state"] == "idle" else availability["state"])
        out: dict[str, Any] = {
            "ready": name == "ready",
            "availability": availability,
            "serving": {
                "mode": RUNTIME_MODE,
                "state": serving_state,
                "checks": {
                    "server_configured": True,
                    "server_reachable": name != "unreachable",
                    "model_ready": name == "ready",
                },
            },
        }
        if name != "ready":
            out["reason"] = "model_server_loading" if name == "loading" else "model_server_unreachable"
        return out

    def _probe(self) -> dict[str, Any]:
        s = self._settings
        client = self._client_factory(s)
        try:
            resp = client.get("/health", timeout=s.readiness_timeout_seconds)
            state = "ready" if resp.status_code == 200 else "loading" if resp.status_code == 503 else "unreachable"
        except httpx.HTTPError:
            state = "unreachable"
        finally:
            client.close()
        if state != self._last_state:
            log_event(self._log, "runtime.server_state", level=logging.INFO if state == "ready" else logging.WARNING,
                      mode=RUNTIME_MODE, state=state, previous=self._last_state)  # fmt: skip
            self._last_state = state
        return {"state": state}

    def describe(self) -> dict[str, Any]:
        s = self._settings
        return {
            "runtime": "llama_server",
            "runtime_mode": RUNTIME_MODE,
            "configured": True,
            "deployment_mode": "hot_lora",
            "context_size": s.context_size,
            "max_new_tokens": s.max_new_tokens,
            "seed": s.seed,
            "n_gpu_layers": s.n_gpu_layers,
            "parallel_slots": s.parallel_slots,
            "sampling": "greedy",
        }


def _int_or_none(v: Any) -> Optional[int]:
    return v if isinstance(v, int) and not isinstance(v, bool) else None
