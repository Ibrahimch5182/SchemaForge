"""Prompt-envelope construction and per-example generation orchestration.

No torch/transformers import here -- `backend` is any object exposing
`generate_one(prompt: str) -> BackendGenerationResult`, so this module is
fully unit-testable with a fake backend/tokenizer. The only real
implementation is `localsql.model.qwen_backend.QwenBackend`.

Reuses the Phase 1/2 canonical prompt (`GenerationExample.prompt`) verbatim
as the content of a single user message -- no second Text-to-SQL prompt is
introduced here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from localsql.benchmark.models import GenerationExample
from localsql.data.prompt_builder import build_prompt

# Errors whose message indicates a GPU out-of-memory condition. Unlike a
# normal per-example failure, OOM likely leaves the CUDA context unusable,
# so the run stops rather than "continuing where safe".
_OOM_MARKERS = ("out of memory", "cuda oom", "outofmemoryerror")


def build_chat_messages(canonical_prompt: str) -> list[dict[str, str]]:
    """Wrap the canonical LocalSQL prompt as a single user message.

    This is the only model-specific envelope step; the prompt content
    itself is untouched.
    """
    return [{"role": "user", "content": canonical_prompt}]


def build_model_inputs(tokenizer: Any, canonical_prompt: str, **kwargs: Any) -> Any:
    """Apply the tokenizer's native chat template to the canonical prompt.

    `add_generation_prompt=True` and `tokenize=True` are always set;
    additional kwargs (e.g. `return_tensors="pt"`) are forwarded so real
    usage can request tensors while tests can pass a fake tokenizer that
    returns plain data.
    """
    messages = build_chat_messages(canonical_prompt)
    return tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, **kwargs
    )


def count_input_tokens(encoded: Any) -> int:
    """Count tokens in the sequence dimension of whatever
    `tokenizer.apply_chat_template(..., tokenize=True)` returned.

    `apply_chat_template` can return several shapes depending on the
    `return_dict`/`return_tensors` arguments (and, empirically, the
    installed transformers version's own defaults):

    - a dict-like object (e.g. a `BatchEncoding`) with an `input_ids` key
      -- `len(encoded)` on this counts dict KEYS (e.g. 2 for
      `input_ids`+`attention_mask`), not tokens. This was the actual bug:
      the token-profiler reported 2 tokens for every prompt.
    - a PyTorch (or other) tensor shaped `[batch, sequence]` (or just
      `[sequence]`) -- token count is the last dimension.
    - a plain Python list of token ids (`tokenize=True`, no
      `return_tensors`), or a batch-of-one nested list `[[...]]`.

    No torch import needed: tensor-ness is duck-typed via `.shape`.
    """
    input_ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
    if hasattr(input_ids, "shape"):
        return int(input_ids.shape[-1])
    if isinstance(input_ids, list) and input_ids and isinstance(input_ids[0], list):
        return len(input_ids[0])
    return len(input_ids)


def resolve_generation_example(example: GenerationExample, context_mode: str) -> GenerationExample:
    """Return the example to actually generate from, for the given context mode.

    `with_business_context` (the primary Phase 3 benchmark) uses the
    Phase 2 manifest's prompt as-is (already built with evidence included
    when present). `without_business_context` rebuilds the prompt via the
    same canonical `build_prompt` with `business_context=None` -- an
    ablation this runner supports architecturally but does not execute in
    this phase.
    """
    if context_mode == "with_business_context":
        return example
    if context_mode == "without_business_context":
        prompt = build_prompt(example.serialized_schema, example.dialect, example.question, None)
        return example.model_copy(update={"prompt": prompt, "business_context": None})
    raise ValueError(f"unknown context_mode: {context_mode!r}")


def normalize_predicted_sql(raw_completion: str) -> str:
    """The ONLY normalization applied to a raw decoded completion.

    Whitespace-trim only. Never search for `SELECT`, strip prefixes/fences,
    or otherwise repair the output -- a baseline that violates the SQL-only
    contract is a legitimate, visible result.
    """
    return raw_completion.strip()


def is_oom_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return any(marker in message for marker in _OOM_MARKERS)


@dataclass(frozen=True)
class BackendGenerationResult:
    raw_completion: str
    input_tokens: int
    output_tokens: int
    latency_ms: float


class GenerationBackend(Protocol):
    def generate_one(self, prompt: str) -> BackendGenerationResult: ...


@dataclass(frozen=True)
class GenerationRecord:
    """One row of `generations.jsonl`. Never carries gold SQL."""

    example_id: str
    db_id: str
    status: str  # "ok" | "error"
    raw_completion: str | None
    predicted_sql: str | None
    input_tokens: int | None
    output_tokens: int | None
    latency_ms: float | None
    model_revision: str | None
    context_mode: str
    error: str | None = None
    is_oom: bool = False

    def to_dict(self) -> dict:
        return {
            "example_id": self.example_id,
            "db_id": self.db_id,
            "status": self.status,
            "raw_completion": self.raw_completion,
            "predicted_sql": self.predicted_sql,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "latency_ms": self.latency_ms,
            "model_revision": self.model_revision,
            "context_mode": self.context_mode,
            "error": self.error,
            "is_oom": self.is_oom,
        }

    def to_prediction_dict(self) -> dict:
        """Phase 2 prediction-contract record (only for status == 'ok')."""
        return {
            "example_id": self.example_id,
            "db_id": self.db_id,
            "predicted_sql": self.predicted_sql,
            "latency_ms": self.latency_ms,
            "prompt_tokens": self.input_tokens,
            "completion_tokens": self.output_tokens,
            "model_id": None,  # filled by the caller, which knows the model id
            "context_mode": self.context_mode,
        }


def generate_one_example(
    example: GenerationExample,
    backend: GenerationBackend,
    model_revision: str | None,
    context_mode: str,
) -> GenerationRecord:
    """Generate a prediction for one gold-free example. Never raises --
    always returns a record, with `status="error"` (and `is_oom=True` for
    an apparent GPU out-of-memory failure) on failure. The caller is
    responsible for persisting every record and stopping the run when
    `is_oom` is set, since the CUDA context is likely unusable afterward.
    """
    try:
        result = backend.generate_one(example.prompt)
    except Exception as e:
        return GenerationRecord(
            example_id=example.example_id,
            db_id=example.db_id,
            status="error",
            raw_completion=None,
            predicted_sql=None,
            input_tokens=None,
            output_tokens=None,
            latency_ms=None,
            model_revision=model_revision,
            context_mode=context_mode,
            error=f"{type(e).__name__}: {e}"[:500],
            is_oom=is_oom_error(e),
        )

    return GenerationRecord(
        example_id=example.example_id,
        db_id=example.db_id,
        status="ok",
        raw_completion=result.raw_completion,
        predicted_sql=normalize_predicted_sql(result.raw_completion),
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        latency_ms=result.latency_ms,
        model_revision=model_revision,
        context_mode=context_mode,
        error=None,
    )
