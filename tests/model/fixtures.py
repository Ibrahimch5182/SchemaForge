"""Fakes and small fixtures for Phase 3 model-package tests.

No torch/transformers here -- these are plain Python doubles standing in
for a real tokenizer/backend, matching the Protocol shapes in
`localsql.model.generation`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from localsql.benchmark.models import GenerationExample
from localsql.model.generation import BackendGenerationResult


def make_example(idx: int = 0, db_id: str = "shop_db") -> GenerationExample:
    return GenerationExample(
        example_id=f"bird-mini-dev-sqlite-{idx:04d}",
        db_id=db_id,
        dialect="sqlite",
        question="How many customers are there?",
        business_context="customers refers to the customers table" if idx % 2 == 0 else None,
        serialized_schema="customers(\n  customer_id INTEGER PK\n)",
        prompt=(
            "SYSTEM:\n...\n\nSCHEMA:\ncustomers(\n  customer_id INTEGER PK\n)\n\n"
            "QUESTION:\nHow many customers are there?"
        ),
        difficulty="simple",
    )


class FakeTokenizer:
    """Records the exact call made to `apply_chat_template`."""

    def __init__(self):
        self.last_messages = None
        self.last_kwargs = None

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True, **kwargs):
        self.last_messages = messages
        self.last_kwargs = {"tokenize": tokenize, "add_generation_prompt": add_generation_prompt, **kwargs}
        # Stand-in "token ids": one per word, for deterministic length checks.
        content = messages[0]["content"]
        return list(range(len(content.split())))


@dataclass
class FakeBackend:
    """A `GenerationBackend` double. Returns a canned result, or raises."""

    raw_completion: str = "SELECT COUNT(*) FROM customers"
    raise_exc: Exception | None = None
    calls: list[str] = field(default_factory=list)

    def generate_one(self, prompt: str) -> BackendGenerationResult:
        self.calls.append(prompt)
        if self.raise_exc is not None:
            raise self.raise_exc
        return BackendGenerationResult(
            raw_completion=self.raw_completion,
            input_tokens=len(prompt.split()),
            output_tokens=len(self.raw_completion.split()),
            latency_ms=12.5,
        )
