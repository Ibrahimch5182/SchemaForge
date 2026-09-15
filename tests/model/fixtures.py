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


class FakeTensor:
    """Duck-types a torch tensor's `.shape` for [batch, sequence] checks
    without requiring torch to be installed."""

    def __init__(self, shape: tuple[int, ...]):
        self.shape = shape


class FakeBatchEncoding(dict):
    """Reproduces the real `transformers.BatchEncoding` shape that caused
    the bug: a dict-like object with `input_ids`/`attention_mask` keys,
    where `len(encoding) == 2` (the key count), not the token count.
    """


class FakeGenTensor:
    """Minimal tensor-like double: indexing/slicing + `.shape` + `.to()`."""

    def __init__(self, data: list):
        self.data = data

    def __getitem__(self, item):
        result = self.data[item]
        return FakeGenTensor(result) if isinstance(result, list) else result

    @property
    def shape(self):
        dims = []
        node = self.data
        while isinstance(node, list):
            dims.append(len(node))
            node = node[0] if node else None
        return tuple(dims)

    def to(self, device):
        return self


class FakeGenBatchEncoding(dict):
    """BatchEncoding double: dict-like, with `.to(device)` returning self
    (matching the real `BatchEncoding.to()`, which moves each tensor value
    and returns itself) -- NOT a bare tensor."""

    def to(self, device):
        return self


class FakeParam:
    device = "cpu"


class FakeGenTokenizer:
    """Reproduces the real Qwen tokenizer's `apply_chat_template(...,
    return_tensors="pt")` behavior: returns a `BatchEncoding`, not a tensor.
    """

    eos_token_id = 999

    def __init__(self, input_ids: list[int]):
        self._input_ids = input_ids

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True, **kwargs):
        return FakeGenBatchEncoding(
            input_ids=FakeGenTensor([list(self._input_ids)]),
            attention_mask=FakeGenTensor([[1] * len(self._input_ids)]),
        )

    def decode(self, tokens, skip_special_tokens=True):
        return "SELECT 1"


class FakeGenModel:
    """Records exactly how `.generate(...)` was called, to prove the
    BatchEncoding is unpacked by keyword, not passed as a raw tensor."""

    def __init__(self, new_token_count: int = 3):
        self.new_token_count = new_token_count
        self.last_generate_kwargs: dict | None = None

    def parameters(self):
        yield FakeParam()

    def eval(self):
        pass

    def generate(self, **kwargs):
        self.last_generate_kwargs = kwargs
        input_ids = kwargs["input_ids"]
        full_sequence = input_ids.data[0] + list(range(900, 900 + self.new_token_count))
        return FakeGenTensor([full_sequence])


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
