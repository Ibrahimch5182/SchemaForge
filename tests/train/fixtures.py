"""Fakes and small fixtures for Phase 4 train-package tests.

No torch/transformers/peft here -- a deterministic word-based fake
tokenizer stands in for the real Qwen tokenizer, matching the interface
`localsql.train.sft_data.build_sft_encoding` actually calls.
"""

from __future__ import annotations

from localsql.data.models import PreparedTextToSQLExample, SourceMetadata


def make_prepared_example(
    idx: int = 0,
    db_id: str = "shop_db",
    business_context: str | None = "customers refers to the customers table",
    completion: str = "SELECT COUNT(*) FROM customers",
    split: str = "train",
) -> PreparedTextToSQLExample:
    return PreparedTextToSQLExample(
        example_id=f"bird23-train-filtered:{idx:05d}",
        db_id=db_id,
        dialect="sqlite",
        question="How many customers are there?",
        business_context=business_context,
        schema_serialized="customers(\n  customer_id INTEGER PK\n)",
        prompt=(
            "SYSTEM:\n...\n\nSCHEMA:\ncustomers(\n  customer_id INTEGER PK\n)\n\n"
            "QUESTION:\nHow many customers are there?"
        ),
        completion=completion,
        source=SourceMetadata(
            dataset_repo_id="birdsql/bird23-train-filtered",
            dataset_revision="abc123",
            row_index=idx,
            evidence_available=business_context is not None,
            business_context_kept=business_context is not None,
        ),
        split=split,
    )


class FakeSftTokenizer:
    """Deterministic word-based tokenizer double: each unique word gets a
    stable integer id (no `hash()` -- CPython string hashing is randomized
    per process, which would make tests flaky). No tensors, no dict-like
    BatchEncoding -- `apply_chat_template` returns a plain flat list, the
    "tokenize=True, no return_tensors" shape.
    """

    pad_token_id = 0
    eos_token_id = 1

    _ROLE_MARKERS = {"user": 3, "assistant": 4}
    _TURN_END = 5

    def __init__(self):
        self._vocab: dict[str, int] = {}
        self._next_id = 10

    def word_id(self, word: str) -> int:
        if word not in self._vocab:
            self._vocab[word] = self._next_id
            self._next_id += 1
        return self._vocab[word]

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True, **kwargs):
        ids = [2]  # bos-like marker
        for m in messages:
            ids.append(self._ROLE_MARKERS[m["role"]])
            for word in m["content"].split():
                ids.append(self.word_id(word))
            ids.append(self._TURN_END)
        if add_generation_prompt:
            # Real ChatML-style templates (incl. Qwen's) make this the exact
            # same tokens that open an assistant turn (`<|im_start|>
            # assistant\n`) -- i.e. a true prefix of the full sequence, which
            # is the whole point of `add_generation_prompt`. Reuse the same
            # role marker here, not a distinct one, to model that faithfully.
            ids.append(self._ROLE_MARKERS["assistant"])
        return ids
