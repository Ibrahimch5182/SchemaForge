"""SFT example loading and explicit completion-only label masking.

No torch/transformers import here -- `tokenizer` is any object exposing
`apply_chat_template(...)` (real Qwen tokenizer or a test double), so this
module is fully unit-testable offline. Reuses Phase 1's prepared examples
and their already-baked-in prompt/completion/evidence-dropout decisions
verbatim -- the dataset is never re-split or re-derived here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from localsql.data.models import PreparedTextToSQLExample
from localsql.model.generation import extract_token_ids

LABEL_IGNORE_INDEX = -100


def load_prepared_examples(path: Path) -> list[PreparedTextToSQLExample]:
    """Load Phase 1's `train.jsonl` / `validation.jsonl` verbatim -- no
    re-splitting, no re-deriving prompt/completion/evidence decisions."""
    examples: list[PreparedTextToSQLExample] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(PreparedTextToSQLExample.model_validate_json(line))
    return examples


@dataclass(frozen=True)
class SftEncoding:
    """One completion-only-masked SFT training example."""

    example_id: str
    input_ids: list[int]
    labels: list[int]
    prompt_token_count: int
    completion_token_count: int
    total_token_count: int
    exceeds_max_seq_length: bool
    prefix_matches: bool


def build_sft_encoding(
    tokenizer: Any,
    example_id: str,
    prompt: str,
    completion: str,
    max_seq_length: int | None = None,
) -> SftEncoding:
    """Tokenize (prompt, completion) via Qwen's native chat template and
    mask the prompt region so only assistant SQL tokens are trainable.

    Two separate `apply_chat_template` calls establish the exact boundary:
    one for the user-only turn (`add_generation_prompt=True`, i.e. what the
    model sees right before it starts generating), one for the full
    user+assistant conversation. The first call's length is the prompt
    token count; everything at or after that position in the full sequence
    is the completion and is left trainable. Because the prompt-only call
    is never given the completion text, the gold SQL cannot leak backward
    into what gets masked as "prompt".
    """
    prompt_only = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
    )
    full = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": completion},
        ],
        tokenize=True,
        add_generation_prompt=False,
    )

    prompt_ids = extract_token_ids(prompt_only)
    full_ids = extract_token_ids(full)

    prompt_len = min(len(prompt_ids), len(full_ids))
    total_len = len(full_ids)

    labels = [LABEL_IGNORE_INDEX] * prompt_len + list(full_ids[prompt_len:])

    # Diagnostic: does the full sequence's prefix actually equal the
    # standalone prompt-only encoding? True for ChatML-style templates
    # (Qwen3's) -- Phase 4 validated this at 0/6,067 mismatches on the
    # real tokenizer. `prompt_len` above assumes this holds; a False here
    # on real data would mean the mask boundary is wrong for this example.
    prefix_matches = list(full_ids[:prompt_len]) == list(prompt_ids[:prompt_len])

    return SftEncoding(
        example_id=example_id,
        input_ids=list(full_ids),
        labels=labels,
        prompt_token_count=prompt_len,
        completion_token_count=max(total_len - prompt_len, 0),
        total_token_count=total_len,
        exceeds_max_seq_length=bool(max_seq_length) and total_len > max_seq_length,
        prefix_matches=prefix_matches,
    )


def pad_sft_encoding(encoding: SftEncoding, max_seq_length: int, pad_token_id: int) -> SftEncoding:
    """Right-pad to `max_seq_length`. Padding positions get label -100 (never
    trainable). No-op if already at or beyond `max_seq_length` -- this
    function never truncates (truncation would risk cutting off the gold
    SQL completion itself; over-length examples are filtered explicitly by
    the caller via `exceeds_max_seq_length`, never silently chopped).
    """
    pad_len = max_seq_length - len(encoding.input_ids)
    if pad_len <= 0:
        return encoding
    return SftEncoding(
        example_id=encoding.example_id,
        input_ids=encoding.input_ids + [pad_token_id] * pad_len,
        labels=encoding.labels + [LABEL_IGNORE_INDEX] * pad_len,
        prompt_token_count=encoding.prompt_token_count,
        completion_token_count=encoding.completion_token_count,
        total_token_count=encoding.total_token_count,
        exceeds_max_seq_length=encoding.exceeds_max_seq_length,
        prefix_matches=encoding.prefix_matches,
    )
