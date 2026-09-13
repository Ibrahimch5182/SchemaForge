"""Deterministic database-level train/validation splitting and evidence dropout.

Splitting is done by `db_id`, never by individual example, so validation
measures generalization to schemas unseen during training.
"""

from __future__ import annotations

import hashlib
import random


def split_database_ids(
    db_ids: list[str],
    seed: int = 42,
    train_fraction: float = 0.90,
) -> tuple[set[str], set[str]]:
    """Deterministically partition database IDs into train/validation sets.

    Sorting before shuffling makes the result independent of input order.
    """
    unique_sorted = sorted(set(db_ids))
    rng = random.Random(seed)
    shuffled = unique_sorted[:]
    rng.shuffle(shuffled)

    split_point = round(len(shuffled) * train_fraction)
    train_ids = set(shuffled[:split_point])
    validation_ids = set(shuffled[split_point:])

    overlap = train_ids & validation_ids
    if overlap:
        raise AssertionError(f"train/validation db_id leakage detected: {overlap}")

    return train_ids, validation_ids


def should_keep_business_context(
    example_key: str,
    seed: int = 42,
    keep_probability: float = 0.50,
) -> bool:
    """Deterministically decide whether to keep an example's evidence field.

    The decision is a stable hash of (seed, example_key), so it does not
    depend on iteration order and is reproducible across runs.
    """
    digest = hashlib.sha256(f"{seed}:{example_key}".encode("utf-8")).hexdigest()
    fraction = int(digest[:16], 16) / float(16**16)
    return fraction < keep_probability
