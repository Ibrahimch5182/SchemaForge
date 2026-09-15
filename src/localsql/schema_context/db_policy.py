"""Phase 5A candidate policy: adaptive per-database full-schema compaction.

This is the SELECTED candidate policy (as of this closeout) -- NOT the
question-conditioned selector in `relevance.py`, which remains in the repo
as documented, evaluated-but-not-chosen research tooling (see
`docs/CONTEXT_BUDGET.md`). No question-conditioning, no gold SQL, no
per-example table/column dropping anywhere in this module.

Rule: for each database, if ANY of its examples' canonical full-SFT
representation exceeds 4096 tokens, every example from that database uses
the compact (description-free) full-schema serializer; every other
database's examples are left byte-identical to Phase 1's canonical
`data/processed/{train,validation}.jsonl`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# --- TRAIN: sourced from the REAL Phase 4 Kaggle Qwen-tokenizer profile ---
# (kaggle-phase4-evidence/runs/qlora-smoke-1/train_token_profile.json,
# tokenizer revision cdbee75f17c01a7cc42f958dc650907174af0554, plus the
# real per-database median/max figures reported from that same Kaggle run).
# This is REAL data, not a local estimate -- verified programmatically
# against data/processed/train.jsonl's actual db_id/example counts (see
# tests/schema_context/test_db_policy.py and scripts/build_phase5_candidate.py).
REAL_PHASE4_SOURCE_CITATION = (
    "kaggle-phase4-evidence/runs/qlora-smoke-1/train_token_profile.json "
    "(tokenizer revision cdbee75f17c01a7cc42f958dc650907174af0554); "
    "per-database concentration figures from the same Kaggle run."
)

TRAIN_COMPACT_DB_IDS: frozenset[str] = frozenset(
    {
        "works_cycles",
        "hockey",
        "movie_3",
        "mondial_geo",
        "synthea",
        "professional_basketball",
        "donor",
        "superstore",
        "world_development_indicators",
    }
)


@dataclass(frozen=True)
class DbPolicyDecision:
    db_id: str
    compact: bool
    example_count: int
    max_estimated_tokens: Optional[float]  # None when not applicable (real-data-based decision)
    basis: str  # "real_phase4_profile" | "local_estimate"


def train_db_policy(db_example_counts: dict[str, int]) -> dict[str, DbPolicyDecision]:
    """Policy for training databases: membership in `TRAIN_COMPACT_DB_IDS`
    is REAL-data-derived (see citation above), not estimated. This function
    only attaches example counts and basis labeling -- it does not
    re-derive which databases are long from local estimates.
    """
    decisions = {}
    for db_id, count in db_example_counts.items():
        decisions[db_id] = DbPolicyDecision(
            db_id=db_id,
            compact=db_id in TRAIN_COMPACT_DB_IDS,
            example_count=count,
            max_estimated_tokens=None,  # not applicable -- decision is real-data-based
            basis="real_phase4_profile",
        )
    return decisions


def validation_db_policy(
    per_db_estimated_lengths: dict[str, list[float]], threshold: float = 4096.0
) -> dict[str, DbPolicyDecision]:
    """Policy for validation databases: Phase 4 never profiled validation
    with the real tokenizer, so this uses the LOCAL character-count
    estimator (`localsql.schema_context.token_estimate`) -- clearly labeled
    `basis="local_estimate"`, never validation gold correctness. Canonical
    (representation A) full-SFT length only; no question-conditioning, no
    gold SQL.
    """
    decisions = {}
    for db_id, lengths in per_db_estimated_lengths.items():
        max_len = max(lengths) if lengths else 0.0
        decisions[db_id] = DbPolicyDecision(
            db_id=db_id,
            compact=max_len > threshold,
            example_count=len(lengths),
            max_estimated_tokens=max_len,
            basis="local_estimate",
        )
    return decisions


def summarize_policy(decisions: dict[str, DbPolicyDecision]) -> dict:
    compact = [d for d in decisions.values() if d.compact]
    unchanged = [d for d in decisions.values() if not d.compact]
    return {
        "compact_db_count": len(compact),
        "compact_example_count": sum(d.example_count for d in compact),
        "compact_db_ids": sorted(d.db_id for d in compact),
        "unchanged_db_count": len(unchanged),
        "unchanged_example_count": sum(d.example_count for d in unchanged),
        "unchanged_db_ids": sorted(d.db_id for d in unchanged),
    }
