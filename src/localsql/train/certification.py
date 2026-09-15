"""Phase 5B GPU/throughput certification support.

Deterministic selection of (a) the longest real-tokenizer-profiled
candidate examples for worst-case memory certification (Task 1), and
(b) a canonical, real-token-length-based representative sample for
throughput benchmarking (Task 3, real-token-manifest hardening pass).
Neither uses gold SQL, mutates content, or truncates anything -- both
return exact copies of candidate records, byte-for-byte.

IMPORTANT: throughput sampling was originally estimator-based (a
character-count proxy) and that approach is NO LONGER ACCEPTED for the
canonical Phase 5B throughput benchmark -- it was replaced after
observing material threshold-count error versus the real Qwen tokenizer.
The canonical path (`build_real_token_throughput_sample`) consumes ONLY
a real per-example token-length manifest produced by the resolved Qwen
tokenizer/chat-template path (see `write_token_length_manifest` and
`scripts/run_qlora_smoke.py`'s `run_token_profile`). The estimator
(`localsql.schema_context.token_estimate`) is not imported here.

Both are scratch/certification data, never canonical training data.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_jsonl_index(path: Path) -> dict[str, dict]:
    """`example_id` -> record, preserving every field verbatim."""
    index: dict[str, dict] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                index[row["example_id"]] = row
    return index


def build_longest_n_set(longest_manifest: dict, candidate_index: dict[str, dict], n: int) -> list[dict]:
    """Top-N examples by real full-SFT token length (Task 1).

    `longest_manifest` is the real-tokenizer-generated
    `train_longest_examples.json` (already deterministic: real length
    descending, `example_id` tie-break -- see
    `scripts/run_qlora_smoke.py`'s `write_longest_examples_manifest`).
    This function only takes its first `n` entries and copies the
    corresponding candidate records verbatim -- no re-ranking, no
    mutation, no truncation.
    """
    entries = sorted(longest_manifest["entries"], key=lambda e: e["rank"])[:n]
    ids = [e["example_id"] for e in entries]
    if len(set(ids)) != len(ids):
        raise AssertionError("longest-N manifest contains duplicate example_ids")
    missing = [eid for eid in ids if eid not in candidate_index]
    if missing:
        raise AssertionError(f"example_id(s) in manifest not found in candidate dataset: {missing}")
    return [dict(candidate_index[eid]) for eid in ids]


# --- Real per-example token-length manifest (canonical throughput input) ---

TOKEN_LENGTH_RECORD_FIELDS = (
    "example_id",
    "db_id",
    "representation",
    "prompt_only_token_count",
    "full_sft_token_count",
    "completion_token_count",
    "prefix_boundary_match",
)


def write_token_length_manifest(path: Path, records: list[dict]) -> str:
    """Write the REAL per-example token-length manifest that is the sole
    input to canonical Phase 5B throughput sampling.

    Every record must already carry real Qwen-tokenizer counts (produced
    by `localsql.train.sft_data.build_sft_encoding` against the resolved
    tokenizer/chat-template) -- this function only validates shape and
    determinism, it never computes or estimates a token count itself.

    Determinism: written sorted by `example_id` (source-JSONL-order
    independent), one `json.dumps(..., sort_keys=True)` per line (stable
    key order regardless of dict construction order). Raises on duplicate
    `example_id`s or a record missing a required field. Returns the
    written file's SHA256.
    """
    missing_fields = [
        (r.get("example_id", "<unknown>"), field)
        for r in records
        for field in TOKEN_LENGTH_RECORD_FIELDS
        if field not in r
    ]
    if missing_fields:
        raise AssertionError(f"token-length manifest record(s) missing required field(s): {missing_fields[:10]}")

    ids = [r["example_id"] for r in records]
    if len(set(ids)) != len(ids):
        raise AssertionError("duplicate example_id in token-length manifest records")

    ordered = sorted(records, key=lambda r: r["example_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in ordered:
            f.write(json.dumps({k: r[k] for k in TOKEN_LENGTH_RECORD_FIELDS}, sort_keys=True) + "\n")
    return file_sha256(path)


def load_token_length_manifest(path: Path) -> list[dict]:
    """Load a real per-example token-length manifest written by
    `write_token_length_manifest` (or an equivalent real-tokenizer run)."""
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


class ThroughputSampleRepresentationError(RuntimeError):
    """Raised when the deterministic equal-rank throughput sample would
    contain only one of `compact_full_schema` / `canonical_unchanged` --
    fail closed rather than silently substituting a hand-picked example
    to force both representations to appear."""


def select_equally_spaced_by_real_rank(sorted_records: list[Any], target_n: int = 64) -> list[Any]:
    """Deterministic equally-spaced rank selection over an already-sorted
    population (Phase 5B canonical throughput sampling, real-token-manifest
    hardening pass).

    `sorted_records` must already be sorted by the caller (ascending, by
    real full-SFT token length with an `example_id` tie-break -- see
    `build_real_token_throughput_sample`). This function only implements
    the generic equal-rank-selection algorithm so it can be tested/reused
    independently of that sort key.

    Documented rounding rule: for i = 0..count-1 (count = min(target_n,
    N)), target_rank = round_half_up(i * (N - 1) / (count - 1)) --
    i.e. `int(raw + 0.5)`, not Python's banker's-rounding `round()`, so
    the rule is unambiguous and reproducible outside Python too. Ranks
    are deduplicated (preserving order) if a collision occurs, which can
    only happen when N is small relative to `target_n`.
    """
    n = len(sorted_records)
    if n == 0:
        return []
    count = min(target_n, n)
    if count == 1:
        return [sorted_records[0]]

    ranks: list[int] = []
    for i in range(count):
        raw = i * (n - 1) / (count - 1)
        rank = int(raw + 0.5)  # round-half-up, documented above
        ranks.append(rank)
    unique_ranks = list(dict.fromkeys(ranks))
    return [sorted_records[r] for r in unique_ranks]


def build_real_token_throughput_sample(
    token_length_records: list[dict],
    candidate_index: dict[str, dict],
    target_n: int = 64,
) -> list[dict]:
    """Canonical Phase 5B throughput sample (real-token-manifest hardening
    pass) -- REPLACES the earlier estimator-based
    `build_stratified_throughput_sample` (removed; not accepted for the
    canonical benchmark).

    1. Sort `token_length_records` by (`full_sft_token_count`,
       `example_id`) ascending -- real Qwen-tokenizer counts only, never
       the local character-count estimator.
    2. Select `target_n` (default 64) approximately equally spaced ranks
       across the full sorted population via
       `select_equally_spaced_by_real_rank` (documented round-half-up
       rule), spanning short to longest.
    3. Map selected `example_id`s to exact candidate-dataset record
       copies -- no mutation, no truncation, no gold SQL involved
       anywhere in selection.

    Raises `AssertionError` if a selected id is missing from
    `candidate_index` or duplicated. Raises
    `ThroughputSampleRepresentationError` -- fail closed, no silent
    substitution -- if the resulting sample contains only one of
    `compact_full_schema` / `canonical_unchanged`.
    """
    sorted_records = sorted(token_length_records, key=lambda r: (r["full_sft_token_count"], r["example_id"]))
    selected = select_equally_spaced_by_real_rank(sorted_records, target_n=target_n)

    ids = [r["example_id"] for r in selected]
    if len(set(ids)) != len(ids):
        raise AssertionError("equal-rank throughput selection produced duplicate example_ids")
    missing = [eid for eid in ids if eid not in candidate_index]
    if missing:
        raise AssertionError(f"example_id(s) selected for throughput sample not found in candidate dataset: {missing}")

    sample = [dict(candidate_index[eid]) for eid in ids]

    has_representation_field = bool(sample) and all("representation" in r for r in sample)
    representations = {r["representation"] for r in sample if "representation" in r}
    if has_representation_field and len(representations) < 2:
        raise ThroughputSampleRepresentationError(
            f"deterministic equal-rank throughput sample contains only representation(s) {representations} -- "
            "expected both compact_full_schema and canonical_unchanged. Refusing to silently substitute a "
            "hand-picked example; this is a BLOCKER, review the candidate dataset / target_n instead."
        )
    return sample


def summarize_lengths(examples: list[dict], lengths: dict[str, float]) -> dict:
    """Real (or, for legacy callers, estimated) length summary + schema
    representation counts for a report. `lengths` must be keyed by
    `example_id` and should be real full-SFT token counts for the
    canonical Phase 5B throughput report."""
    from localsql.model.run_artifacts import numeric_stats

    values = [lengths[e["example_id"]] for e in examples if e["example_id"] in lengths]
    stats = numeric_stats(values)
    stats["mean"] = round(sum(values) / len(values), 2) if values else None
    reprs: dict[str, int] = {}
    for e in examples:
        reprs[e["representation"]] = reprs.get(e["representation"], 0) + 1
    return {**stats, "representation_counts": reprs}
