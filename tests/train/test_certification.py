"""Contract tests for Phase 5B Task 1/3: certification dataset construction.

CPU/offline, synthetic data only -- no model, no CUDA, no network, and no
gold SQL involved in selection. The throughput-sampling tests here cover
the REAL-token-manifest hardening pass -- the earlier estimator-based
`build_stratified_throughput_sample` was removed; it is not accepted for
the canonical Phase 5B throughput benchmark.
"""

import json

import pytest

from localsql.train.certification import (
    ThroughputSampleRepresentationError,
    build_longest_n_set,
    build_real_token_throughput_sample,
    file_sha256,
    load_jsonl_index,
    load_token_length_manifest,
    select_equally_spaced_by_real_rank,
    summarize_lengths,
    write_token_length_manifest,
)


def _candidate(idx: int, representation: str = "canonical_unchanged") -> dict:
    return {
        "example_id": f"synthetic:{idx:04d}",
        "db_id": "db_a",
        "split": "train",
        "source": {"dataset_repo_id": "synthetic", "row_index": idx},
        "question": f"question {idx}",
        "business_context": None,
        "completion": f"SELECT {idx}",
        "dialect": "sqlite",
        "representation": representation,
        "policy_basis": "test",
        "serialized_schema": "SCHEMA",
        "prompt": "PROMPT_" + "x" * idx,
    }


def _token_length_record(example_id: str, full_sft_token_count: float, representation: str = "canonical_unchanged") -> dict:
    return {
        "example_id": example_id,
        "db_id": "db_a",
        "representation": representation,
        "prompt_only_token_count": max(full_sft_token_count - 5, 0),
        "full_sft_token_count": full_sft_token_count,
        "completion_token_count": 5,
        "prefix_boundary_match": True,
    }


def _manifest(entries: list[tuple[str, float]]) -> dict:
    return {
        "profile_name": "test",
        "top_n": len(entries),
        "total_examples_considered": len(entries),
        "entries": [
            {"rank": i + 1, "example_id": eid, "full_sft_token_count": count} for i, (eid, count) in enumerate(entries)
        ],
    }


# --- longest-N set (Task 1, unchanged methodology) ---


def test_build_longest_n_set_returns_exact_n_unique_examples():
    candidates = {c["example_id"]: c for c in [_candidate(i) for i in range(20)]}
    entries = sorted(((eid, float(len(c["prompt"]))) for eid, c in candidates.items()), key=lambda p: -p[1])
    manifest = _manifest(entries)

    longest = build_longest_n_set(manifest, candidates, n=16)

    assert len(longest) == 16
    assert len({e["example_id"] for e in longest}) == 16


def test_build_longest_n_set_is_descending_by_real_length():
    candidates = {c["example_id"]: c for c in [_candidate(i) for i in range(10)]}
    entries = sorted(((eid, float(len(c["prompt"]))) for eid, c in candidates.items()), key=lambda p: -p[1])
    manifest = _manifest(entries)

    longest = build_longest_n_set(manifest, candidates, n=5)

    lengths = [len(e["prompt"]) for e in longest]
    assert lengths == sorted(lengths, reverse=True)


def test_build_longest_n_set_records_are_byte_identical_copies():
    candidates = {c["example_id"]: c for c in [_candidate(i) for i in range(5)]}
    entries = [(eid, float(len(c["prompt"]))) for eid, c in candidates.items()]
    manifest = _manifest(entries)

    longest = build_longest_n_set(manifest, candidates, n=3)

    for record in longest:
        original = candidates[record["example_id"]]
        assert record == original
        assert record is not original  # a copy, not the same object


def test_build_longest_n_set_raises_on_duplicate_manifest_ids():
    candidates = {c["example_id"]: c for c in [_candidate(i) for i in range(3)]}
    manifest = _manifest([("synthetic:0000", 10.0), ("synthetic:0000", 9.0)])
    with pytest.raises(AssertionError):
        build_longest_n_set(manifest, candidates, n=2)


def test_build_longest_n_set_raises_on_missing_candidate():
    candidates = {c["example_id"]: c for c in [_candidate(0)]}
    manifest = _manifest([("synthetic:9999", 10.0)])
    with pytest.raises(AssertionError):
        build_longest_n_set(manifest, candidates, n=1)


def test_build_longest_n_set_does_not_mutate_original_candidate_dict():
    candidates = {c["example_id"]: c for c in [_candidate(i) for i in range(3)]}
    manifest = _manifest([(eid, float(len(c["prompt"]))) for eid, c in candidates.items()])
    before = json.dumps(candidates, sort_keys=True)

    build_longest_n_set(manifest, candidates, n=2)

    assert json.dumps(candidates, sort_keys=True) == before


# --- real per-example token-length manifest (write/load) ---


def test_write_token_length_manifest_writes_all_required_fields(tmp_path):
    records = [_token_length_record(f"synthetic:{i:04d}", float(100 + i)) for i in range(5)]
    path = tmp_path / "train_token_lengths.jsonl"

    sha256 = write_token_length_manifest(path, records)

    assert path.exists()
    assert sha256 == file_sha256(path)
    loaded = load_token_length_manifest(path)
    assert len(loaded) == 5
    for row in loaded:
        assert set(row.keys()) == {
            "example_id",
            "db_id",
            "representation",
            "prompt_only_token_count",
            "full_sft_token_count",
            "completion_token_count",
            "prefix_boundary_match",
        }


def test_write_token_length_manifest_is_sorted_by_example_id_regardless_of_input_order(tmp_path):
    records = [_token_length_record(f"synthetic:{i:04d}", float(100 + i)) for i in reversed(range(5))]
    path = tmp_path / "out.jsonl"

    write_token_length_manifest(path, records)

    loaded = load_token_length_manifest(path)
    assert [r["example_id"] for r in loaded] == sorted(r["example_id"] for r in loaded)


def test_write_token_length_manifest_is_deterministic_byte_for_byte(tmp_path):
    records = [_token_length_record(f"synthetic:{i:04d}", float(100 + i)) for i in range(10)]
    path_a = tmp_path / "a.jsonl"
    path_b = tmp_path / "b.jsonl"

    sha_a = write_token_length_manifest(path_a, list(reversed(records)))
    sha_b = write_token_length_manifest(path_b, records)

    assert sha_a == sha_b
    assert path_a.read_bytes() == path_b.read_bytes()


def test_write_token_length_manifest_rejects_duplicate_example_ids(tmp_path):
    records = [_token_length_record("synthetic:0000", 100.0), _token_length_record("synthetic:0000", 200.0)]
    with pytest.raises(AssertionError):
        write_token_length_manifest(tmp_path / "out.jsonl", records)


def test_write_token_length_manifest_rejects_missing_required_field(tmp_path):
    record = _token_length_record("synthetic:0000", 100.0)
    del record["completion_token_count"]
    with pytest.raises(AssertionError):
        write_token_length_manifest(tmp_path / "out.jsonl", [record])


def test_write_token_length_manifest_exact_example_count(tmp_path):
    records = [_token_length_record(f"synthetic:{i:04d}", float(100 + i)) for i in range(37)]
    path = tmp_path / "out.jsonl"

    write_token_length_manifest(path, records)

    assert len(load_token_length_manifest(path)) == 37


# --- select_equally_spaced_by_real_rank (generic algorithm) ---


def test_select_equally_spaced_by_real_rank_exact_formula_small_case():
    # N=10 sorted records, target_n=5 -> ranks = round_half_up(i*9/4) for i=0..4
    # i=0: 0.0->0, i=1: 2.25->2, i=2: 4.5->5 (round-half-up), i=3: 6.75->7, i=4: 9.0->9
    sorted_records = [{"id": i} for i in range(10)]
    selected = select_equally_spaced_by_real_rank(sorted_records, target_n=5)
    assert [r["id"] for r in selected] == [0, 2, 5, 7, 9]


def test_select_equally_spaced_by_real_rank_includes_both_ends():
    sorted_records = [{"id": i} for i in range(1000)]
    selected = select_equally_spaced_by_real_rank(sorted_records, target_n=64)
    assert selected[0]["id"] == 0
    assert selected[-1]["id"] == 999


def test_select_equally_spaced_by_real_rank_exactly_64_unique_on_large_population():
    sorted_records = [{"id": i} for i in range(6067)]
    selected = select_equally_spaced_by_real_rank(sorted_records, target_n=64)
    ids = [r["id"] for r in selected]
    assert len(ids) == 64
    assert len(set(ids)) == 64


def test_select_equally_spaced_by_real_rank_is_deterministic():
    sorted_records = [{"id": i} for i in range(500)]
    a = select_equally_spaced_by_real_rank(sorted_records, target_n=64)
    b = select_equally_spaced_by_real_rank(sorted_records, target_n=64)
    assert [r["id"] for r in a] == [r["id"] for r in b]


def test_select_equally_spaced_by_real_rank_generic_on_small_fixture():
    """Works generically -- not hardcoded to N=6067 -- including when N <
    target_n (dedupe rather than crash)."""
    sorted_records = [{"id": i} for i in range(3)]
    selected = select_equally_spaced_by_real_rank(sorted_records, target_n=64)
    ids = [r["id"] for r in selected]
    assert len(ids) == len(set(ids))
    assert set(ids).issubset({0, 1, 2})


def test_select_equally_spaced_by_real_rank_empty_input():
    assert select_equally_spaced_by_real_rank([], target_n=64) == []


# --- build_real_token_throughput_sample (canonical Task 3 path) ---


def _build_6067_like_fixture(n=6067, num_flagged_dbs=9):
    """A population shaped like the real candidate set: most examples
    canonical_unchanged, a minority compact_full_schema, real-looking
    ascending token lengths."""
    candidates = {}
    token_lengths = []
    for i in range(n):
        representation = "compact_full_schema" if i % 7 == 0 else "canonical_unchanged"
        c = _candidate(i, representation)
        candidates[c["example_id"]] = c
        token_lengths.append(_token_length_record(c["example_id"], float(400 + i), representation))
    return candidates, token_lengths


def test_build_real_token_throughput_sample_generic_on_6067_like_fixture():
    candidates, token_lengths = _build_6067_like_fixture()
    sample = build_real_token_throughput_sample(token_lengths, candidates, target_n=64)
    assert len(sample) == 64
    assert len({e["example_id"] for e in sample}) == 64


def test_build_real_token_throughput_sample_records_are_exact_copies():
    candidates, token_lengths = _build_6067_like_fixture(n=500)
    sample = build_real_token_throughput_sample(token_lengths, candidates, target_n=32)
    for record in sample:
        assert record == candidates[record["example_id"]]


def test_build_real_token_throughput_sample_is_deterministic():
    candidates, token_lengths = _build_6067_like_fixture(n=500)
    a = build_real_token_throughput_sample(token_lengths, candidates, target_n=64)
    b = build_real_token_throughput_sample(token_lengths, candidates, target_n=64)
    assert [e["example_id"] for e in a] == [e["example_id"] for e in b]


def test_build_real_token_throughput_sample_includes_both_representations():
    candidates, token_lengths = _build_6067_like_fixture()
    sample = build_real_token_throughput_sample(token_lengths, candidates, target_n=64)
    representations = {e["representation"] for e in sample}
    assert representations == {"compact_full_schema", "canonical_unchanged"}


def test_build_real_token_throughput_sample_spans_short_to_long_real_lengths():
    candidates, token_lengths = _build_6067_like_fixture()
    real_lengths = {r["example_id"]: r["full_sft_token_count"] for r in token_lengths}
    sample = build_real_token_throughput_sample(token_lengths, candidates, target_n=64)
    sample_lengths = [real_lengths[e["example_id"]] for e in sample]
    assert min(sample_lengths) < 500  # near the short end
    assert max(sample_lengths) > 6000  # near the long end


def test_build_real_token_throughput_sample_raises_on_single_representation():
    candidates = {c["example_id"]: c for c in [_candidate(i, "canonical_unchanged") for i in range(100)]}
    token_lengths = [_token_length_record(c["example_id"], float(400 + i)) for i, c in enumerate(candidates.values())]
    with pytest.raises(ThroughputSampleRepresentationError):
        build_real_token_throughput_sample(token_lengths, candidates, target_n=16)


def test_build_real_token_throughput_sample_does_not_silently_substitute_on_representation_gap():
    """Fail-closed guarantee: a representation gap raises rather than
    returning a sample with a hand-picked substitution."""
    candidates = {c["example_id"]: c for c in [_candidate(i, "canonical_unchanged") for i in range(20)]}
    token_lengths = [_token_length_record(c["example_id"], float(400 + i)) for i, c in enumerate(candidates.values())]
    try:
        build_real_token_throughput_sample(token_lengths, candidates, target_n=8)
        raised = False
    except ThroughputSampleRepresentationError:
        raised = True
    assert raised


def test_build_real_token_throughput_sample_raises_on_missing_candidate():
    token_lengths = [_token_length_record("synthetic:9999", 100.0)]
    with pytest.raises(AssertionError):
        build_real_token_throughput_sample(token_lengths, {}, target_n=1)


def test_build_real_token_throughput_sample_no_estimator_dependency():
    """Structural guarantee: the canonical selection path never imports or
    calls the local character-count estimator (the module docstring's own
    disclaimer mentioning it by name is fine; an import or call is not)."""
    import inspect

    from localsql.train import certification as mod

    assert "token_estimate" not in dir(mod)
    assert "estimate_tokens" not in dir(mod)
    selection_source = inspect.getsource(build_real_token_throughput_sample) + inspect.getsource(
        select_equally_spaced_by_real_rank
    )
    assert "estimate" not in selection_source.lower()


def test_build_real_token_throughput_sample_signature_has_no_gold_or_sql_parameter():
    import inspect

    params = set(inspect.signature(build_real_token_throughput_sample).parameters)
    assert not {"sql", "gold", "gold_sql"} & params


# --- shared helpers ---


def test_file_sha256_matches_known_content(tmp_path):
    p = tmp_path / "f.txt"
    p.write_bytes(b"hello world")
    import hashlib

    expected = hashlib.sha256(b"hello world").hexdigest()
    assert file_sha256(p) == expected


def test_load_jsonl_index_preserves_all_fields(tmp_path):
    records = [_candidate(0), _candidate(1)]
    p = tmp_path / "data.jsonl"
    with p.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    index = load_jsonl_index(p)

    assert set(index.keys()) == {"synthetic:0000", "synthetic:0001"}
    assert index["synthetic:0000"] == records[0]


def test_summarize_lengths_reports_representation_counts_and_percentiles():
    examples = [_candidate(0, "compact_full_schema"), _candidate(1, "canonical_unchanged"), _candidate(2, "canonical_unchanged")]
    lengths = {e["example_id"]: float(100 * (i + 1)) for i, e in enumerate(examples)}

    summary = summarize_lengths(examples, lengths)

    assert summary["count"] == 3
    assert summary["representation_counts"] == {"compact_full_schema": 1, "canonical_unchanged": 2}
    assert summary["min"] == 100.0
    assert summary["max"] == 300.0
    assert summary["mean"] == pytest.approx(200.0)
