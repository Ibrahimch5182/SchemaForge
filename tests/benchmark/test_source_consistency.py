"""Regression test for the HF-vs-archive source-consistency check.

Added after diagnosing the Phase 2 oracle-sanity 96% EX result: HF's
`bird_mini_dev` `mini_dev_sqlite` split diverges from the official
`minidev.zip` archive's own gold SQL at 18/500 positions (2 corrected/
differing gold SQL rows, plus a 16-row block where the archive contains
duplicate questions that HF's set does not). `analyze_source_consistency`
must catch this class of divergence automatically on any future setup run.
"""

from localsql.benchmark.minidev_loader import analyze_source_consistency


def _row(question: str, sql: str) -> dict:
    return {"question": question, "SQL": sql}


def test_detects_no_mismatch_when_sources_agree():
    hf = [_row("Q1", "SELECT 1"), _row("Q2", "SELECT 2")]
    archive = [_row("Q1", "SELECT 1"), _row("Q2", "SELECT 2")]
    gold_pairs = [("SELECT 1", "db1"), ("SELECT 2", "db1")]

    result = analyze_source_consistency(hf, archive, gold_pairs)

    assert result["archive_internal_mismatch_count"] == 0
    assert result["hf_vs_archive_question_mismatch_count"] == 0
    assert result["hf_vs_archive_sql_mismatch_count"] == 0


def test_detects_hf_vs_archive_sql_divergence_same_question():
    # Mirrors idx 32 / 194: same question, corrected/different SQL.
    hf = [_row("Same question", "SELECT fixed_version")]
    archive = [_row("Same question", "SELECT old_version")]
    gold_pairs = [("SELECT old_version", "db1")]

    result = analyze_source_consistency(hf, archive, gold_pairs)

    assert result["hf_vs_archive_question_mismatch_count"] == 0
    assert result["hf_vs_archive_sql_mismatch_indices"] == [0]
    assert result["archive_internal_mismatch_count"] == 0  # archive is self-consistent


def test_detects_reordered_block_as_question_and_sql_mismatch():
    # Mirrors idx 484-499: a block where archive duplicates rows HF doesn't.
    hf = [_row("A", "SQL_A"), _row("B", "SQL_B"), _row("C", "SQL_C")]
    archive = [_row("B", "SQL_B"), _row("C", "SQL_C"), _row("B", "SQL_B")]  # duplicated row
    gold_pairs = [("SQL_B", "db1"), ("SQL_C", "db1"), ("SQL_B", "db1")]

    result = analyze_source_consistency(hf, archive, gold_pairs)

    assert result["archive_internal_mismatch_count"] == 0  # archive agrees with its own gold.sql
    assert result["hf_vs_archive_question_mismatch_indices"] == [0, 1, 2]
    assert result["hf_vs_archive_sql_mismatch_indices"] == [0, 1, 2]


def test_flags_archive_internal_inconsistency():
    # The one case that would indicate OUR pipeline mis-paired archive's own
    # question file against its own gold.sql (an adapter/alignment bug).
    hf = [_row("Q1", "SELECT 1")]
    archive = [_row("Q1", "SELECT 1")]
    gold_pairs = [("SELECT DIFFERENT", "db1")]

    result = analyze_source_consistency(hf, archive, gold_pairs)

    assert result["archive_internal_mismatch_indices"] == [0]


def test_raises_on_row_count_mismatch():
    try:
        analyze_source_consistency([_row("Q", "S")], [], [])
    except AssertionError:
        pass
    else:
        raise AssertionError("expected a row-count mismatch to raise")
