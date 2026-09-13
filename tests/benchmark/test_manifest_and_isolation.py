import json

from localsql.benchmark.manifest_builder import build_manifests, make_example_id
from localsql.benchmark.models import FORBIDDEN_GENERATION_FIELDS

from tests.benchmark.fixtures import ARCHIVE_GOLD_SQL_OVERRIDES, QUESTIONS, write_fixture_workspace


def test_stable_ids_are_deterministic_and_zero_padded():
    assert make_example_id(0) == "bird-mini-dev-sqlite-0000"
    assert make_example_id(499) == "bird-mini-dev-sqlite-0499"


def test_build_manifests_produces_matching_counts(tmp_path):
    ws = write_fixture_workspace(tmp_path)
    generation, grading = build_manifests(
        QUESTIONS, ws["schema_path"], ws["gold_path"], ws["databases_dir"]
    )
    assert len(generation) == len(QUESTIONS)
    assert len(grading) == len(QUESTIONS)


def test_generation_and_grading_join_by_stable_id(tmp_path):
    ws = write_fixture_workspace(tmp_path)
    generation, grading = build_manifests(
        QUESTIONS, ws["schema_path"], ws["gold_path"], ws["databases_dir"]
    )
    gen_ids = [ex.example_id for ex in generation]
    grade_ids = [ex.example_id for ex in grading]
    assert gen_ids == grade_ids
    assert gen_ids == [make_example_id(i) for i in range(len(QUESTIONS))]


def test_generation_manifest_has_no_gold_sql_leakage(tmp_path):
    ws = write_fixture_workspace(tmp_path)
    generation, grading = build_manifests(QUESTIONS, ws["schema_path"], ws["gold_path"], ws["databases_dir"])

    for ex, grade in zip(generation, grading):
        dumped = json.loads(ex.model_dump_json())
        leaked = FORBIDDEN_GENERATION_FIELDS & set(dumped.keys())
        assert not leaked, f"generation example leaked gold fields: {leaked}"
        # Neither the canonical (archive) gold SQL nor HF's diagnostic-only
        # SQL field may appear anywhere in the generation record.
        serialized = json.dumps(dumped)
        matching_question = next(q for q in QUESTIONS if q["question"] == ex.question)
        assert grade.sql not in serialized
        assert matching_question["SQL"] not in serialized


def test_grading_reference_sources_sql_from_archive_gold_not_hf(tmp_path):
    """Canonical grading truth is the archive's mini_dev_sqlite_gold.sql,
    never the HF questions file's `SQL` field (Phase 2A decision)."""
    ws = write_fixture_workspace(tmp_path)
    _, grading = build_manifests(QUESTIONS, ws["schema_path"], ws["gold_path"], ws["databases_dir"])
    for i, (ex, q) in enumerate(zip(grading, QUESTIONS)):
        assert ex.db_id == q["db_id"]
        expected_gold_sql = ARCHIVE_GOLD_SQL_OVERRIDES.get(i, q["SQL"])
        assert ex.sql == expected_gold_sql
        if i in ARCHIVE_GOLD_SQL_OVERRIDES:
            # The whole point of the fixture override: HF's SQL must NOT
            # have silently won out over the archive's differing gold SQL.
            assert ex.sql != q["SQL"]


def test_build_manifests_raises_on_row_count_mismatch(tmp_path):
    ws = write_fixture_workspace(tmp_path)
    try:
        build_manifests(QUESTIONS[:-1], ws["schema_path"], ws["gold_path"], ws["databases_dir"])
    except AssertionError:
        pass
    else:
        raise AssertionError("expected a question/gold row-count mismatch to raise")


def test_manifests_reuse_phase1_prompt_and_schema_format(tmp_path):
    ws = write_fixture_workspace(tmp_path)
    generation, _ = build_manifests(QUESTIONS, ws["schema_path"], ws["gold_path"], ws["databases_dir"])
    ex = generation[0]
    assert "SYSTEM:" in ex.prompt
    assert "SCHEMA:" in ex.prompt
    assert "QUESTION:" in ex.prompt
    assert "customers(" in ex.serialized_schema
    assert "customer_id INTEGER PK" in ex.serialized_schema


def test_business_context_preserves_evidence_without_dropout(tmp_path):
    ws = write_fixture_workspace(tmp_path)
    generation, _ = build_manifests(QUESTIONS, ws["schema_path"], ws["gold_path"], ws["databases_dir"])
    with_evidence = [ex for ex in generation if ex.business_context]
    # QUESTIONS[1] has non-empty evidence and must retain it (no 50% dropout here).
    assert len(with_evidence) == 1
    assert "France" in with_evidence[0].business_context
