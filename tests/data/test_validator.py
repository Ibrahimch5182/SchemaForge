from localsql.data.models import PreparedTextToSQLExample, SourceMetadata
from localsql.data.validator import (
    COMPLETION_NOT_PARSEABLE_SQL,
    COMPLETION_NOT_SQL_ONLY,
    DUPLICATE_EXAMPLE_ID,
    EMPTY_PROMPT,
    EMPTY_QUESTION,
    EMPTY_SERIALIZED_SCHEMA,
    EMPTY_SQL,
    MISSING_SCHEMA,
    check_split_leakage,
    validate_prepared_example,
    validate_raw_example,
)

from tests.data.fixtures import make_raw_example, make_shop_schema


def test_rejects_missing_question():
    ex = make_raw_example(question="")
    reasons = validate_raw_example(ex, schemas={"shop": make_shop_schema()})
    assert any(r.reason == EMPTY_QUESTION for r in reasons)


def test_rejects_missing_sql():
    ex = make_raw_example(sql="")
    reasons = validate_raw_example(ex, schemas={"shop": make_shop_schema()})
    assert any(r.reason == EMPTY_SQL for r in reasons)


def test_rejects_missing_schema():
    ex = make_raw_example(db_id="unknown_db")
    reasons = validate_raw_example(ex, schemas={"shop": make_shop_schema()})
    assert any(r.reason == MISSING_SCHEMA for r in reasons)


def test_valid_raw_example_has_no_rejections():
    ex = make_raw_example()
    reasons = validate_raw_example(ex, schemas={"shop": make_shop_schema()})
    assert reasons == []


def _make_prepared(**overrides) -> PreparedTextToSQLExample:
    defaults = dict(
        example_id="src:00000",
        db_id="shop",
        dialect="sqlite",
        question="How many customers are there?",
        business_context=None,
        schema_serialized="customers(\n  customer_id INTEGER PK\n)",
        prompt="SYSTEM:\n...\n\nQUESTION:\nHow many customers are there?",
        completion="SELECT COUNT(*) FROM customers",
        source=SourceMetadata(
            dataset_repo_id="src",
            dataset_revision="abc",
            row_index=0,
            evidence_available=False,
            business_context_kept=False,
        ),
        split="train",
    )
    defaults.update(overrides)
    return PreparedTextToSQLExample(**defaults)


def test_rejects_empty_serialized_schema():
    ex = _make_prepared(schema_serialized="")
    reasons = validate_prepared_example(ex, seen_ids=set())
    assert any(r.reason == EMPTY_SERIALIZED_SCHEMA for r in reasons)


def test_rejects_empty_prompt():
    ex = _make_prepared(prompt="")
    reasons = validate_prepared_example(ex, seen_ids=set())
    assert any(r.reason == EMPTY_PROMPT for r in reasons)


def test_rejects_non_sql_only_completion():
    ex = _make_prepared(completion="```sql\nSELECT 1\n```")
    reasons = validate_prepared_example(ex, seen_ids=set())
    assert any(r.reason == COMPLETION_NOT_SQL_ONLY for r in reasons)


def test_rejects_unparseable_sql_completion():
    ex = _make_prepared(completion="SELECT FROM WHERE")
    reasons = validate_prepared_example(ex, seen_ids=set())
    assert any(r.reason == COMPLETION_NOT_PARSEABLE_SQL for r in reasons)


def test_rejects_duplicate_example_id():
    ex = _make_prepared(example_id="src:00000")
    reasons = validate_prepared_example(ex, seen_ids={"src:00000"})
    assert any(r.reason == DUPLICATE_EXAMPLE_ID for r in reasons)


def test_valid_prepared_example_has_no_rejections():
    ex = _make_prepared()
    reasons = validate_prepared_example(ex, seen_ids=set())
    assert reasons == []


def test_check_split_leakage_raises_on_overlap():
    try:
        check_split_leakage({"a", "b"}, {"b", "c"})
    except AssertionError as e:
        assert "b" in str(e)
    else:
        raise AssertionError("expected leakage to raise")


def test_check_split_leakage_passes_when_disjoint():
    result = check_split_leakage({"a", "b"}, {"c", "d"})
    assert result == set()
