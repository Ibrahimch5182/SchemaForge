"""Contract tests for scripts/build_phase5_candidate.py. CPU/offline,
synthetic schema/examples only.
"""

import importlib.util
import json
from pathlib import Path

from localsql.data.models import PreparedTextToSQLExample, SourceMetadata

from tests.schema_context.fixtures import make_shop_schema

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "build_phase5_candidate.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("build_phase5_candidate_under_test", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _example(idx, db_id, sql="SELECT customer_id FROM customers", split="train"):
    ex = PreparedTextToSQLExample(
        example_id=f"synthetic:{idx:04d}",
        db_id=db_id,
        dialect="sqlite",
        question=f"question {idx}",
        business_context=None,
        schema_serialized="ORIGINAL_SCHEMA_TEXT_UNCHANGED",
        prompt=f"ORIGINAL_PROMPT_{idx}_UNCHANGED",
        completion=sql,
        source=SourceMetadata(dataset_repo_id="synthetic", row_index=idx, evidence_available=False, business_context_kept=False),
        split=split,
    )
    return json.loads(ex.model_dump_json())


def test_build_candidate_split_compacts_only_flagged_dbs():
    module = _load_module()
    schema = make_shop_schema()
    examples = [_example(0, "shop_db"), _example(1, "other_db")]
    schemas = {"shop_db": schema, "other_db": schema}

    candidate = module.build_candidate_split("train", examples, schemas, {"shop_db"}, "test_basis")

    shop_record = next(c for c in candidate if c["db_id"] == "shop_db")
    other_record = next(c for c in candidate if c["db_id"] == "other_db")
    assert shop_record["representation"] == "compact_full_schema"
    assert other_record["representation"] == "canonical_unchanged"


def test_unchanged_db_examples_are_byte_identical_to_original():
    module = _load_module()
    schema = make_shop_schema()
    examples = [_example(0, "other_db")]
    candidate = module.build_candidate_split("train", examples, {"other_db": schema}, set(), "test_basis")

    assert candidate[0]["prompt"] == "ORIGINAL_PROMPT_0_UNCHANGED"
    assert candidate[0]["serialized_schema"] == "ORIGINAL_SCHEMA_TEXT_UNCHANGED"


def test_completion_is_byte_identical_for_both_representations():
    module = _load_module()
    schema = make_shop_schema()
    sql = "SELECT customer_id FROM customers WHERE country = 'France'"
    examples = [_example(0, "shop_db", sql=sql), _example(1, "other_db", sql=sql)]
    candidate = module.build_candidate_split("train", examples, {"shop_db": schema, "other_db": schema}, {"shop_db"}, "b")
    assert all(c["completion"] == sql for c in candidate)


def test_no_example_loss_and_db_isolation_preserved():
    module = _load_module()
    schema = make_shop_schema()
    examples = [_example(i, "shop_db" if i % 2 == 0 else "other_db") for i in range(6)]
    schemas = {"shop_db": schema, "other_db": schema}
    candidate = module.build_candidate_split("train", examples, schemas, {"shop_db"}, "b")

    assert len(candidate) == len(examples)
    assert {c["example_id"] for c in candidate} == {e["example_id"] for e in examples}
    assert {c["db_id"] for c in candidate} == {e["db_id"] for e in examples}


def _candidate_shaped(example: dict, representation: str = "canonical_unchanged") -> dict:
    """Build a dict shaped like build_candidate_split's actual output
    (which uses `serialized_schema`, not the original's `schema_serialized`)."""
    return {
        "example_id": example["example_id"],
        "db_id": example["db_id"],
        "split": example["split"],
        "source": example["source"],
        "question": example["question"],
        "business_context": example.get("business_context"),
        "completion": example["completion"],
        "dialect": example["dialect"],
        "representation": representation,
        "policy_basis": "test_basis",
        "serialized_schema": example["schema_serialized"],
        "prompt": example["prompt"],
    }


def test_assert_candidate_integrity_passes_for_valid_data():
    module = _load_module()
    examples = [_example(0, "shop_db"), _example(1, "shop_db")]
    candidate = [_candidate_shaped(e) for e in examples]
    module.assert_candidate_integrity("train", candidate, examples, expected_count=2)  # must not raise


def test_assert_candidate_integrity_catches_dropped_example():
    module = _load_module()
    examples = [_example(0, "shop_db"), _example(1, "shop_db")]
    candidate = [dict(examples[0], representation="canonical_unchanged")]
    try:
        module.assert_candidate_integrity("train", candidate, examples, expected_count=2)
    except AssertionError:
        pass
    else:
        raise AssertionError("expected dropped example to be caught")


def test_assert_candidate_integrity_catches_completion_mutation():
    module = _load_module()
    examples = [_example(0, "shop_db", sql="SELECT 1")]
    candidate = [dict(examples[0], representation="canonical_unchanged")]
    candidate[0]["completion"] = "SELECT 2"
    try:
        module.assert_candidate_integrity("train", candidate, examples, expected_count=1)
    except AssertionError:
        pass
    else:
        raise AssertionError("expected mutated completion to be caught")


def test_assert_candidate_integrity_catches_wrong_expected_count():
    module = _load_module()
    examples = [_example(0, "shop_db")]
    candidate = [dict(examples[0], representation="canonical_unchanged")]
    try:
        module.assert_candidate_integrity("train", candidate, examples, expected_count=6067)
    except AssertionError:
        pass
    else:
        raise AssertionError("expected wrong count to be caught")


def test_structural_preservation_passes_for_compact_schema():
    from localsql.schema_context.compact_serializer import serialize_schema_compact

    module = _load_module()
    schema = make_shop_schema()
    compact_text = serialize_schema_compact(schema)
    module.assert_structural_preservation(schema, compact_text)  # must not raise


def test_structural_preservation_scopes_columns_to_their_own_table():
    """Regression: a naive whole-text search for a column name could match
    the wrong table's line when column names repeat across tables."""
    from localsql.data.models import ColumnSchema, DatabaseSchema, TableSchema

    module = _load_module()
    # Two tables sharing a column name "id"; only table_a's "id" is a PK.
    table_a = TableSchema(name="table_a", columns=[ColumnSchema(name="id", data_type="INTEGER", is_primary_key=True)])
    table_b = TableSchema(name="table_b", columns=[ColumnSchema(name="id", data_type="INTEGER", is_primary_key=False)])
    schema = DatabaseSchema(db_id="dup_cols_db", dialect="sqlite", tables=[table_a, table_b])

    from localsql.schema_context.compact_serializer import serialize_schema_compact

    compact_text = serialize_schema_compact(schema)
    module.assert_structural_preservation(schema, compact_text)  # must not raise (correct per-table scoping)


def test_build_candidate_split_is_reproducible():
    module = _load_module()
    schema = make_shop_schema()
    examples = [_example(0, "shop_db"), _example(1, "other_db")]
    schemas = {"shop_db": schema, "other_db": schema}

    a = module.build_candidate_split("train", examples, schemas, {"shop_db"}, "b")
    b = module.build_candidate_split("train", examples, schemas, {"shop_db"}, "b")
    assert a == b
