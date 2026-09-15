"""Contract tests for scripts/analyze_schema_context.py's derived-dataset
writer (Task 7). CPU/offline, synthetic data only.
"""

import importlib.util
import json
from pathlib import Path

from localsql.data.models import PreparedTextToSQLExample, SourceMetadata

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "analyze_schema_context.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("analyze_schema_context_under_test", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _example(idx, db_id, sql="SELECT customer_id FROM customers", business_context=None):
    ex = PreparedTextToSQLExample(
        example_id=f"synthetic:{idx:04d}",
        db_id=db_id,
        dialect="sqlite",
        question=f"question {idx}",
        business_context=business_context,
        schema_serialized="customers(\n  customer_id INTEGER PK\n)",
        prompt=f"SYSTEM:\n...\n\nQUESTION:\nquestion {idx}",
        completion=sql,
        source=SourceMetadata(
            dataset_repo_id="synthetic", row_index=idx, evidence_available=False, business_context_kept=False
        ),
        split="train",
    )
    return json.loads(ex.model_dump_json())


def test_build_derived_records_drops_no_examples_and_preserves_order():
    from tests.schema_context.fixtures import make_shop_schema

    module = _load_script_module()
    schema = make_shop_schema()
    examples = [_example(i, "shop_db") for i in range(5)]
    schemas = {"shop_db": schema}
    from localsql.schema_context.compact_serializer import serialize_schema_compact

    compact_cache = {"shop_db": serialize_schema_compact(schema)}

    derived = module.build_derived_records(examples, schemas, compact_cache, budget_tokens=100_000)

    assert len(derived) == len(examples)
    assert [d["example_id"] for d in derived] == [e["example_id"] for e in examples]


def test_build_derived_records_never_mutates_completion():
    from tests.schema_context.fixtures import make_shop_schema
    from localsql.schema_context.compact_serializer import serialize_schema_compact

    module = _load_script_module()
    schema = make_shop_schema()
    sql = "SELECT customer_id FROM customers WHERE country = 'France'"
    examples = [_example(0, "shop_db", sql=sql)]
    derived = module.build_derived_records(
        examples, {"shop_db": schema}, {"shop_db": serialize_schema_compact(schema)}, budget_tokens=1  # tiny
    )
    assert derived[0]["completion"] == sql


def test_assert_derived_dataset_integrity_passes_for_matching_data():
    module = _load_script_module()
    examples = [_example(0, "shop_db"), _example(1, "shop_db")]
    derived = [dict(e) for e in examples]
    module.assert_derived_dataset_integrity(derived, examples, expected_count=2)  # must not raise


def test_assert_derived_dataset_integrity_catches_dropped_example():
    module = _load_script_module()
    examples = [_example(0, "shop_db"), _example(1, "shop_db")]
    derived = [dict(examples[0])]  # dropped example 1
    try:
        module.assert_derived_dataset_integrity(derived, examples, expected_count=2)
    except AssertionError:
        pass
    else:
        raise AssertionError("expected a dropped example to be caught")


def test_assert_derived_dataset_integrity_catches_mutated_completion():
    module = _load_script_module()
    examples = [_example(0, "shop_db", sql="SELECT 1")]
    derived = [dict(examples[0])]
    derived[0]["completion"] = "SELECT 2"  # mutated
    try:
        module.assert_derived_dataset_integrity(derived, examples, expected_count=1)
    except AssertionError:
        pass
    else:
        raise AssertionError("expected a mutated completion to be caught")


def test_write_derived_dataset_preserves_db_isolation_across_splits(tmp_path):
    """Train/validation db isolation must be unaffected by this
    transformation -- the derived writer never touches split assignment."""
    from tests.schema_context.fixtures import make_shop_schema
    from localsql.schema_context.compact_serializer import serialize_schema_compact

    module = _load_script_module()
    schema = make_shop_schema()
    train_examples = [_example(i, "shop_db") for i in range(3)]
    schemas = {"shop_db": schema}
    compact_cache = {"shop_db": serialize_schema_compact(schema)}

    out_dir = tmp_path / "derived"
    out_path = module.write_derived_dataset(
        "unittest_split", train_examples, schemas, compact_cache, budget_tokens=100_000, out_dir=out_dir
    )
    written = [json.loads(l) for l in out_path.read_text(encoding="utf-8").splitlines()]
    assert {d["db_id"] for d in written} == {"shop_db"}
    assert len(written) == 3

    provenance = json.loads((out_dir / "unittest_split_provenance.json").read_text(encoding="utf-8"))
    assert provenance["derived_example_count"] == 3
    assert provenance["original_example_count"] == 3


def test_full_schema_examples_under_budget_are_unchanged_by_derivation():
    """If the whole compact schema already fits the budget, the derived
    record's schema must equal the plain compact serialization -- the
    budgeter must not run/alter anything for in-budget examples."""
    from tests.schema_context.fixtures import make_shop_schema
    from localsql.schema_context.compact_serializer import serialize_schema_compact

    module = _load_script_module()
    schema = make_shop_schema()
    compact_text = serialize_schema_compact(schema)
    examples = [_example(0, "shop_db")]

    derived = module.build_derived_records(
        examples, {"shop_db": schema}, {"shop_db": compact_text}, budget_tokens=100_000
    )
    assert derived[0]["schema_representation"] == "compact_full"
    assert derived[0]["serialized_schema"] == compact_text


def test_threshold_counts_helper():
    module = _load_script_module()
    counts = module.threshold_counts([100.0, 5000.0, 9000.0], thresholds=(3584, 4096, 8192))
    assert counts == {">3584": 2, ">4096": 2, ">8192": 1}
