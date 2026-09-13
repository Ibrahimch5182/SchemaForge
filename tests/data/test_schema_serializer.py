from localsql.data.schema_serializer import serialize_schema

from tests.data.fixtures import make_shop_schema


def test_serialization_is_deterministic():
    schema = make_shop_schema()
    first = serialize_schema(schema)
    second = serialize_schema(schema)
    assert first == second


def test_serialization_contains_table_and_column_names():
    text = serialize_schema(make_shop_schema())
    assert "customers(" in text
    assert "orders(" in text
    assert "customer_id" in text
    assert "order_date DATE" in text


def test_serialization_annotates_primary_key():
    text = serialize_schema(make_shop_schema())
    assert "customer_id INTEGER PK" in text


def test_serialization_annotates_foreign_key():
    text = serialize_schema(make_shop_schema())
    assert "FK->customers.customer_id" in text


def test_serialization_does_not_invent_fields():
    text = serialize_schema(make_shop_schema())
    # 'segment' has no PK/FK and only a description; must not gain fabricated
    # PK/FK annotations.
    segment_line = next(line for line in text.splitlines() if "segment" in line)
    assert "PK" not in segment_line
    assert "FK->" not in segment_line
