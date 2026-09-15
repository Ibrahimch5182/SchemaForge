from localsql.data.schema_serializer import serialize_schema
from localsql.schema_context.compact_serializer import serialize_schema_compact

from tests.schema_context.fixtures import make_shop_schema


def test_compact_serialization_is_deterministic():
    schema = make_shop_schema()
    assert serialize_schema_compact(schema) == serialize_schema_compact(schema)


def test_compact_serialization_preserves_all_identifiers_types_pk_fk():
    text = serialize_schema_compact(make_shop_schema())
    assert "customers(" in text
    assert "orders(" in text
    assert "order_items(" in text
    assert "customer_id INTEGER PK" in text
    assert "order_id INTEGER FK->orders.order_id" in text
    assert "product_id INTEGER FK->products.product_id" in text
    assert "quantity INTEGER" in text


def test_compact_serialization_removes_descriptions_present_in_canonical():
    canonical = serialize_schema(make_shop_schema())
    compact = serialize_schema_compact(make_shop_schema())
    assert "Customer full name." in canonical
    assert "Customer full name." not in compact
    assert "References the ordering customer." in canonical
    assert "References the ordering customer." not in compact


def test_compact_is_strictly_shorter_than_canonical_when_descriptions_present():
    canonical = serialize_schema(make_shop_schema())
    compact = serialize_schema_compact(make_shop_schema())
    assert len(compact) < len(canonical)
