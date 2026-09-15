import inspect

from localsql.schema_context.relevance import (
    add_fk_closure,
    normalize_identifier,
    score_table_relevance,
    select_schema_within_budget,
    tokenize_text,
)

from tests.schema_context.fixtures import make_shop_schema


def test_selector_api_cannot_accept_gold_sql_by_construction():
    """The whole point of the inference-time-safe design: there must be no
    parameter through which gold SQL/tables/columns could even be passed,
    not just a convention not to pass one."""
    params = set(inspect.signature(select_schema_within_budget).parameters)
    forbidden = {"sql", "gold_sql", "gold", "target", "target_sql", "answer", "gold_tables", "gold_columns"}
    assert not (params & forbidden), f"selector accepts forbidden gold-like params: {params & forbidden}"


def test_normalize_identifier_handles_snake_and_camel_case():
    assert normalize_identifier("customer_id") == {"customer", "id"}
    assert normalize_identifier("CustomerID") == {"customer", "id"}


def test_add_fk_closure_pulls_in_two_hop_targets():
    schema = make_shop_schema()
    closure = add_fk_closure(schema, frozenset({"order_items"}))
    # order_items FKs to orders and products; orders itself FKs to
    # customers -- the closure must chase the full FK chain, not just one hop.
    assert closure == frozenset({"order_items", "orders", "products", "customers"})


def test_add_fk_closure_is_a_fixed_point_noop_on_already_closed_set():
    schema = make_shop_schema()
    already_closed = frozenset({"order_items", "orders", "products", "customers"})
    closure = add_fk_closure(schema, already_closed)
    assert closure == already_closed


def test_selection_is_deterministic_across_repeated_calls():
    schema = make_shop_schema()
    results = [
        select_schema_within_budget(schema, "list order items and customers", None, budget=10_000)
        for _ in range(3)
    ]
    texts = {r.serialized_schema for r in results}
    assert len(texts) == 1
    tables = {r.selected_table_names for r in results}
    assert len(tables) == 1


def test_selection_pulls_relevant_table_plus_fk_closure():
    schema = make_shop_schema()
    result = select_schema_within_budget(schema, "list order items and customers", None, budget=10_000)
    selected = set(result.selected_table_names)
    assert "order_items" in selected
    assert "customers" in selected
    # FK closure from order_items should have pulled in orders and products
    # even though the question didn't name them directly.
    assert "orders" in selected
    assert "products" in selected


def test_irrelevant_table_excluded_when_budget_is_tight():
    """With an ample budget the policy keeps the whole schema (see
    test_full_schema_under_budget_is_unchanged); exclusion of irrelevant
    tables only has to happen once the budget actually forces a choice."""
    schema = make_shop_schema()
    # 385 chars fits {customers, orders, products, order_items}; adding the
    # irrelevant warehouse_logs would push it to 437 -- budget=400 forces
    # the choice.
    result = select_schema_within_budget(schema, "list order items and customers", None, budget=400)
    assert "warehouse_logs" not in result.selected_table_names
    assert {"customers", "orders", "products", "order_items"} <= set(result.selected_table_names)


def test_full_schema_under_budget_is_unchanged():
    """If the whole compact schema already fits the budget, the policy is
    to keep everything -- no table should be dropped just because the
    question didn't happen to mention it."""
    schema = make_shop_schema()
    from localsql.schema_context.compact_serializer import serialize_schema_compact

    full_text = serialize_schema_compact(schema)
    result = select_schema_within_budget(schema, "an unrelated question about nothing here", None, budget=len(full_text) + 100)
    assert set(result.selected_table_names) == {t.name for t in schema.tables}
    assert result.within_budget is True


def test_token_budget_is_enforced_and_never_partially_cuts_a_table():
    schema = make_shop_schema()
    result = select_schema_within_budget(schema, "list order items and customers", None, budget=40)
    assert result.estimated_length <= 40 or result.fell_back_to_single_table
    # Every selected table's opening line must appear intact -- no
    # mid-identifier truncation.
    for name in result.selected_table_names:
        assert f"{name}(" in result.serialized_schema


def test_never_returns_empty_selection_even_at_pathological_budget():
    schema = make_shop_schema()
    result = select_schema_within_budget(schema, "list order items and customers", None, budget=1)
    assert len(result.selected_table_names) >= 1
    assert result.fell_back_to_single_table is True


def test_pk_and_fk_annotations_preserved_in_selected_output():
    schema = make_shop_schema()
    result = select_schema_within_budget(schema, "list order items and customers", None, budget=10_000)
    assert "customer_id INTEGER PK" in result.serialized_schema
    assert "FK->orders.order_id" in result.serialized_schema
    assert "FK->products.product_id" in result.serialized_schema


def test_business_context_terms_also_influence_relevance():
    schema = make_shop_schema()
    question_terms = tokenize_text("show me data")
    context_terms = tokenize_text("products refers to the products table")
    table = next(t for t in schema.tables if t.name == "products")
    score_without_context = score_table_relevance(table, question_terms)
    score_with_context = score_table_relevance(table, question_terms | context_terms)
    assert score_with_context.score > score_without_context.score
