from localsql.schema_context.coverage import compute_coverage, extract_gold_references, summarize_coverage

SCHEMA_TABLES = {
    "customers": ["customer_id", "name", "country"],
    "orders": ["order_id", "customer_id", "order_date"],
    "products": ["product_id", "product_name", "price"],
}


def test_extract_gold_references_finds_tables_and_columns():
    sql = "SELECT T1.name FROM customers AS T1 WHERE T1.country = 'France'"
    gold = extract_gold_references(sql, SCHEMA_TABLES)
    assert gold is not None
    assert "customers" in gold.tables
    assert "name" in gold.columns


def test_extract_gold_references_handles_joins():
    sql = (
        "SELECT c.name, o.order_date FROM customers c "
        "JOIN orders o ON c.customer_id = o.customer_id WHERE o.order_date > '2020-01-01'"
    )
    gold = extract_gold_references(sql, SCHEMA_TABLES)
    assert gold is not None
    assert gold.tables == frozenset({"customers", "orders"})


def test_extract_gold_references_returns_none_for_unparseable_sql():
    gold = extract_gold_references("SELECT FROM WHERE", SCHEMA_TABLES)
    assert gold is None


def test_compute_coverage_full_retention():
    from localsql.schema_context.coverage import GoldReferences

    gold = GoldReferences(tables=frozenset({"customers", "orders"}), columns=frozenset({"name", "order_date"}))
    result = compute_coverage("ex-0", frozenset({"customers", "orders", "products"}), gold)
    assert result.retained_all_tables is True
    assert result.retained_all_columns is True
    assert result.gold_tables_retained == 2


def test_compute_coverage_partial_retention():
    from localsql.schema_context.coverage import GoldReferences

    gold = GoldReferences(tables=frozenset({"customers", "orders"}), columns=frozenset({"name", "order_date"}))
    result = compute_coverage("ex-0", frozenset({"customers"}), gold)
    assert result.retained_all_tables is False
    assert result.gold_tables_retained == 1
    # whole-table granularity: columns from the dropped table are not "retained"
    assert result.retained_all_columns is False


def test_compute_coverage_reports_unparseable_separately():
    result = compute_coverage("ex-0", frozenset({"customers"}), None)
    assert result.parseable is False
    summary = summarize_coverage([result])
    assert summary["unparseable_or_ambiguous_examples"] == 1
    assert summary["parseable_examples"] == 0


def test_summarize_coverage_aggregates_ratios():
    from localsql.schema_context.coverage import GoldReferences

    gold_a = GoldReferences(tables=frozenset({"customers"}), columns=frozenset({"name"}))
    gold_b = GoldReferences(tables=frozenset({"orders"}), columns=frozenset({"order_date"}))
    results = [
        compute_coverage("ex-0", frozenset({"customers"}), gold_a),  # fully retained
        compute_coverage("ex-1", frozenset({"customers"}), gold_b),  # not retained
    ]
    summary = summarize_coverage(results)
    assert summary["total_examples"] == 2
    assert summary["parseable_examples"] == 2
    assert summary["table_recall"] == 0.5
    assert summary["pct_examples_retaining_all_gold_tables"] == 0.5
