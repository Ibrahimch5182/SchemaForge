from localsql.benchmark.models import GenerationExample
from localsql.benchmark.prediction import validate_predictions


def _manifest():
    return [
        GenerationExample(
            example_id="bird-mini-dev-sqlite-0000",
            db_id="shop_db",
            dialect="sqlite",
            question="How many customers?",
            serialized_schema="customers(\n  customer_id INTEGER PK\n)",
            prompt="SYSTEM:\n...\n\nQUESTION:\nHow many customers?",
            difficulty="simple",
        ),
        GenerationExample(
            example_id="bird-mini-dev-sqlite-0001",
            db_id="shop_db",
            dialect="sqlite",
            question="How many customers from France?",
            serialized_schema="customers(\n  customer_id INTEGER PK\n)",
            prompt="SYSTEM:\n...\n\nQUESTION:\nHow many customers from France?",
            difficulty="moderate",
        ),
    ]


def test_valid_predictions_pass():
    manifest = _manifest()
    records = [
        {"example_id": "bird-mini-dev-sqlite-0000", "db_id": "shop_db", "predicted_sql": "SELECT 1"},
        {"example_id": "bird-mini-dev-sqlite-0001", "db_id": "shop_db", "predicted_sql": "SELECT 2"},
    ]
    result = validate_predictions(records, manifest)
    assert result.is_complete
    assert len(result.valid_records) == 2


def test_detects_missing_predictions():
    manifest = _manifest()
    records = [{"example_id": "bird-mini-dev-sqlite-0000", "db_id": "shop_db", "predicted_sql": "SELECT 1"}]
    result = validate_predictions(records, manifest)
    assert result.missing_ids == ["bird-mini-dev-sqlite-0001"]
    assert not result.is_complete


def test_detects_duplicate_ids():
    manifest = _manifest()
    records = [
        {"example_id": "bird-mini-dev-sqlite-0000", "db_id": "shop_db", "predicted_sql": "SELECT 1"},
        {"example_id": "bird-mini-dev-sqlite-0000", "db_id": "shop_db", "predicted_sql": "SELECT 1 -- dup"},
    ]
    result = validate_predictions(records, manifest)
    assert result.duplicate_ids == ["bird-mini-dev-sqlite-0000"]


def test_detects_unknown_example_id():
    manifest = _manifest()
    records = [{"example_id": "does-not-exist", "db_id": "shop_db", "predicted_sql": "SELECT 1"}]
    result = validate_predictions(records, manifest)
    assert result.unknown_ids == ["does-not-exist"]


def test_detects_db_id_mismatch():
    manifest = _manifest()
    records = [{"example_id": "bird-mini-dev-sqlite-0000", "db_id": "wrong_db", "predicted_sql": "SELECT 1"}]
    result = validate_predictions(records, manifest)
    assert result.db_id_mismatches == ["bird-mini-dev-sqlite-0000"]


def test_detects_non_string_sql():
    manifest = _manifest()
    records = [{"example_id": "bird-mini-dev-sqlite-0000", "db_id": "shop_db", "predicted_sql": None}]
    result = validate_predictions(records, manifest)
    assert result.non_string_sql == ["bird-mini-dev-sqlite-0000"]
