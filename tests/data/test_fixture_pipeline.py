"""Integration-style tests against a tiny committed real-BIRD fixture.

These use `tests/data/sample_*` files only -- no network access, no full
dataset download. They exercise the real parsing path end-to-end.
"""

from pathlib import Path

from localsql.data.bird_loader import load_column_meaning, load_raw_examples
from localsql.data.prompt_builder import build_completion, build_prompt
from localsql.data.schema_loader import parse_database_schemas
from localsql.data.schema_serializer import serialize_schema
from localsql.data.validator import validate_raw_example

FIXTURE_DIR = Path(__file__).parent


def test_load_sample_raw_rows():
    examples = load_raw_examples(FIXTURE_DIR / "sample_bird_rows.jsonl")
    assert len(examples) == 6
    assert all(e.question.strip() for e in examples)
    assert all(e.sql.strip() for e in examples)


def test_parse_sample_schemas_with_descriptions():
    meaning = load_column_meaning(FIXTURE_DIR / "sample_column_meaning.json")
    schemas = parse_database_schemas(FIXTURE_DIR / "sample_train_tables.json", meaning)
    assert set(schemas) == {"book_publishing_company", "movie_platform", "retail_complains"}

    movie_schema = schemas["movie_platform"]
    assert any(t.name == "movies" for t in movie_schema.tables)
    movies_table = next(t for t in movie_schema.tables if t.name == "movies")
    pk_columns = [c for c in movies_table.columns if c.is_primary_key]
    assert len(pk_columns) >= 1
    described = [c for c in movies_table.columns if c.description]
    assert len(described) > 0


def test_full_example_construction_from_fixtures():
    examples = load_raw_examples(FIXTURE_DIR / "sample_bird_rows.jsonl")
    meaning = load_column_meaning(FIXTURE_DIR / "sample_column_meaning.json")
    schemas = parse_database_schemas(FIXTURE_DIR / "sample_train_tables.json", meaning)

    for ex in examples:
        assert not validate_raw_example(ex, schemas)
        schema_text = serialize_schema(schemas[ex.db_id])
        assert schema_text
        prompt = build_prompt(schema_text, "sqlite", ex.question, ex.evidence)
        completion = build_completion(ex.sql)
        assert "QUESTION:" in prompt
        assert completion == ex.sql.strip()
