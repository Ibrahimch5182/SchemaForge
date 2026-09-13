from localsql.data.prompt_builder import build_completion, build_prompt


def test_prompt_has_canonical_sections():
    prompt = build_prompt(
        schema_serialized="customers(\n  customer_id INTEGER PK\n)",
        dialect="sqlite",
        question="How many customers are there?",
        business_context="customers refers to the customers table",
    )
    assert "SYSTEM:" in prompt
    assert "DIALECT:" in prompt
    assert "SCHEMA:" in prompt
    assert "BUSINESS CONTEXT:" in prompt
    assert "QUESTION:" in prompt
    # Sections appear in the documented order.
    assert prompt.index("SYSTEM:") < prompt.index("DIALECT:") < prompt.index("SCHEMA:")
    assert prompt.index("SCHEMA:") < prompt.index("BUSINESS CONTEXT:") < prompt.index("QUESTION:")


def test_prompt_omits_business_context_when_absent():
    prompt = build_prompt(
        schema_serialized="customers(\n  customer_id INTEGER PK\n)",
        dialect="sqlite",
        question="How many customers are there?",
        business_context=None,
    )
    assert "BUSINESS CONTEXT:" not in prompt


def test_completion_is_sql_only_not_embedded_in_prompt():
    sql = "SELECT COUNT(*) FROM customers"
    prompt = build_prompt(
        schema_serialized="customers(\n  customer_id INTEGER PK\n)",
        dialect="sqlite",
        question="How many customers are there?",
        business_context=None,
    )
    completion = build_completion(sql)
    assert completion == sql
    # The gold SQL must not leak into the prompt itself.
    assert sql not in prompt


def test_completion_strips_whitespace_only():
    assert build_completion("  SELECT 1  \n") == "SELECT 1"
    # Internal spacing (e.g. inside string literals) must be preserved.
    assert build_completion("SELECT 'a  b'") == "SELECT 'a  b'"
