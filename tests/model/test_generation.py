from localsql.model.generation import (
    build_chat_messages,
    build_model_inputs,
    generate_one_example,
    is_oom_error,
    normalize_predicted_sql,
    resolve_generation_example,
)

from tests.model.fixtures import FakeBackend, FakeTokenizer, make_example


def test_chat_envelope_wraps_canonical_prompt_as_single_user_message():
    prompt = "SYSTEM:\n...\n\nQUESTION:\nHow many customers?"
    messages = build_chat_messages(prompt)
    assert messages == [{"role": "user", "content": prompt}]


def test_build_model_inputs_uses_tokenizer_native_chat_template():
    tokenizer = FakeTokenizer()
    prompt = "SYSTEM:\n...\n\nQUESTION:\nHow many customers?"
    build_model_inputs(tokenizer, prompt, return_tensors="pt")

    assert tokenizer.last_messages == [{"role": "user", "content": prompt}]
    assert tokenizer.last_kwargs["tokenize"] is True
    assert tokenizer.last_kwargs["add_generation_prompt"] is True
    assert tokenizer.last_kwargs["return_tensors"] == "pt"


def test_normalize_predicted_sql_only_strips_whitespace():
    assert normalize_predicted_sql("  SELECT 1  \n") == "SELECT 1"
    # Must NOT repair/strip markdown fences, prefixes, or prose.
    messy = "```sql\nSELECT 1\n```"
    assert normalize_predicted_sql(messy) == messy
    prefixed = "SQL: SELECT 1"
    assert normalize_predicted_sql(prefixed) == prefixed


def test_resolve_generation_example_with_business_context_is_unchanged():
    ex = make_example(0)
    resolved = resolve_generation_example(ex, "with_business_context")
    assert resolved is ex


def test_resolve_generation_example_without_business_context_rebuilds_via_canonical_builder():
    ex = make_example(0)  # has business_context set
    resolved = resolve_generation_example(ex, "without_business_context")
    assert resolved.business_context is None
    assert "BUSINESS CONTEXT:" not in resolved.prompt
    assert "customers(" in resolved.prompt  # schema still present, canonical builder reused


def test_generate_one_example_ok():
    ex = make_example(0)
    backend = FakeBackend(raw_completion="SELECT COUNT(*) FROM customers")
    record = generate_one_example(ex, backend, model_revision="abc123", context_mode="with_business_context")
    assert record.status == "ok"
    assert record.predicted_sql == "SELECT COUNT(*) FROM customers"
    assert record.raw_completion == "SELECT COUNT(*) FROM customers"
    assert record.is_oom is False
    assert backend.calls == [ex.prompt]


def test_generate_one_example_records_error_without_raising():
    ex = make_example(0)
    backend = FakeBackend(raise_exc=RuntimeError("model exploded"))
    record = generate_one_example(ex, backend, model_revision="abc123", context_mode="with_business_context")
    assert record.status == "error"
    assert "model exploded" in record.error
    assert record.predicted_sql is None
    assert record.is_oom is False


def test_generate_one_example_flags_oom():
    ex = make_example(0)
    backend = FakeBackend(raise_exc=RuntimeError("CUDA out of memory."))
    record = generate_one_example(ex, backend, model_revision="abc123", context_mode="with_business_context")
    assert record.status == "error"
    assert record.is_oom is True


def test_is_oom_error_detection():
    assert is_oom_error(RuntimeError("CUDA out of memory. Tried to allocate ..."))
    assert not is_oom_error(RuntimeError("some other failure"))


def test_prediction_dict_matches_phase2_contract_shape():
    ex = make_example(0)
    backend = FakeBackend()
    record = generate_one_example(ex, backend, model_revision="abc123", context_mode="with_business_context")
    pred = record.to_prediction_dict()
    assert pred["example_id"] == ex.example_id
    assert pred["db_id"] == ex.db_id
    assert isinstance(pred["predicted_sql"], str)
    # No gold field of any kind -- only "predicted_sql" is allowed.
    assert not {"sql", "gold_sql", "target", "target_sql"} & set(pred.keys())
