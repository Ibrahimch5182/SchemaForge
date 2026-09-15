from localsql.model.generation import (
    build_chat_messages,
    build_model_inputs,
    count_input_tokens,
    generate_one_example,
    is_oom_error,
    normalize_predicted_sql,
    resolve_generation_example,
)

from tests.model.fixtures import (
    FakeBackend,
    FakeBatchEncoding,
    FakeGenModel,
    FakeGenTokenizer,
    FakeTensor,
    FakeTokenizer,
    make_example,
)


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


def test_count_input_tokens_batch_encoding_does_not_count_keys():
    """Regression: real `tokenizer.apply_chat_template(..., tokenize=True,
    add_generation_prompt=True, return_dict=True, return_tensors="pt")`
    returned a BatchEncoding with keys `input_ids`, `attention_mask`.
    `len(encoding)` on that gave 2 (the key count) instead of the real
    239-token sequence length -- must not reproduce that."""
    encoding = FakeBatchEncoding(
        input_ids=FakeTensor((1, 239)),
        attention_mask=FakeTensor((1, 239)),
    )
    assert len(encoding) == 2  # sanity: this IS the trap
    assert count_input_tokens(encoding) == 239


def test_count_input_tokens_batch_encoding_with_list_input_ids():
    # BatchEncoding without return_tensors -- input_ids is a batch-of-one
    # nested list, not a tensor.
    encoding = FakeBatchEncoding(input_ids=[[1, 2, 3, 4, 5]], attention_mask=[[1, 1, 1, 1, 1]])
    assert count_input_tokens(encoding) == 5


def test_count_input_tokens_raw_tensor():
    tensor = FakeTensor((1, 239))
    assert count_input_tokens(tensor) == 239


def test_count_input_tokens_plain_list():
    # tokenize=True, no return_tensors -- a flat list of token ids.
    assert count_input_tokens([1, 2, 3, 4, 5, 6, 7]) == 7


def test_qwen_backend_count_prompt_tokens_regression():
    """Reproduces the exact reported Kaggle bug at the real call site:
    `QwenBackend.count_prompt_tokens` must report the tensor's token
    dimension (239), not the BatchEncoding's key count (2). No torch/CUDA
    needed -- `count_prompt_tokens` has no ML imports of its own."""
    from pathlib import Path

    from localsql.model.config import load_model_config
    from localsql.model.qwen_backend import QwenBackend

    config = load_model_config(Path(__file__).resolve().parents[2] / "configs" / "model.yaml")
    backend = QwenBackend(config)

    class BatchEncodingTokenizer:
        def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True, **kwargs):
            return FakeBatchEncoding(
                input_ids=FakeTensor((1, 239)),
                attention_mask=FakeTensor((1, 239)),
            )

    backend._tokenizer = BatchEncodingTokenizer()
    assert backend.count_prompt_tokens("some canonical prompt") == 239


def test_generate_one_handles_batch_encoding_not_raw_tensor(monkeypatch):
    """Regression: the real Qwen tokenizer's `apply_chat_template(...,
    return_tensors="pt")` returns a `BatchEncoding` (dict-like), not a bare
    tensor. Treating it as a tensor directly (`encoded.shape[-1]`,
    `model.generate(encoded, ...)`) raised `AttributeError` on the first
    real Kaggle run. Exercises the actual `QwenBackend.generate_one()` code
    path (not a re-implementation) by injecting a fake `torch` module into
    `sys.modules`, so no real torch install is needed."""
    import contextlib
    import sys
    import types
    from pathlib import Path

    from localsql.model.config import load_model_config
    from localsql.model.qwen_backend import QwenBackend

    fake_torch = types.ModuleType("torch")

    @contextlib.contextmanager
    def _inference_mode():
        yield

    fake_torch.inference_mode = _inference_mode
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    config = load_model_config(Path(__file__).resolve().parents[2] / "configs" / "model.yaml")
    backend = QwenBackend(config)
    backend._tokenizer = FakeGenTokenizer(input_ids=[1, 2, 3, 4, 5])
    model = FakeGenModel(new_token_count=3)
    backend._model = model

    result = backend.generate_one("some canonical prompt")

    assert result.input_tokens == 5
    assert result.output_tokens == 3
    assert result.raw_completion == "SELECT 1"
    # The BatchEncoding must be unpacked by keyword into model.generate(),
    # not passed as a single positional tensor.
    assert model.last_generate_kwargs is not None
    assert "input_ids" in model.last_generate_kwargs
    assert "attention_mask" in model.last_generate_kwargs


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
