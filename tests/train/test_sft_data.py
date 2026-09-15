from pathlib import Path

from localsql.train.sft_data import (
    LABEL_IGNORE_INDEX,
    build_sft_encoding,
    load_prepared_examples,
    pad_sft_encoding,
)

from tests.train.fixtures import FakeSftTokenizer, make_prepared_example


def test_prompt_tokens_are_masked():
    tokenizer = FakeSftTokenizer()
    encoding = build_sft_encoding(tokenizer, "ex-0", "some schema and question", "SELECT 1")
    prompt_labels = encoding.labels[: encoding.prompt_token_count]
    assert prompt_labels and all(label == LABEL_IGNORE_INDEX for label in prompt_labels)


def test_completion_tokens_remain_trainable():
    tokenizer = FakeSftTokenizer()
    encoding = build_sft_encoding(tokenizer, "ex-0", "some schema and question", "SELECT COUNT FROM customers")
    completion_labels = encoding.labels[encoding.prompt_token_count :]
    assert completion_labels
    assert all(label != LABEL_IGNORE_INDEX for label in completion_labels)
    # Trainable labels must equal the actual completion token ids, not a
    # placeholder -- this is what "trainable" means for a causal LM.
    assert completion_labels == encoding.input_ids[encoding.prompt_token_count :]


def test_padding_tokens_are_masked():
    tokenizer = FakeSftTokenizer()
    encoding = build_sft_encoding(tokenizer, "ex-0", "hello world", "SELECT 1")
    padded = pad_sft_encoding(encoding, max_seq_length=encoding.total_token_count + 5, pad_token_id=tokenizer.pad_token_id)

    pad_region_labels = padded.labels[encoding.total_token_count :]
    pad_region_ids = padded.input_ids[encoding.total_token_count :]
    assert len(pad_region_labels) == 5
    assert all(label == LABEL_IGNORE_INDEX for label in pad_region_labels)
    assert all(tok == tokenizer.pad_token_id for tok in pad_region_ids)


def test_padding_is_noop_when_already_at_or_over_length():
    tokenizer = FakeSftTokenizer()
    encoding = build_sft_encoding(tokenizer, "ex-0", "hello world", "SELECT 1")
    padded = pad_sft_encoding(encoding, max_seq_length=encoding.total_token_count, pad_token_id=0)
    assert padded == encoding


def test_at_least_one_trainable_target_token():
    tokenizer = FakeSftTokenizer()
    encoding = build_sft_encoding(tokenizer, "ex-0", "some schema and question", "SELECT 1")
    assert any(label != LABEL_IGNORE_INDEX for label in encoding.labels)


def test_no_gold_sql_leaks_into_prompt_region():
    tokenizer = FakeSftTokenizer()
    prompt = "schema and question about customers"
    completion = "SELECT secretcolumn FROM secrettable"
    encoding = build_sft_encoding(tokenizer, "ex-0", prompt, completion)

    prompt_region_ids = set(encoding.input_ids[: encoding.prompt_token_count])
    completion_only_ids = {tokenizer.word_id("secretcolumn"), tokenizer.word_id("secrettable")}
    assert not (completion_only_ids & prompt_region_ids)


def test_prompt_only_encoding_matches_prefix_of_full_encoding():
    """The prompt/completion boundary must be a real prefix match -- proof
    the two separate tokenizer calls (prompt-only vs. full) agree on where
    the prompt ends, not just an assumed/miscounted split point."""
    tokenizer = FakeSftTokenizer()
    prompt = "schema and question"
    encoding = build_sft_encoding(tokenizer, "ex-0", prompt, "SELECT 1")
    standalone_prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True
    )
    assert encoding.input_ids[: encoding.prompt_token_count] == standalone_prompt_ids


def test_exceeds_max_seq_length_flag():
    tokenizer = FakeSftTokenizer()
    encoding = build_sft_encoding(tokenizer, "ex-0", "a b c d e f g h", "SELECT 1", max_seq_length=3)
    assert encoding.exceeds_max_seq_length is True

    encoding_ok = build_sft_encoding(tokenizer, "ex-1", "a b c d e f g h", "SELECT 1", max_seq_length=1000)
    assert encoding_ok.exceeds_max_seq_length is False


def test_load_prepared_examples_reuses_phase1_prompt_and_completion_verbatim(tmp_path):
    examples = [make_prepared_example(0), make_prepared_example(1, business_context=None)]
    path = tmp_path / "train.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(ex.model_dump_json() + "\n")

    loaded = load_prepared_examples(path)
    assert len(loaded) == 2
    assert loaded[0].prompt == examples[0].prompt
    assert loaded[0].completion == examples[0].completion
    # Evidence-dropout decision (business_context) preserved verbatim, not re-derived.
    assert loaded[0].business_context == examples[0].business_context
    assert loaded[1].business_context is None
