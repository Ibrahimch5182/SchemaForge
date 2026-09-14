from pathlib import Path

from localsql.model.config import generation_summary, load_model_config, quantization_summary

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "model.yaml"


def test_locked_baseline_model_and_quantization():
    config = load_model_config(CONFIG_PATH)
    assert config.model.id == "Qwen/Qwen3-4B-Instruct-2507"
    assert config.runtime.load_in_4bit is True
    assert config.runtime.quantization == "nf4"
    assert config.runtime.double_quant is True
    assert config.runtime.compute_dtype == "float16"


def test_deterministic_generation_settings():
    config = load_model_config(CONFIG_PATH)
    assert config.generation.do_sample is False
    assert config.generation.seed == 42


def test_context_mode_default_is_with_business_context():
    config = load_model_config(CONFIG_PATH)
    assert config.prompt.context_mode == "with_business_context"


def test_quantization_and_generation_summaries_are_plain_dicts():
    config = load_model_config(CONFIG_PATH)
    q = quantization_summary(config)
    g = generation_summary(config)
    assert q == {
        "load_in_4bit": True,
        "quant_type": "nf4",
        "double_quant": True,
        "compute_dtype": "float16",
    }
    assert g == {"do_sample": False, "max_new_tokens": 512, "seed": 42}
