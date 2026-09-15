from pathlib import Path

from localsql.train.config import load_train_config, lora_summary, optimization_summary, quantization_summary

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "train.yaml"


def test_locked_base_model_and_quantization_match_phase3():
    config = load_train_config(CONFIG_PATH)
    assert config.model.id == "Qwen/Qwen3-4B-Instruct-2507"
    assert config.runtime.load_in_4bit is True
    assert config.runtime.quantization == "nf4"
    assert config.runtime.double_quant is True
    assert config.runtime.compute_dtype == "float16"


def test_starting_lora_configuration():
    config = load_train_config(CONFIG_PATH)
    assert config.lora.r == 16
    assert config.lora.alpha == 32
    assert config.lora.dropout == 0.05
    assert set(config.lora.target_modules) == {
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    }


def test_starting_optimization_configuration():
    config = load_train_config(CONFIG_PATH)
    assert config.optimization.per_device_train_batch_size == 1
    assert config.optimization.gradient_accumulation_steps == 8
    assert config.optimization.learning_rate == 1.0e-4
    assert config.optimization.warmup_ratio == 0.05
    assert config.optimization.gradient_checkpointing is True
    assert config.optimization.seed == 42


def test_completion_only_loss_is_enabled():
    config = load_train_config(CONFIG_PATH)
    assert config.loss.completion_only is True


def test_max_seq_length_is_provisional_but_set():
    config = load_train_config(CONFIG_PATH)
    assert config.sequence.max_seq_length == 4096


def test_summaries_are_plain_dicts():
    config = load_train_config(CONFIG_PATH)
    assert quantization_summary(config)["quant_type"] == "nf4"
    assert lora_summary(config)["r"] == 16
    assert optimization_summary(config)["seed"] == 42
