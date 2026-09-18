"""Tests for optional PEFT adapter loading in `QwenBackend.load()`.

No real torch/transformers/bitsandbytes/peft install needed -- a minimal
fake stack is injected into `sys.modules` (same technique already used in
`tests/model/test_generation.py` for `generate_one()`), so these exercise
the actual `QwenBackend.load()` code path, not a re-implementation.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from localsql.model.config import load_model_config
from localsql.model.qwen_backend import AdapterValidationError, QwenBackend, validate_adapter_path

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "model.yaml"


def _make_adapter_dir(tmp_path: Path, complete: bool = True) -> Path:
    adapter_dir = tmp_path / "checkpoint-1518"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    if complete:
        (adapter_dir / "adapter_model.safetensors").write_bytes(b"fake-weights")
    return adapter_dir


class _FakeCuda:
    def __init__(self):
        self.reset_peak_memory_stats_calls = 0

    def is_available(self):
        return True

    def reset_peak_memory_stats(self):
        self.reset_peak_memory_stats_calls += 1

    def get_device_name(self, idx):
        return "Fake GPU"

    def get_device_properties(self, idx):
        return types.SimpleNamespace(total_memory=16 * 1024**3)

    def max_memory_allocated(self):
        return 0


class _FakeBaseModel:
    def __init__(self):
        self.eval_calls = 0

    def eval(self):
        self.eval_calls += 1


class _FakeAutoModelForCausalLM:
    calls: list[dict] = []

    @classmethod
    def from_pretrained(cls, model_id, revision=None, quantization_config=None, device_map=None):
        cls.calls.append(
            dict(model_id=model_id, revision=revision, quantization_config=quantization_config, device_map=device_map)
        )
        return _FakeBaseModel()


class _FakeAutoTokenizer:
    @classmethod
    def from_pretrained(cls, model_id, revision=None):
        return object()


class _FakeBitsAndBytesConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeHfApi:
    def model_info(self, model_id, revision=None):
        return types.SimpleNamespace(sha="resolved-sha-fake")


class _FakePeftModel:
    instances: list["_FakePeftModel"] = []

    def __init__(self, base_model, adapter_path, is_trainable, peft_config):
        self.base_model = base_model
        self.adapter_path = adapter_path
        self.is_trainable = is_trainable
        self.peft_config = peft_config
        self.eval_calls = 0
        self.merge_and_unload_calls = 0

    def eval(self):
        self.eval_calls += 1

    def merge_and_unload(self):  # pragma: no cover - must never be called
        self.merge_and_unload_calls += 1

    @classmethod
    def from_pretrained(cls, base_model, adapter_path, is_trainable=None):
        instance = cls(base_model, adapter_path, is_trainable, peft_config={"default": object()})
        cls.instances.append(instance)
        return instance


@pytest.fixture
def fake_ml_stack(monkeypatch):
    """Installs fake torch/transformers/bitsandbytes/huggingface_hub modules.
    Does NOT install a `peft` module -- tests that need adapter loading add
    it explicitly, proving the adapter_path=None path never imports peft.
    """
    _FakeAutoModelForCausalLM.calls = []
    _FakePeftModel.instances = []

    fake_torch = types.ModuleType("torch")
    fake_torch.__version__ = "2.4.0-fake"
    fake_torch.float16 = "float16-dtype"
    fake_torch.cuda = _FakeCuda()
    fake_torch.version = types.SimpleNamespace(cuda="12.1-fake")
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.__version__ = "4.46.0-fake"
    fake_transformers.AutoModelForCausalLM = _FakeAutoModelForCausalLM
    fake_transformers.AutoTokenizer = _FakeAutoTokenizer
    fake_transformers.BitsAndBytesConfig = _FakeBitsAndBytesConfig
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    fake_bnb = types.ModuleType("bitsandbytes")
    fake_bnb.__version__ = "0.44.0-fake"
    monkeypatch.setitem(sys.modules, "bitsandbytes", fake_bnb)

    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.HfApi = _FakeHfApi
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    # Guarantee that touching `peft` at all fails loudly and immediately,
    # so the adapter_path=None assertions below are a real regression test.
    monkeypatch.setitem(sys.modules, "peft", None)

    return fake_torch


@pytest.fixture
def fake_peft(monkeypatch):
    fake_peft_mod = types.ModuleType("peft")
    fake_peft_mod.PeftModel = _FakePeftModel
    monkeypatch.setitem(sys.modules, "peft", fake_peft_mod)
    return fake_peft_mod


def test_load_without_adapter_matches_baseline_behavior(fake_ml_stack):
    """adapter_path=None must be semantically identical to the pre-adapter
    Phase 3 baseline path: same base-model kwargs, no peft import, and
    adapter_active=False in the returned info."""
    backend = QwenBackend(load_model_config(CONFIG_PATH))
    info = backend.load()

    assert info.adapter_path is None
    assert info.adapter_active is False
    assert len(_FakeAutoModelForCausalLM.calls) == 1
    call = _FakeAutoModelForCausalLM.calls[0]
    assert call["model_id"] == "Qwen/Qwen3-4B-Instruct-2507"
    assert call["device_map"] == "cuda"
    assert backend._model.eval_calls == 1
    assert fake_ml_stack.cuda.reset_peak_memory_stats_calls == 1


def test_load_with_adapter_wraps_base_model_and_activates(fake_ml_stack, fake_peft, tmp_path):
    adapter_dir = _make_adapter_dir(tmp_path)
    backend = QwenBackend(load_model_config(CONFIG_PATH))
    info = backend.load(adapter_path=adapter_dir)

    assert info.adapter_active is True
    assert info.adapter_path == str(adapter_dir)
    # Same base-model load call as the no-adapter path.
    base_call = _FakeAutoModelForCausalLM.calls[0]
    assert base_call["model_id"] == "Qwen/Qwen3-4B-Instruct-2507"

    peft_instance = _FakePeftModel.instances[-1]
    assert peft_instance.is_trainable is False
    assert peft_instance.merge_and_unload_calls == 0  # never merged
    assert isinstance(peft_instance.base_model, _FakeBaseModel)
    assert backend._model is peft_instance


def test_load_missing_adapter_directory_fails_before_model_load(fake_ml_stack, tmp_path):
    missing = tmp_path / "does-not-exist"
    backend = QwenBackend(load_model_config(CONFIG_PATH))

    with pytest.raises(AdapterValidationError, match="not found"):
        backend.load(adapter_path=missing)

    assert _FakeAutoModelForCausalLM.calls == []  # never got as far as loading a model


def test_load_incomplete_adapter_directory_fails_before_model_load(fake_ml_stack, tmp_path):
    incomplete = _make_adapter_dir(tmp_path, complete=False)
    backend = QwenBackend(load_model_config(CONFIG_PATH))

    with pytest.raises(AdapterValidationError, match="adapter_model.safetensors"):
        backend.load(adapter_path=incomplete)

    assert _FakeAutoModelForCausalLM.calls == []


def test_validate_adapter_path_missing_directory(tmp_path):
    with pytest.raises(AdapterValidationError, match="not found"):
        validate_adapter_path(tmp_path / "nope")


def test_validate_adapter_path_missing_config_file(tmp_path):
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"x")
    with pytest.raises(AdapterValidationError, match="adapter_config.json"):
        validate_adapter_path(adapter_dir)


def test_validate_adapter_path_complete_directory_ok(tmp_path):
    adapter_dir = _make_adapter_dir(tmp_path)
    validate_adapter_path(adapter_dir)  # must not raise
