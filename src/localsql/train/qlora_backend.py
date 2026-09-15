"""QLoRA 4-bit training backend for the Phase 4 smoke test.

torch/transformers/peft/bitsandbytes are imported lazily inside methods
(never at module level), matching `localsql.model.qwen_backend`'s pattern,
so this module stays importable without the optional "train" dependency
group. Only `load_for_training()`, `train_smoke()`, and
`load_adapter_for_verification()` need them, and only the first two
require CUDA. `load_tokenizer_only()` needs transformers + huggingface_hub
+ network only (no CUDA, no 4-bit model weights).

Uses plain `transformers.Trainer`, not `trl.SFTTrainer`: completion-only
masking is already fully computed by `localsql.train.sft_data` before the
trainer ever sees an example, so none of SFTTrainer's own dataset
formatting/masking behavior is needed or relied upon. TRL remains an
installed dependency (see `pyproject.toml`'s "train" group) for any later
phase that wants it.

One QLoRA backend only -- not a generic multi-model training framework.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from localsql.train.config import TrainConfig


class CudaNotAvailableError(RuntimeError):
    """Real QLoRA training/adapter verification requires CUDA. Use
    --dry-run/--token-profile for infra checks without a GPU, or run this
    on a Linux CUDA cloud/Kaggle instance."""


class NonFiniteLossError(RuntimeError):
    """Raised when a NaN/Inf training loss is observed -- the smoke run
    stops rather than continuing on a corrupted training state."""


@dataclass(frozen=True)
class LoadedTrainInfo:
    model_id: str
    resolved_revision: str
    tokenizer_revision: str
    quantization: dict
    lora: dict
    trainable_param_count: int
    total_param_count: int
    torch_version: str
    transformers_version: str
    peft_version: str
    bitsandbytes_version: str
    accelerate_version: str
    trl_version: str
    cuda_version: Optional[str]
    gpu_name: Optional[str]
    gpu_total_memory_mb: Optional[float]


class QLoraBackend:
    def __init__(self, config: TrainConfig):
        self.config = config
        self._model = None
        self._tokenizer = None
        self.info: Optional[LoadedTrainInfo] = None

    def _resolve_revision(self) -> str:
        from huggingface_hub import HfApi

        cfg = self.config
        return HfApi().model_info(cfg.model.id, revision=cfg.model.revision).sha

    def load_tokenizer_only(self) -> str:
        """Lightweight load for --token-profile mode -- no model weights,
        no bitsandbytes, no CUDA requirement. Returns the resolved revision."""
        from transformers import AutoTokenizer

        resolved_revision = self._resolve_revision()
        self._tokenizer = AutoTokenizer.from_pretrained(self.config.model.id, revision=resolved_revision)
        return resolved_revision

    def _build_bnb_config(self, torch):
        from transformers import BitsAndBytesConfig

        cfg = self.config
        return BitsAndBytesConfig(
            load_in_4bit=cfg.runtime.load_in_4bit,
            bnb_4bit_quant_type=cfg.runtime.quantization,
            bnb_4bit_use_double_quant=cfg.runtime.double_quant,
            bnb_4bit_compute_dtype=getattr(torch, cfg.runtime.compute_dtype),
        )

    def load_for_training(self) -> LoadedTrainInfo:
        import accelerate
        import bitsandbytes
        import peft
        import torch
        import transformers
        import trl
        from peft import LoraConfig as PeftLoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available():
            raise CudaNotAvailableError(
                "No usable CUDA device found. Real QLoRA smoke training must run on a "
                "Linux CUDA cloud/Kaggle GPU. Use --dry-run/--token-profile to validate "
                "infrastructure without a model/GPU on this machine."
            )

        cfg = self.config
        resolved_revision = self._resolve_revision()
        tokenizer = AutoTokenizer.from_pretrained(cfg.model.id, revision=resolved_revision)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            cfg.model.id,
            revision=resolved_revision,
            quantization_config=self._build_bnb_config(torch),
            device_map=cfg.runtime.device,
        )
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=cfg.optimization.gradient_checkpointing
        )
        lora_config = PeftLoraConfig(
            r=cfg.lora.r,
            lora_alpha=cfg.lora.alpha,
            lora_dropout=cfg.lora.dropout,
            target_modules=list(cfg.lora.target_modules),
            task_type=cfg.lora.task_type,
        )
        model = get_peft_model(model, lora_config)

        trainable, total = 0, 0
        for p in model.parameters():
            total += p.numel()
            if p.requires_grad:
                trainable += p.numel()

        torch.cuda.reset_peak_memory_stats()

        self._model = model
        self._tokenizer = tokenizer
        self.info = LoadedTrainInfo(
            model_id=cfg.model.id,
            resolved_revision=resolved_revision,
            tokenizer_revision=resolved_revision,
            quantization={
                "load_in_4bit": cfg.runtime.load_in_4bit,
                "quant_type": cfg.runtime.quantization,
                "double_quant": cfg.runtime.double_quant,
                "compute_dtype": cfg.runtime.compute_dtype,
            },
            lora={
                "r": cfg.lora.r,
                "alpha": cfg.lora.alpha,
                "dropout": cfg.lora.dropout,
                "target_modules": list(cfg.lora.target_modules),
            },
            trainable_param_count=trainable,
            total_param_count=total,
            torch_version=torch.__version__,
            transformers_version=transformers.__version__,
            peft_version=peft.__version__,
            bitsandbytes_version=bitsandbytes.__version__,
            accelerate_version=accelerate.__version__,
            trl_version=trl.__version__,
            cuda_version=torch.version.cuda,
            gpu_name=torch.cuda.get_device_name(0),
            gpu_total_memory_mb=round(torch.cuda.get_device_properties(0).total_memory / (1024**2), 1),
        )
        return self.info

    def train_smoke(
        self,
        encodings: list,
        output_dir: Path,
        max_steps: int,
        save_steps: int | None = None,
        save_total_limit: int | None = None,
        resume_from_checkpoint: str | None = None,
    ) -> dict:
        """Bounded-step training over pre-masked `SftEncoding` examples --
        used for the Phase 4 smoke test, and (with checkpointing enabled)
        Phase 5B's memory/throughput certification and stop/resume runs.

        Checkpointing: `save_steps` enables HF `Trainer`'s own periodic
        checkpoint saves under `output_dir/checkpoint-<step>` (full
        resumable state -- model/adapter, optimizer, LR scheduler, RNG,
        `trainer_state.json` with `global_step` -- NOT just the final
        LoRA-only adapter export from `save_adapter()`). `save_total_limit`
        caps how many checkpoints are kept. `resume_from_checkpoint`, if
        given, is passed straight to `Trainer.train()` -- this is the
        explicit resume source; nothing here ever auto-discovers or
        auto-resumes from an arbitrary directory.

        Raises `NonFiniteLossError` on NaN/Inf loss; propagates CUDA OOM
        (`torch.cuda.OutOfMemoryError` / RuntimeError) unmodified -- both
        stop the run rather than continuing on a corrupted state.
        """
        import torch
        from transformers import Trainer, TrainerCallback, TrainingArguments

        cfg = self.config
        pad_id = self._tokenizer.pad_token_id

        class _ListDataset(torch.utils.data.Dataset):
            def __init__(self, items):
                self.items = items

            def __len__(self):
                return len(self.items)

            def __getitem__(self, idx):
                e = self.items[idx]
                return {
                    "input_ids": torch.tensor(e.input_ids, dtype=torch.long),
                    "attention_mask": torch.tensor([1] * len(e.input_ids), dtype=torch.long),
                    "labels": torch.tensor(e.labels, dtype=torch.long),
                }

        def _collate(batch):
            max_len = max(len(b["input_ids"]) for b in batch)
            input_ids, attention_mask, labels = [], [], []
            for b in batch:
                pad_len = max_len - len(b["input_ids"])
                input_ids.append(torch.cat([b["input_ids"], torch.full((pad_len,), pad_id, dtype=torch.long)]))
                attention_mask.append(torch.cat([b["attention_mask"], torch.zeros(pad_len, dtype=torch.long)]))
                labels.append(torch.cat([b["labels"], torch.full((pad_len,), -100, dtype=torch.long)]))
            return {
                "input_ids": torch.stack(input_ids),
                "attention_mask": torch.stack(attention_mask),
                "labels": torch.stack(labels),
            }

        metrics_log: list[dict] = []

        class _LogCallback(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                if logs:
                    metrics_log.append(dict(logs))

        args = TrainingArguments(
            output_dir=str(output_dir),
            per_device_train_batch_size=cfg.optimization.per_device_train_batch_size,
            gradient_accumulation_steps=cfg.optimization.gradient_accumulation_steps,
            learning_rate=cfg.optimization.learning_rate,
            warmup_ratio=cfg.optimization.warmup_ratio,
            max_steps=max_steps,
            gradient_checkpointing=cfg.optimization.gradient_checkpointing,
            optim=cfg.optimization.optim,
            seed=cfg.optimization.seed,
            logging_steps=1,
            save_strategy="steps" if save_steps else "no",
            save_steps=save_steps or 500,  # ignored when save_strategy="no"
            save_total_limit=save_total_limit,
            report_to=[],
        )

        trainer = Trainer(
            model=self._model,
            args=args,
            train_dataset=_ListDataset(encodings),
            data_collator=_collate,
            callbacks=[_LogCallback()],
        )

        starting_global_step = 0
        if resume_from_checkpoint:
            state_path = Path(resume_from_checkpoint) / "trainer_state.json"
            if state_path.exists():
                starting_global_step = json.loads(state_path.read_text(encoding="utf-8")).get("global_step", 0)

        start = time.perf_counter()
        train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        runtime_s = time.perf_counter() - start

        for entry in metrics_log:
            loss = entry.get("loss")
            if loss is not None and (loss != loss or loss in (float("inf"), float("-inf"))):
                raise NonFiniteLossError(f"Non-finite training loss detected: {entry}")

        return {
            "runtime_seconds": round(runtime_s, 2),
            "global_step": trainer.state.global_step,
            "starting_global_step": starting_global_step,
            "resumed_from_checkpoint": resume_from_checkpoint,
            "metrics_log": metrics_log,
            "peak_gpu_memory_allocated_mb": round(torch.cuda.max_memory_allocated() / (1024**2), 1),
            "peak_gpu_memory_reserved_mb": round(torch.cuda.max_memory_reserved() / (1024**2), 1),
            "train_result_metrics": dict(train_result.metrics) if train_result else {},
        }

    def save_adapter(self, path: Path) -> None:
        self._model.save_pretrained(str(path))
        self._tokenizer.save_pretrained(str(path))

    def load_adapter_for_verification(self, adapter_path: Path) -> dict:
        """Reload the base model + saved LoRA adapter for the Phase 4
        sanity check. NOT an accuracy evaluation."""
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available():
            raise CudaNotAvailableError(
                "No usable CUDA device found for adapter verification. Run on the same "
                "Linux CUDA cloud/Kaggle instance used for training."
            )

        cfg = self.config
        resolved_revision = self._resolve_revision()
        tokenizer = AutoTokenizer.from_pretrained(str(adapter_path))
        base_model = AutoModelForCausalLM.from_pretrained(
            cfg.model.id,
            revision=resolved_revision,
            quantization_config=self._build_bnb_config(torch),
            device_map=cfg.runtime.device,
        )
        model = PeftModel.from_pretrained(base_model, str(adapter_path))
        model.eval()

        adapter_config = getattr(model, "peft_config", None) or {}

        self._model = model
        self._tokenizer = tokenizer
        return {
            "resolved_revision": resolved_revision,
            "adapter_active": bool(adapter_config),
            "adapter_names": list(adapter_config.keys()),
        }

    def generate_one(self, prompt: str, max_new_tokens: int = 128) -> str:
        """Tiny sanity-check generation -- NOT an accuracy evaluation.
        Reuses the same BatchEncoding-safe envelope construction fixed for
        the Phase 3 baseline backend (`build_model_inputs`), so the same
        `AttributeError` regression found there cannot recur here.
        """
        import torch

        from localsql.model.generation import build_model_inputs

        device = next(self._model.parameters()).device
        encoded = build_model_inputs(self._tokenizer, prompt, return_tensors="pt").to(device)
        input_ids = encoded["input_ids"]
        input_len = input_ids.shape[-1]

        with torch.inference_mode():
            output_ids = self._model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        new_tokens = output_ids[0][input_len:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
