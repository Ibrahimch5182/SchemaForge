"""Phase 4 QLoRA smoke-test training package. NOT full-experiment training.

`config.py` and `sft_data.py` have no ML dependencies and are always
importable. `qlora_backend.py` lazily imports torch/transformers/peft/
bitsandbytes/trl inside methods, so this package stays importable (and
CPU/offline-testable) without the optional "train" dependency group.
"""
