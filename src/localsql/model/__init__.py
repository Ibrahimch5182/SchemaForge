"""Phase 3 baseline model-inference package. No fine-tuning/training here.

`config.py` has no ML dependencies and is always importable. Everything
that touches torch/transformers/bitsandbytes does so via lazy imports
inside functions, so this package can be imported (and dry-run tested) on
a machine without the optional "model" dependency group installed.
"""
