"""Phase 7: quantization + local inference engineering (llama.cpp / GGUF).

Pure orchestration and bookkeeping. No torch/transformers import, no
vendored llama.cpp: external tools are supplied by path and invoked via
subprocess (never via a shell).
"""
