"""Shared argparse helpers for the Phase 7 llama.cpp inference CLIs."""

from __future__ import annotations

import argparse
from pathlib import Path

from localsql.deploy.config import Phase7Config
from localsql.deploy.runtime import LlamaSettings

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "phase7.yaml"


def add_llama_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--llama-exe", type=Path, required=True, help="llama.cpp completion/CLI executable (explicit path).")
    parser.add_argument("--model", type=Path, required=True, help="Base GGUF model file (F16 or Q4_K_M).")
    parser.add_argument("--lora", type=Path, default=None, help="LoRA GGUF loaded at runtime (--lora). Required unless --allow-no-lora.")
    parser.add_argument(
        "--allow-no-lora", action="store_true",
        help="Run a single GGUF without a LoRA (e.g. an optional merged GGUF). Not the primary mode.",
    )
    parser.add_argument("--manifest", type=Path, default=None, help="Gold-free generation manifest (default from config).")
    parser.add_argument("--context-size", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--n-gpu-layers", type=int, default=None, help="GPU offload (optional; 0 = CPU-only).")
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=None, help="Per-request subprocess timeout (seconds).")
    parser.add_argument("--extra-arg", action="append", default=[], help="Extra llama.cpp argument (repeatable).")
    parser.add_argument(
        "--omit-no-cnv", action="store_true", help="Do not pass -no-cnv (for tools that reject it)."
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the plan/command; run nothing.")


def settings_from_args(args: argparse.Namespace, cfg: Phase7Config) -> LlamaSettings:
    inf = cfg.inference
    return LlamaSettings(
        executable=args.llama_exe,
        model=args.model,
        context_size=args.context_size or inf.context_size,
        max_new_tokens=args.max_new_tokens or inf.max_new_tokens,
        seed=inf.seed,
        n_gpu_layers=inf.n_gpu_layers if args.n_gpu_layers is None else args.n_gpu_layers,
        threads=args.threads,
        timeout_seconds=args.timeout or inf.timeout_seconds,
        no_conversation_flag=not args.omit_no_cnv,
        extra_args=tuple(args.extra_arg),
        lora=args.lora,
    )


def require_lora_or_opt_out(args: argparse.Namespace) -> None:
    if args.lora is None and not args.allow_no_lora:
        raise SystemExit(
            "BLOCKER: --lora <LoRA GGUF> is required (primary mode is base GGUF + runtime LoRA). "
            "Pass --allow-no-lora only for an intentional single-GGUF run."
        )


def manifest_path(args: argparse.Namespace, cfg: Phase7Config) -> Path:
    return args.manifest or (REPO_ROOT / cfg.inference.prompt_manifest)
