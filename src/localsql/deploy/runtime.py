"""Local llama.cpp inference over the canonical SchemaForge prompt.

The prompt content is `GenerationExample.prompt` verbatim (the Phase 1/2
canonical prompt) -- no second Text-to-SQL prompt is introduced. The only
model-specific step is the Qwen ChatML envelope, which llama.cpp's plain
completion tools do not apply on their own. Completions go through the same
whitespace-only `normalize_predicted_sql` as Phases 3-6.
"""

from __future__ import annotations

import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from localsql.deploy.provenance import artifact_record
from localsql.deploy.process import ProcessResult, format_command, run_captured
from localsql.model.generation import normalize_predicted_sql

# llama.cpp prints this literal marker on some builds when generation ends
# on EOS; it is tool chatter, not model output.
END_OF_TEXT_MARKER = "[end of text]"

_PROMPT_EVAL = re.compile(
    r"prompt eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*(?:tokens|runs)\s*\(\s*[\d.]+\s*ms per token,\s*([\d.]+)\s*tokens per second\)"
)
_EVAL = re.compile(
    r"(?<!prompt )eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*(?:tokens|runs)\s*\(\s*[\d.]+\s*ms per token,\s*([\d.]+)\s*tokens per second\)"
)


def build_chatml_prompt(canonical_prompt: str) -> str:
    """Qwen ChatML envelope for ONE user message + generation prompt.

    Equivalent to the tokenizer's chat template for Qwen3-4B-Instruct-2507
    with a single user turn and `add_generation_prompt=True` (the same
    envelope as `localsql.model.generation.build_model_inputs`).
    """
    return f"<|im_start|>user\n{canonical_prompt}<|im_end|>\n<|im_start|>assistant\n"


@dataclass(frozen=True)
class LlamaSettings:
    executable: Path
    model: Path
    context_size: int
    max_new_tokens: int
    seed: int
    n_gpu_layers: int = 0
    threads: Optional[int] = None
    timeout_seconds: Optional[float] = 900
    no_conversation_flag: bool = True
    extra_args: tuple[str, ...] = ()
    lora: Optional[Path] = None  # LoRA GGUF loaded at runtime (primary deployment mode)
    # llama.cpp's `-f` strips ONE trailing newline from the prompt file, so the ChatML
    # envelope's final `assistant<newline>` reaches the model as `assistant` (measured on b10964:
    # 9 vs 10 tokens). That train/serve drift makes the model emit a leading newline/space/`:`.
    # True writes one extra newline so exactly `assistant<newline>` is tokenized. Default False keeps
    # the frozen Phase 7 benchmark behavior reproducible; the Phase 8 backend enables it.
    guard_prompt_trailing_newline: bool = False


def build_llama_command(settings: LlamaSettings, prompt_file: Path) -> list[str]:
    """Greedy (temp 0, top-k 1), CPU-valid command. The prompt goes through a
    file (-f) so a multi-thousand-token prompt never hits the Windows
    command-line length limit or shell-quoting issues."""
    cmd = [
        str(settings.executable),
        "-m", str(settings.model),
        "-f", str(prompt_file),
        "-n", str(settings.max_new_tokens),
        "-c", str(settings.context_size),
        "--temp", "0",
        "--top-k", "1",
        "--seed", str(settings.seed),
        "-ngl", str(settings.n_gpu_layers),
        "--no-display-prompt",
    ]  # fmt: skip
    if settings.lora is not None:
        cmd += ["--lora", str(settings.lora)]
    if settings.no_conversation_flag:
        cmd.append("-no-cnv")
    if settings.threads is not None:
        cmd += ["-t", str(settings.threads)]
    cmd += list(settings.extra_args)
    return cmd


def parse_perf(stderr: str) -> dict:
    """Extract llama.cpp's own timing lines. Anything not exposed by this
    build is reported as None (never estimated)."""
    perf: dict = {
        "prompt_eval_ms": None,
        "prompt_tokens": None,
        "prompt_tokens_per_second": None,
        "eval_ms": None,
        "generated_tokens": None,
        "generated_tokens_per_second": None,
    }
    m = _PROMPT_EVAL.search(stderr)
    if m:
        perf.update(prompt_eval_ms=float(m[1]), prompt_tokens=int(m[2]), prompt_tokens_per_second=float(m[3]))
    m = _EVAL.search(stderr)
    if m:
        perf.update(eval_ms=float(m[1]), generated_tokens=int(m[2]), generated_tokens_per_second=float(m[3]))
    perf["availability"] = (
        "llama.cpp timing lines parsed"
        if perf["prompt_eval_ms"] is not None and perf["eval_ms"] is not None
        else "not_available: this llama.cpp build did not print parseable timing lines"
    )
    return perf


def clean_completion(stdout: str) -> tuple[str, bool]:
    """Strip one trailing llama.cpp end-of-text marker (tool chatter).
    Returns (raw_completion, marker_stripped). Nothing else is altered."""
    text = stdout
    stripped_trailing = text.rstrip()
    if stripped_trailing.endswith(END_OF_TEXT_MARKER):
        return stripped_trailing[: -len(END_OF_TEXT_MARKER)], True
    return text, False


@dataclass(frozen=True)
class GgufRun:
    status: str  # "ok" | "error" | "timeout" | "cancelled"
    raw_completion: Optional[str]
    predicted_sql: Optional[str]
    latency_ms: float
    perf: dict
    peak_rss_bytes: Optional[int]
    peak_rss_note: str
    end_of_text_marker_stripped: bool
    command: list[str]
    returncode: Optional[int]
    error: Optional[str]


def run_gguf_once(
    canonical_prompt: str,
    settings: LlamaSettings,
    work_dir: Optional[Path] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> GgufRun:
    """One inference process (model is loaded per call). Never raises on a
    tool failure -- returns status/error so callers can record and set the
    exit code."""
    with tempfile.TemporaryDirectory(dir=work_dir) as td:
        prompt_file = Path(td) / "prompt.txt"
        # newline="" keeps bytes exact (no CRLF translation on Windows).
        with prompt_file.open("w", encoding="utf-8", newline="") as f:
            f.write(build_chatml_prompt(canonical_prompt) + ("\n" if settings.guard_prompt_trailing_newline else ""))
        cmd = build_llama_command(settings, prompt_file)
        result: ProcessResult = run_captured(cmd, timeout=settings.timeout_seconds, should_cancel=should_cancel)

    perf = parse_perf(result.stderr)
    base = dict(
        latency_ms=result.wall_ms,
        perf=perf,
        peak_rss_bytes=result.peak_rss_bytes,
        peak_rss_note=result.peak_rss_note,
        command=cmd,
        returncode=result.returncode,
    )
    if result.cancelled:
        return GgufRun("cancelled", None, None, end_of_text_marker_stripped=False, error="cancelled", **base)
    if result.timed_out:
        return GgufRun("timeout", None, None, end_of_text_marker_stripped=False,
                       error=f"timed out after {settings.timeout_seconds}s", **base)  # fmt: skip
    if result.returncode != 0:
        return GgufRun("error", None, None, end_of_text_marker_stripped=False,
                       error=f"exit code {result.returncode}: {result.stderr[-500:]}", **base)  # fmt: skip
    raw, stripped = clean_completion(result.stdout)
    return GgufRun("ok", raw, normalize_predicted_sql(raw), end_of_text_marker_stripped=stripped, error=None, **base)


def deployment_mode(settings: LlamaSettings) -> str:
    return "hot_lora" if settings.lora is not None else "single_gguf"


def deployment_info(settings: LlamaSettings) -> dict:
    """Hashes/sizes of every artifact making up the effective model."""
    base = artifact_record(settings.model)
    lora = artifact_record(settings.lora) if settings.lora is not None else None
    return {
        "mode": deployment_mode(settings),
        "base": base,
        "lora": lora,
        "combined_size_bytes": base["size_bytes"] + (lora["size_bytes"] if lora else 0),
    }


def describe_command(settings: LlamaSettings) -> str:
    """Display-only rendering with a placeholder prompt path (for dry-run)."""
    return format_command(build_llama_command(settings, Path("<prompt-file>")))


def settings_dict(settings: LlamaSettings) -> dict:
    return {
        "executable": str(settings.executable),
        "model": str(settings.model),
        "lora": None if settings.lora is None else str(settings.lora),
        "deployment_mode": deployment_mode(settings),
        "context_size": settings.context_size,
        "max_new_tokens": settings.max_new_tokens,
        "seed": settings.seed,
        "n_gpu_layers": settings.n_gpu_layers,
        "threads": settings.threads,
        "timeout_seconds": settings.timeout_seconds,
        "no_conversation_flag": settings.no_conversation_flag,
        "extra_args": list(settings.extra_args),
        "guard_prompt_trailing_newline": settings.guard_prompt_trailing_newline,
        "sampling": "greedy (--temp 0 --top-k 1)",
        "prompt_envelope": "qwen_chatml_single_user_turn",
    }


__all__ = [
    "LlamaSettings", "GgufRun", "build_chatml_prompt", "build_llama_command", "parse_perf",
    "clean_completion", "run_gguf_once", "describe_command", "settings_dict", "deployment_info", "deployment_mode",
]  # fmt: skip
