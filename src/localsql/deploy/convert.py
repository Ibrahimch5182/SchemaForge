"""Stage orchestration for the llama.cpp GGUF pipeline.

Stages (each invokes an EXTERNAL, explicitly supplied llama.cpp tool):
  base      HF base snapshot      -> base F16 GGUF     (convert_hf_to_gguf.py)
  lora      PEFT LoRA adapter     -> LoRA F16 GGUF     (convert_lora_to_gguf.py)
  quantize  base F16 GGUF         -> base Q4_K_M GGUF  (llama-quantize)
  merge     OPTIONAL: base GGUF + LoRA GGUF -> merged GGUF (llama-export-lora);
            not required -- the primary path loads the LoRA at runtime (--lora).

The flags below match recent llama.cpp builds but are NOT assumed stable:
every stage accepts `extra_args` and prints its exact command in dry-run so
they can be checked against the installed build's `--help` before running.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

from localsql.deploy.process import ProcessResult, format_command, run_captured
from localsql.deploy.provenance import artifact_record, dir_fingerprint

LOG_TAIL_CHARS = 4000


class StageError(RuntimeError):
    """A stage could not run or its tool failed."""


@dataclass(frozen=True)
class StagePlan:
    name: str
    command: list[str]
    output: Path
    tools: dict[str, Path]  # executables/scripts that must exist
    file_inputs: dict[str, Path] = field(default_factory=dict)
    dir_inputs: dict[str, Path] = field(default_factory=dict)
    extra_provenance: dict = field(default_factory=dict)

    def problems(self) -> list[str]:
        out = []
        for label, p in self.tools.items():
            if not Path(p).is_file():
                out.append(f"tool '{label}' not found: {p}")
        for label, p in self.file_inputs.items():
            if not Path(p).is_file():
                out.append(f"input '{label}' not found: {p}")
        for label, p in self.dir_inputs.items():
            if not Path(p).is_dir():
                out.append(f"input dir '{label}' not found: {p}")
        return out


def plan_base_gguf(
    python_exe: str, convert_script: Path, hf_dir: Path, output: Path, outtype: str, extra_args: Sequence[str] = ()
) -> StagePlan:
    cmd = [str(python_exe), str(convert_script), str(hf_dir), "--outfile", str(output), "--outtype", outtype, *extra_args]
    return StagePlan(
        "base", cmd, Path(output), {"convert_hf_to_gguf": Path(convert_script)}, dir_inputs={"hf_base": Path(hf_dir)}
    )


def plan_lora_gguf(
    python_exe: str,
    convert_script: Path,
    adapter_dir: Path,
    hf_dir: Path,
    output: Path,
    outtype: str,
    extra_args: Sequence[str] = (),
) -> StagePlan:
    cmd = [
        str(python_exe),
        str(convert_script),
        str(adapter_dir),
        "--base",
        str(hf_dir),
        "--outfile",
        str(output),
        "--outtype",
        outtype,
        *extra_args,
    ]
    return StagePlan(
        "lora",
        cmd,
        Path(output),
        {"convert_lora_to_gguf": Path(convert_script)},
        dir_inputs={"adapter": Path(adapter_dir), "hf_base": Path(hf_dir)},
    )


def plan_merge(
    export_lora_exe: Path, base_gguf: Path, lora_gguf: Path, output: Path, extra_args: Sequence[str] = ()
) -> StagePlan:
    cmd = [str(export_lora_exe), "-m", str(base_gguf), "--lora", str(lora_gguf), "-o", str(output), *extra_args]
    return StagePlan(
        "merge",
        cmd,
        Path(output),
        {"llama_export_lora": Path(export_lora_exe)},
        file_inputs={"base_gguf": Path(base_gguf), "lora_gguf": Path(lora_gguf)},
    )


def plan_quantize(
    quantize_exe: Path, input_gguf: Path, output: Path, quant_type: str, extra_args: Sequence[str] = ()
) -> StagePlan:
    # llama-quantize usage: [options] input.gguf [output.gguf] type
    cmd = [str(quantize_exe), *extra_args, str(input_gguf), str(output), quant_type]
    return StagePlan(
        "quantize",
        cmd,
        Path(output),
        {"llama_quantize": Path(quantize_exe)},
        file_inputs={"input_gguf": Path(input_gguf)},
        extra_provenance={"quant_type": quant_type},
    )


def partial_path(output: Path) -> Path:
    return output.with_name(f"{output.stem}.partial{output.suffix}")


def record_path(records_dir: Path, plan: StagePlan) -> Path:
    return Path(records_dir) / f"stage-{plan.name}.json"


def dry_run_report(plan: StagePlan) -> dict:
    """What would run. Touches nothing on disk and hashes nothing."""
    return {
        "stage": plan.name,
        "dry_run": True,
        "command": plan.command,
        "command_display": format_command(plan.command),
        "output": str(plan.output),
        "output_exists": plan.output.exists(),
        "preflight_problems": plan.problems(),
    }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _input_records(plan: StagePlan) -> dict:
    recs: dict = {}
    for label, p in plan.file_inputs.items():
        recs[label] = artifact_record(p)
    for label, p in plan.dir_inputs.items():
        recs[label] = dir_fingerprint(p)
    return recs


def _tail(text: str) -> str:
    return text[-LOG_TAIL_CHARS:]


def _existing_output_matches(plan: StagePlan, rec_path: Path, inputs: dict) -> dict:
    """Return the stored record if `plan.output` is a verified, identical
    prior result of this exact stage; otherwise raise (never overwrite)."""
    hint = "Move/delete it deliberately or choose a different output path."
    if not rec_path.is_file():
        raise StageError(f"Output already exists without a stage record: {plan.output}. Refusing to overwrite. {hint}")
    rec = json.loads(rec_path.read_text(encoding="utf-8"))
    current = artifact_record(plan.output)
    same = (
        rec.get("command") == plan.command
        and rec.get("inputs") == inputs
        and rec.get("output", {}).get("sha256") == current["sha256"]
    )
    if not same:
        raise StageError(
            f"Output exists but differs from the recorded run (command, inputs, or content changed): "
            f"{plan.output}. Refusing to overwrite. {hint}"
        )
    return rec


def execute_stage(plan: StagePlan, records_dir: Path, timeout: Optional[float] = None) -> dict:
    """Run one stage for real. Raises StageError on any failure; returns the
    (also persisted) stage record."""
    problems = plan.problems()
    if problems:
        raise StageError(f"Stage '{plan.name}' preflight failed:\n  - " + "\n  - ".join(problems))
    records_dir = Path(records_dir)
    records_dir.mkdir(parents=True, exist_ok=True)
    rec_path = record_path(records_dir, plan)
    inputs = _input_records(plan)

    if plan.output.exists():
        rec = _existing_output_matches(plan, rec_path, inputs)
        return {**rec, "status": "already_complete"}

    plan.output.parent.mkdir(parents=True, exist_ok=True)
    partial = partial_path(plan.output)
    if partial.exists():
        partial.unlink()  # our own leftover from an interrupted run
    executed = [str(partial) if a == str(plan.output) else a for a in plan.command]

    started = _now()
    result: ProcessResult = run_captured(executed, timeout=timeout)
    log_path = records_dir / f"stage-{plan.name}.log"
    log_path.write_text(
        f"$ {format_command(executed)}\n--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}\n",
        encoding="utf-8",
    )
    if result.timed_out:
        raise StageError(f"Stage '{plan.name}' timed out after {timeout}s. Log: {log_path}")
    if result.returncode != 0:
        raise StageError(
            f"Stage '{plan.name}' failed with exit code {result.returncode}. Log: {log_path}\n"
            f"stderr tail:\n{_tail(result.stderr)}"
        )
    if not partial.is_file() or partial.stat().st_size == 0:
        raise StageError(f"Stage '{plan.name}' exited 0 but produced no output at {partial}. Log: {log_path}")
    partial.rename(plan.output)

    record = {
        "stage": plan.name,
        "status": "completed",
        "started_utc": started,
        "finished_utc": _now(),
        "command": plan.command,
        "executed_command": executed,
        "command_display": format_command(plan.command),
        "returncode": result.returncode,
        "wall_seconds": round(result.wall_ms / 1000, 2),
        "peak_rss_bytes": result.peak_rss_bytes,
        "tools": {k: str(v) for k, v in plan.tools.items()},
        "inputs": inputs,
        "output": artifact_record(plan.output),
        "log_path": str(log_path),
        "stdout_tail": _tail(result.stdout),
        "stderr_tail": _tail(result.stderr),
        **plan.extra_provenance,
    }
    rec_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record
