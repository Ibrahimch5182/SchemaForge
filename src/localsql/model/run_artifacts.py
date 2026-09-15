"""Run directory layout, resume/checkpoint logic, and provenance recording.

No torch/transformers import here -- pure filesystem/JSON bookkeeping, so
it is fully testable without CUDA or the optional "model" dependency group.
"""

from __future__ import annotations

import hashlib
import json
import platform
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional


def percentile(sorted_values: list[float], pct: float) -> float:
    """Linear-interpolation percentile (pct in [0, 100]) of an already-sorted list."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (pct / 100) * (len(sorted_values) - 1)
    lo, hi = int(rank), min(int(rank) + 1, len(sorted_values) - 1)
    frac = rank - lo
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * frac


def numeric_stats(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "min": None, "median": None, "p90": None, "p95": None, "p99": None, "max": None}
    s = sorted(values)
    return {
        "count": len(s),
        "min": s[0],
        "median": statistics.median(s),
        "p90": round(percentile(s, 90), 2),
        "p95": round(percentile(s, 95), 2),
        "p99": round(percentile(s, 99), 2),
        "max": s[-1],
    }


class ResumeConflictError(RuntimeError):
    """Raised when resuming a run whose stored config doesn't match the
    requested one -- the caller must pick a new `run_id` instead."""


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class RunConfig:
    """Everything that defines "the same run" for resume-safety purposes."""

    run_id: str
    model_id: str
    model_revision: Optional[str]
    quantization: dict
    generation: dict
    context_mode: str
    manifest_path: str
    manifest_sha256: str
    limit: Optional[int] = None

    _IDENTITY_FIELDS = (
        "model_id",
        "model_revision",
        "quantization",
        "generation",
        "context_mode",
        "manifest_sha256",
        "limit",
    )

    def matches(self, other: "RunConfig") -> bool:
        return all(getattr(self, f) == getattr(other, f) for f in self._IDENTITY_FIELDS)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "RunConfig":
        return cls(**data)


@dataclass(frozen=True)
class Provenance:
    """Environment/software metadata recorded with every real run."""

    python_version: str
    torch_version: Optional[str] = None
    transformers_version: Optional[str] = None
    bitsandbytes_version: Optional[str] = None
    cuda_version: Optional[str] = None
    gpu_name: Optional[str] = None
    gpu_total_memory_mb: Optional[float] = None

    @classmethod
    def collect(cls, backend_info: Optional[Any] = None) -> "Provenance":
        """`backend_info` is a `qwen_backend.LoadedModelInfo` when a real
        model was loaded; omitted entirely in dry-run mode."""
        kwargs: dict = {"python_version": platform.python_version()}
        if backend_info is not None:
            kwargs.update(
                torch_version=backend_info.torch_version,
                transformers_version=backend_info.transformers_version,
                bitsandbytes_version=backend_info.bitsandbytes_version,
                cuda_version=backend_info.cuda_version,
                gpu_name=backend_info.gpu_name,
                gpu_total_memory_mb=backend_info.gpu_total_memory_mb,
            )
        return cls(**kwargs)

    def to_dict(self) -> dict:
        return asdict(self)


class RunDirectory:
    """Layout: run_config.json, generations.jsonl, predictions.jsonl, summary.json."""

    def __init__(self, runs_root: Path, run_id: str):
        self.run_id = run_id
        self.root = runs_root / run_id
        self.run_config_path = self.root / "run_config.json"
        self.generations_path = self.root / "generations.jsonl"
        self.predictions_path = self.root / "predictions.jsonl"
        self.summary_path = self.root / "summary.json"

    def prepare(self, config: RunConfig) -> None:
        """Validate resume-safety (raises `ResumeConflictError` on mismatch)
        and write/refresh `run_config.json`. Safe to call for a fresh run."""
        self.root.mkdir(parents=True, exist_ok=True)
        if self.run_config_path.exists():
            existing = RunConfig.from_dict(json.loads(self.run_config_path.read_text(encoding="utf-8")))
            if not existing.matches(config):
                raise ResumeConflictError(
                    f"Run '{config.run_id}' already exists with a different configuration "
                    "(model/revision/quantization/generation/context_mode/manifest/limit). "
                    "Use a new --run-id for a different configuration."
                )
        self.run_config_path.write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")

    def completed_example_ids(self) -> set[str]:
        """`example_id`s already recorded with `status == "ok"` -- resume skips these."""
        if not self.generations_path.exists():
            return set()
        completed: set[str] = set()
        with self.generations_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("status") == "ok":
                    completed.add(rec["example_id"])
        return completed

    def append_generation(self, record: dict) -> None:
        with self.generations_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
            f.flush()

    def rewrite_predictions(self, model_id: str) -> int:
        """Rebuild `predictions.jsonl` from `generations.jsonl` (the source
        of truth). A full rebuild, keyed by `example_id` with the latest
        record winning, so re-running never duplicates or drifts out of
        sync even after a crash mid-write. Returns the row count written.
        """
        if not self.generations_path.exists():
            return 0
        latest: dict[str, dict] = {}
        with self.generations_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("status") == "ok":
                    latest[rec["example_id"]] = rec

        with self.predictions_path.open("w", encoding="utf-8") as f:
            for rec in latest.values():
                prediction = {
                    "example_id": rec["example_id"],
                    "db_id": rec["db_id"],
                    "predicted_sql": rec["predicted_sql"],
                    "latency_ms": rec.get("latency_ms"),
                    "prompt_tokens": rec.get("input_tokens"),
                    "completion_tokens": rec.get("output_tokens"),
                    "model_id": model_id,
                    "context_mode": rec.get("context_mode"),
                }
                f.write(json.dumps(prediction) + "\n")
            f.flush()
        return len(latest)

    def write_summary(self, summary: dict) -> None:
        self.summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
