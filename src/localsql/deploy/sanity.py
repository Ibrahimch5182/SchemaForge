"""Quantization regression/sanity helpers and runtime-benchmark statistics.

The sanity set is a small, fixed, gold-free prompt subset used to compare a
F16 base GGUF + LoRA GGUF against Q4_K_M base GGUF + the SAME LoRA GGUF. It is NOT an
accuracy benchmark and never reads grading/gold files.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional, Sequence

from localsql.benchmark.diagnostics import PARSE_OK, check_parse
from localsql.benchmark.models import GenerationExample
from localsql.model.run_artifacts import percentile


def load_generation_manifest(path: Path) -> list[GenerationExample]:
    """Gold-free manifest only: `GenerationExample` forbids extra (gold) fields."""
    examples = []
    with Path(path).open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    examples.append(GenerationExample.model_validate_json(line))
                except Exception as e:
                    raise ValueError(
                        f"{path}:{line_no}: not a gold-free GenerationExample ({e}). "
                        "Point at the generation manifest, never the grading reference."
                    ) from e
    return examples


def select_sanity_examples(
    examples: Sequence[GenerationExample], sample_size: int, salt: str
) -> list[GenerationExample]:
    """Deterministic subset: order by sha256(salt:example_id), take the first
    N. Independent of input order and of Python's hash seed."""
    ranked = sorted(examples, key=lambda e: hashlib.sha256(f"{salt}:{e.example_id}".encode()).hexdigest())
    return list(ranked[:sample_size])


def text_sha256(text: Optional[str]) -> Optional[str]:
    return None if text is None else hashlib.sha256(text.encode("utf-8")).hexdigest()


def latency_summary(values_ms: Sequence[float]) -> dict:
    """P50/P95 (linear interpolation, same `percentile` as the rest of the
    repo) plus mean/min/max. Empty input yields nulls, not zeros."""
    if not values_ms:
        return {"count": 0, "p50_ms": None, "p95_ms": None, "mean_ms": None, "min_ms": None, "max_ms": None}
    s = sorted(values_ms)
    return {
        "count": len(s),
        "p50_ms": round(percentile(s, 50), 2),
        "p95_ms": round(percentile(s, 95), 2),
        "mean_ms": round(sum(s) / len(s), 2),
        "min_ms": s[0],
        "max_ms": s[-1],
    }


def rate_summary(values: Sequence[Optional[float]], name: str) -> dict:
    """Mean of a per-run rate; null with an explanation when no run exposed it."""
    present = [v for v in values if v is not None]
    if not present:
        return {"mean": None, "count": 0, "availability": f"not_available: no run exposed {name}"}
    return {"mean": round(sum(present) / len(present), 2), "count": len(present), "availability": "measured"}


def sql_shape_diagnostics(sql: Optional[str], dialect: str = "sqlite") -> dict:
    """Parse-level shape only (reuses the Phase 2 diagnostic). Not correctness."""
    if not sql:
        return {"parseable": False, "starts_with_select_or_with": False, "has_markdown_fence": False}
    status, _, _ = check_parse(sql, dialect)
    head = sql.lstrip().lower()
    return {
        "parseable": status == PARSE_OK,
        "starts_with_select_or_with": head.startswith(("select", "with")),
        "has_markdown_fence": "```" in sql,
    }


def compare_generations(reference: dict, candidate: dict) -> dict:
    """Compare two `run_local_gguf` result documents by example_id.

    Agreement is exact match of `predicted_sql` (whitespace-trim normalized,
    the repo's only normalization). Failed generations count as disagreement.
    """
    ref_lora = (reference.get("lora") or {}).get("sha256")
    cand_lora = (candidate.get("lora") or {}).get("sha256")
    if ref_lora != cand_lora:
        raise ValueError(
            f"LoRA GGUF differs between runs (reference={ref_lora}, candidate={cand_lora}); the quantization "
            "sanity check requires the SAME LoRA GGUF for both so only the base representation changes."
        )
    ref = {r["example_id"]: r for r in reference["results"]}
    cand = {r["example_id"]: r for r in candidate["results"]}
    if set(ref) != set(cand):
        raise ValueError(
            f"Result sets differ: only in reference={sorted(set(ref) - set(cand))}, "
            f"only in candidate={sorted(set(cand) - set(ref))}"
        )
    changed = []
    agree = 0
    for eid in sorted(ref):
        a, b = ref[eid].get("predicted_sql"), cand[eid].get("predicted_sql")
        ok = a is not None and a == b
        agree += ok
        if not ok:
            changed.append({"example_id": eid, "reference_sql": a, "candidate_sql": b})
    n = len(ref)

    def shapes(results: dict) -> dict:
        diags = [sql_shape_diagnostics(r.get("predicted_sql")) for r in results.values()]
        return {k: sum(d[k] for d in diags) for k in ("parseable", "starts_with_select_or_with", "has_markdown_fence")}

    return {
        "kind": "quantization_regression_sanity_check",
        "note": "Output-agreement sanity check on a small fixed sample; NOT an accuracy benchmark.",
        "n": n,
        "exact_normalized_agreement": agree,
        "exact_normalized_agreement_rate": None if n == 0 else round(agree / n, 4),
        "shared_lora_sha256": ref_lora,
        "comparison": "base GGUF (reference) + LoRA vs base GGUF (candidate) + the same LoRA",
        "reference_model_sha256": reference.get("model", {}).get("sha256"),
        "candidate_model_sha256": candidate.get("model", {}).get("sha256"),
        "reference_shape_counts": shapes(ref),
        "candidate_shape_counts": shapes(cand),
        "changed_examples": changed,
    }
