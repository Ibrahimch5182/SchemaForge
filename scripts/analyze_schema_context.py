"""Phase 5A: schema-context token analysis and deterministic budgeting.

ANALYSIS AND DESIGN ONLY. Does not modify Phase 1's canonical
`data/processed/{train,validation}.jsonl`, does not train/run a model, and
does not touch BIRD Mini-Dev. Offline: reuses the already-cached BIRD
schema/column-meaning files from Phase 1 (no network needed if they are
already cached; falls back to the same legitimate Phase 1 sources
otherwise).

Token counts are ESTIMATES (`localsql.schema_context.token_estimate`),
calibrated against real Qwen-tokenizer numbers already obtained on Kaggle
in Phase 4 -- not a substitute for a real tokenizer run. See
`--write-real-profile-inputs` for how to get exact numbers on Kaggle.

Usage:
    uv run python scripts/analyze_schema_context.py --split train
    uv run python scripts/analyze_schema_context.py --split validation
    uv run python scripts/analyze_schema_context.py --split train --budget-tokens 3584 --write-derived
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from localsql.data.bird_loader import (  # noqa: E402
    COLUMN_MEANING_FILE,
    DATASET_REPO_ID,
    download_column_meaning,
    load_column_meaning,
)
from localsql.data.models import DatabaseSchema  # noqa: E402
from localsql.data.prompt_builder import build_prompt  # noqa: E402
from localsql.data.schema_loader import fetch_official_train_tables_json, parse_database_schemas  # noqa: E402
from localsql.data.schema_serializer import serialize_schema  # noqa: E402
from localsql.schema_context.coverage import (  # noqa: E402
    compute_coverage,
    extract_gold_references,
    summarize_coverage,
)
from localsql.schema_context.compact_serializer import serialize_schema_compact  # noqa: E402
from localsql.schema_context.relevance import select_schema_within_budget  # noqa: E402
from localsql.schema_context.token_estimate import CALIBRATED_CHARS_PER_TOKEN, estimate_tokens  # noqa: E402
from localsql.model.run_artifacts import numeric_stats  # noqa: E402

DEFAULT_THRESHOLDS = (3584, 4096, 8192)


def load_schemas() -> dict[str, DatabaseSchema]:
    """Reuse Phase 1's exact schema sources (train_tables.json + BIRD
    column-meaning), already cached locally from Phase 1's own run."""
    raw_dir = REPO_ROOT / "data" / "raw"
    schema_path = raw_dir / "train_tables.json"
    fetch_official_train_tables_json(schema_path)
    meaning_path = download_column_meaning(DATASET_REPO_ID, COLUMN_MEANING_FILE)
    column_meaning = load_column_meaning(meaning_path)
    return parse_database_schemas(schema_path, column_meaning, dialect="sqlite")


def load_split_examples(split: str) -> list[dict]:
    filename = "train.jsonl" if split == "train" else "validation.jsonl"
    path = REPO_ROOT / "data" / "processed" / filename
    if not path.exists():
        print(f"BLOCKER: {path} not found. Run `uv run python scripts/prepare_bird.py` first (Phase 1).")
        sys.exit(1)
    examples = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def threshold_counts(values: list[float], thresholds=DEFAULT_THRESHOLDS) -> dict:
    return {f">{t}": sum(1 for v in values if v > t) for t in thresholds}


def per_db_table_column_counts(schemas: dict[str, DatabaseSchema], db_ids: set[str]) -> dict:
    out = {}
    for db_id in sorted(db_ids):
        s = schemas[db_id]
        out[db_id] = {"tables": len(s.tables), "columns": sum(len(t.columns) for t in s.tables)}
    return out


def run_analysis(split: str, budget_tokens: int, write_derived: bool) -> dict:
    print(f"Loading schemas (reusing Phase 1's cached sources) ...")
    schemas = load_schemas()
    examples = load_split_examples(split)
    db_ids = sorted({e["db_id"] for e in examples})
    print(f"  {len(examples)} examples across {len(db_ids)} databases ({split})")

    schema_tables = {
        db_id: {t.name: [c.name for c in t.columns] for t in schemas[db_id].tables} for db_id in db_ids
    }

    # --- Task 1: representation A vs B, per-DB character/estimated-token lengths ---
    compact_cache: dict[str, str] = {}
    canonical_cache: dict[str, str] = {}
    for db_id in db_ids:
        canonical_cache[db_id] = serialize_schema(schemas[db_id])
        compact_cache[db_id] = serialize_schema_compact(schemas[db_id])

    per_db_report = {}
    for db_id in db_ids:
        a_text, b_text = canonical_cache[db_id], compact_cache[db_id]
        per_db_report[db_id] = {
            "tables": len(schemas[db_id].tables),
            "columns": sum(len(t.columns) for t in schemas[db_id].tables),
            "schema_only_chars_A": len(a_text),
            "schema_only_chars_B": len(b_text),
            "schema_only_char_reduction_pct": round(100 * (1 - len(b_text) / len(a_text)), 1) if a_text else 0.0,
            "schema_only_estimated_tokens_A": round(estimate_tokens(a_text), 1),
            "schema_only_estimated_tokens_B": round(estimate_tokens(b_text), 1),
        }

    # --- Task 2: whole-split full-SFT distributions, representation A (real, from file) vs B (estimated) ---
    est_A = [estimate_tokens(e["prompt"] + e["completion"]) for e in examples]

    est_B: list[float] = []
    over_budget_examples: list[dict] = []
    for e in examples:
        schema_b = compact_cache[e["db_id"]]
        prompt_b = build_prompt(schema_b, e["dialect"], e["question"], e.get("business_context"))
        tok = estimate_tokens(prompt_b + e["completion"])
        est_B.append(tok)
        if tok > budget_tokens:
            over_budget_examples.append(e)

    print(f"  representation B: {len(over_budget_examples)} / {len(examples)} examples exceed budget={budget_tokens}")

    # --- Task 3 + 4 (only if needed): budgeter + gold-coverage diagnostic for over-budget examples ---
    budgeting_report = None
    coverage_report = None
    if over_budget_examples:
        # Leave headroom for question/evidence/completion/chat overhead below the full budget.
        schema_only_budget_tokens = max(int(budget_tokens * 0.92), 1)
        schema_only_char_budget = int(schema_only_budget_tokens * CALIBRATED_CHARS_PER_TOKEN)

        budgeted_est: list[float] = []
        coverage_results = []
        for e in over_budget_examples:
            schema = schemas[e["db_id"]]
            sel = select_schema_within_budget(
                schema, e["question"], e.get("business_context"), budget=schema_only_char_budget
            )
            prompt = build_prompt(sel.serialized_schema, e["dialect"], e["question"], e.get("business_context"))
            tok = estimate_tokens(prompt + e["completion"])
            budgeted_est.append(tok)

            gold = extract_gold_references(e["completion"], schema_tables[e["db_id"]], dialect=e["dialect"])
            coverage_results.append(
                compute_coverage(e["example_id"], frozenset(sel.selected_table_names), gold)
            )

        budgeting_report = {
            "schema_only_char_budget": schema_only_char_budget,
            "examples_budgeted": len(over_budget_examples),
            "estimated_tokens_after_budgeting": numeric_stats(budgeted_est),
            "still_over_budget_after_budgeting": sum(1 for t in budgeted_est if t > budget_tokens),
        }
        coverage_report = summarize_coverage(coverage_results)

    report = {
        "split": split,
        "example_count": len(examples),
        "database_count": len(db_ids),
        "budget_tokens_candidate": budget_tokens,
        "token_estimation_method": {
            "description": "Character-count / calibrated-chars-per-token. NOT the real Qwen tokenizer.",
            "calibrated_chars_per_token": CALIBRATED_CHARS_PER_TOKEN,
            "calibration_note": (
                "Calibrated against real Kaggle Phase 4 per-database median token counts for 8 "
                "long-context databases; validated against the real whole-training-set min/median/max "
                "(predicted vs. real: median +8.7%, min -6.9%, max -0.1%). Use "
                "scripts/run_qlora_smoke.py --token-profile on Kaggle for exact numbers."
            ),
        },
        "per_database": per_db_report,
        "representation_A_canonical_full_sft": {
            "stats": numeric_stats(est_A),
            "counts_over_threshold": threshold_counts(est_A),
        },
        "representation_B_compact_full_sft": {
            "stats": numeric_stats(est_B),
            "counts_over_threshold": threshold_counts(est_B),
        },
        "budgeting_needed": bool(over_budget_examples),
        "budgeting": budgeting_report,
        "gold_coverage_diagnostic": coverage_report,
    }

    if write_derived:
        write_derived_dataset(split, examples, schemas, compact_cache, budget_tokens, REPO_ROOT / "data" / "processed_context_budgeted")

    return report


def build_derived_records(
    examples: list[dict],
    schemas: dict[str, DatabaseSchema],
    compact_cache: dict[str, str],
    budget_tokens: int,
) -> list[dict]:
    """Pure transformation (no I/O): one derived record per input example,
    in the same order, with the same count -- no example is ever dropped.
    `completion` is copied verbatim (byte-identical, never truncated).
    Examples whose full compact-schema SFT sequence already fits the
    budget are left as `compact_full` and unchanged in content; only
    over-budget examples get `compact_budgeted` (deterministic
    question-conditioned selection, gold-free).
    """
    schema_only_budget_tokens = max(int(budget_tokens * 0.92), 1)
    schema_only_char_budget = int(schema_only_budget_tokens * CALIBRATED_CHARS_PER_TOKEN)

    derived = []
    for e in examples:
        schema = schemas[e["db_id"]]
        compact_text = compact_cache[e["db_id"]]
        est_full_compact = estimate_tokens(
            build_prompt(compact_text, e["dialect"], e["question"], e.get("business_context")) + e["completion"]
        )
        if est_full_compact <= budget_tokens:
            schema_text = compact_text
            budgeted = False
        else:
            sel = select_schema_within_budget(
                schema, e["question"], e.get("business_context"), budget=schema_only_char_budget
            )
            schema_text = sel.serialized_schema
            budgeted = True

        prompt = build_prompt(schema_text, e["dialect"], e["question"], e.get("business_context"))
        derived.append(
            {
                "example_id": e["example_id"],
                "db_id": e["db_id"],
                "dialect": e["dialect"],
                "split": e["split"],
                "source": e["source"],
                "question": e["question"],
                "business_context": e.get("business_context"),
                "schema_representation": "compact_budgeted" if budgeted else "compact_full",
                "serialized_schema": schema_text,
                "prompt": prompt,
                "completion": e["completion"],  # byte-identical to the original -- never truncated
            }
        )

    return derived


def assert_derived_dataset_integrity(
    derived: list[dict], examples: list[dict], expected_count: int
) -> None:
    """Task 7 assertions, factored out so tests can exercise them directly
    (with tiny synthetic data) without writing any files."""
    assert len(derived) == len(examples) == expected_count, (
        f"example count changed: derived={len(derived)} original={len(examples)} expected={expected_count}"
    )
    assert {d["example_id"] for d in derived} == {e["example_id"] for e in examples}, "example_id set changed"
    assert {d["db_id"] for d in derived} == {e["db_id"] for e in examples}, "db_id set changed (split unchanged)"
    for d, e in zip(derived, examples):
        assert d["completion"] == e["completion"], f"completion mutated for {d['example_id']}"


def write_derived_dataset(
    split: str,
    examples: list[dict],
    schemas: dict[str, DatabaseSchema],
    compact_cache: dict[str, str],
    budget_tokens: int,
    out_dir: Path,
) -> Path:
    """Task 7: write a SEPARATE derived artifact under `out_dir`, never
    touching the original Phase 1 `data/processed/{split}.jsonl`."""
    import hashlib

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{split}.jsonl"

    derived = build_derived_records(examples, schemas, compact_cache, budget_tokens)

    expected_count = 6067 if split == "train" else 534 if split == "validation" else len(examples)
    assert_derived_dataset_integrity(derived, examples, expected_count)

    with out_path.open("w", encoding="utf-8") as f:
        for d in derived:
            f.write(json.dumps(d) + "\n")

    original_path = REPO_ROOT / "data" / "processed" / f"{split}.jsonl"
    schema_only_budget_tokens = max(int(budget_tokens * 0.92), 1)
    provenance = {
        "split": split,
        "transformation_version": "phase5a-context-budget-v1",
        "budget_tokens_candidate": budget_tokens,
        "schema_only_char_budget": int(schema_only_budget_tokens * CALIBRATED_CHARS_PER_TOKEN),
        "derived_example_count": len(derived),
        "original_example_count": len(examples),
        "original_file_sha256": hashlib.sha256(original_path.read_bytes()).hexdigest() if original_path.exists() else None,
        "derived_file_sha256": hashlib.sha256(out_path.read_bytes()).hexdigest(),
        "budgeted_example_count": sum(1 for d in derived if d["schema_representation"] == "compact_budgeted"),
    }
    (out_dir / f"{split}_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(f"  Wrote derived dataset: {out_path}")
    print(f"  Wrote provenance: {out_dir / f'{split}_provenance.json'}")
    print(f"  Assertions passed: count={len(derived)}, db_id set unchanged, completions byte-identical.")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "validation"], default="train")
    parser.add_argument("--budget-tokens", type=int, default=4096)
    parser.add_argument("--write-derived", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    report = run_analysis(args.split, args.budget_tokens, args.write_derived)

    out_path = args.out or (REPO_ROOT / "data" / "reports" / f"schema_context_{args.split}_b{args.budget_tokens}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n=== {args.split} summary (budget candidate: {args.budget_tokens}) ===")
    print(f"Representation A over threshold: {report['representation_A_canonical_full_sft']['counts_over_threshold']}")
    print(f"Representation B over threshold: {report['representation_B_compact_full_sft']['counts_over_threshold']}")
    if report["budgeting"]:
        print(f"Budgeting: {report['budgeting']}")
    if report["gold_coverage_diagnostic"]:
        print(f"Gold coverage diagnostic (NOT used for selection): {report['gold_coverage_diagnostic']}")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
