"""Run a GGUF model locally through llama.cpp on the fixed Phase 7 sanity
prompt set (deterministic, gold-free subset of the Phase 2 generation
manifest). Uses the canonical SchemaForge prompt verbatim; CPU inference is
valid, GPU offload is optional. Exit code is nonzero if any request fails.

This is NOT a BIRD accuracy run. Run it once per base GGUF (F16, Q4_K_M) with the SAME --lora
and compare the outputs with scripts/compare_gguf_outputs.py.

    uv run python scripts/run_local_gguf.py --llama-exe C:\\llama.cpp\\llama-completion.exe `
        --model .artifacts\\phase7\\q4_k_m\\base-Q4_K_M.gguf `
        --lora .artifacts/phase7/lora-gguf/lora-1518-f16.gguf `
        --output .artifacts\\phase7\\benchmarks\\sanity-q4_k_m-base-lora.json --label q4_k_m
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from localsql.deploy.cli import add_llama_args, manifest_path, require_lora_or_opt_out, settings_from_args  # noqa: E402
from localsql.deploy.config import load_phase7_config  # noqa: E402
from localsql.deploy.runtime import deployment_info, describe_command, run_gguf_once, settings_dict  # noqa: E402
from localsql.deploy.sanity import load_generation_manifest, select_sanity_examples, text_sha256  # noqa: E402


def select_examples(args, cfg):
    examples = load_generation_manifest(manifest_path(args, cfg))
    if args.example_id:
        by_id = {e.example_id: e for e in examples}
        missing = [i for i in args.example_id if i not in by_id]
        if missing:
            raise SystemExit(f"BLOCKER: example ids not in manifest: {missing}")
        return [by_id[i] for i in args.example_id]
    return select_sanity_examples(examples, cfg.quality_sanity.sample_size, cfg.quality_sanity.selection_salt)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_llama_args(parser)
    parser.add_argument("--output", type=Path, required=True, help="Result JSON path (not overwritten unless --force).")
    parser.add_argument("--label", type=str, default="", help="Free-text label recorded in the output.")
    parser.add_argument("--example-id", action="append", default=[], help="Override the sanity set (repeatable).")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing --output.")
    args = parser.parse_args()

    cfg = load_phase7_config(args.config)
    require_lora_or_opt_out(args)
    settings = settings_from_args(args, cfg)
    examples = select_examples(args, cfg)

    if args.dry_run:
        print(f"Would run {len(examples)} prompt(s), one llama.cpp process each:")
        print(f"  {describe_command(settings)}")
        print("Example ids: " + ", ".join(e.example_id for e in examples))
        print("Dry run OK -- nothing executed, no model hashed.")
        return 0

    if args.output.exists() and not args.force:
        print(f"BLOCKER: {args.output} exists; pass --force to overwrite.")
        return 1
    for label, p in (("llama executable", settings.executable), ("model", settings.model), ("lora", settings.lora)):
        if p is None:
            continue
        if not Path(p).is_file():
            print(f"BLOCKER: {label} not found: {p}")
            return 1

    deployment = deployment_info(settings)
    results, failures = [], 0
    for ex in examples:
        run = run_gguf_once(ex.prompt, settings)
        failures += run.status != "ok"
        print(f"  {ex.example_id}: {run.status} {run.latency_ms:.0f} ms")
        results.append(
            {
                "example_id": ex.example_id,
                "db_id": ex.db_id,
                "status": run.status,
                "raw_completion": run.raw_completion,
                "predicted_sql": run.predicted_sql,
                "predicted_sql_sha256": text_sha256(run.predicted_sql),
                "latency_ms": round(run.latency_ms, 2),
                "perf": run.perf,
                "peak_rss_bytes": run.peak_rss_bytes,
                "peak_rss_note": run.peak_rss_note,
                "end_of_text_marker_stripped": run.end_of_text_marker_stripped,
                "command": run.command,
                "error": run.error,
            }
        )

    doc = {
        "kind": "gguf_local_generation",
        "label": args.label,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": deployment["base"],  # base GGUF (kept for compare tooling)
        "lora": deployment["lora"],
        "deployment": deployment,  # effective model: base + LoRA, incl. combined_size_bytes
        "settings": settings_dict(settings),
        "sample": {
            "manifest": str(manifest_path(args, cfg)),
            "selection_salt": cfg.quality_sanity.selection_salt,
            "example_ids": [e.example_id for e in examples],
        },
        "failures": failures,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(f"Wrote {args.output} ({len(results) - failures}/{len(results)} ok)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
