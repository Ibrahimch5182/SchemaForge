"""Runtime benchmark of the EFFECTIVE model (base GGUF + runtime LoRA GGUF) via llama.cpp: latency (P50/P95), generation
and prompt-processing throughput (from llama.cpp's own timing lines), peak
process memory, file size, and determinism across repeats.

Each request is a fresh llama.cpp process, so wall latency INCLUDES model
load; llama.cpp's own eval timings (which exclude load) are reported
separately. Metrics a build does not expose are recorded as null with an
explanation, never estimated. Warm-up requests are not measured.

    uv run python scripts/benchmark_gguf.py --llama-exe C:\\llama.cpp\\llama-completion.exe `
        --model .artifacts\\phase7\\q4_k_m\\base-Q4_K_M.gguf `
        --lora .artifacts/phase7/lora-gguf/lora-1518-f16.gguf `
        --output .artifacts\\phase7\\benchmarks\\bench-q4_k_m.json
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
from localsql.deploy.sanity import (  # noqa: E402
    latency_summary,
    load_generation_manifest,
    rate_summary,
    select_sanity_examples,
    text_sha256,
)


def summarize(measured: list[dict], per_prompt_hashes: dict[str, set]) -> dict:
    ok = [r for r in measured if r["status"] == "ok"]
    peaks = [r["peak_rss_bytes"] for r in measured if r["peak_rss_bytes"] is not None]
    inference_ms = [
        r["perf"]["prompt_eval_ms"] + r["perf"]["eval_ms"]
        for r in ok
        if r["perf"]["prompt_eval_ms"] is not None and r["perf"]["eval_ms"] is not None
    ]
    return {
        "requests_total": len(measured),
        "requests_ok": len(ok),
        "requests_failed": len(measured) - len(ok),
        "wall_latency_including_model_load": latency_summary([r["latency_ms"] for r in ok]),
        "inference_latency_excluding_load": (
            latency_summary(inference_ms) if inference_ms else "not_available: llama.cpp timing lines not parsed"
        ),
        "generation_tokens_per_second": rate_summary(
            [r["perf"]["generated_tokens_per_second"] for r in ok], "generation tokens/sec"
        ),
        "prompt_tokens_per_second": rate_summary(
            [r["perf"]["prompt_tokens_per_second"] for r in ok], "prompt-processing tokens/sec"
        ),
        "peak_rss_bytes_max": max(peaks) if peaks else None,
        "peak_rss_note": (
            "max over requests of the launched process's peak working set (includes touched mmapped model pages)"
            if peaks
            else "not_available: peak memory could not be read on this platform"
        ),
        "deterministic_across_repeats": bool(per_prompt_hashes) and all(len(v) == 1 for v in per_prompt_hashes.values()),
        "distinct_outputs_per_prompt": {k: len(v) for k, v in per_prompt_hashes.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_llama_args(parser)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-prompts", type=int, default=None)
    parser.add_argument("--warmup-runs", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=None, help="Measured runs per prompt.")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = load_phase7_config(args.config)
    bcfg = cfg.benchmark
    num_prompts = args.num_prompts or bcfg.num_prompts
    warmup = bcfg.warmup_runs if args.warmup_runs is None else args.warmup_runs
    repeats = args.repeats or bcfg.measured_repeats
    require_lora_or_opt_out(args)
    settings = settings_from_args(args, cfg)

    examples = select_sanity_examples(
        load_generation_manifest(manifest_path(args, cfg)), cfg.quality_sanity.sample_size, cfg.quality_sanity.selection_salt
    )[:num_prompts]

    if args.dry_run:
        print(f"Would run {warmup} warm-up + {len(examples)} prompts x {repeats} measured repeats:")
        print(f"  {describe_command(settings)}")
        print("Dry run OK -- nothing executed.")
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
    for i in range(warmup):  # excluded from every statistic
        w = run_gguf_once(examples[0].prompt, settings)
        print(f"  warm-up {i + 1}/{warmup}: {w.status} {w.latency_ms:.0f} ms")

    measured, hashes = [], {}
    for ex in examples:
        for rep in range(repeats):
            run = run_gguf_once(ex.prompt, settings)
            print(f"  {ex.example_id} #{rep + 1}: {run.status} {run.latency_ms:.0f} ms")
            sql_hash = text_sha256(run.predicted_sql)
            if run.status == "ok":
                hashes.setdefault(ex.example_id, set()).add(sql_hash)
            measured.append(
                {
                    "example_id": ex.example_id,
                    "repeat": rep,
                    "status": run.status,
                    "latency_ms": round(run.latency_ms, 2),
                    "perf": run.perf,
                    "peak_rss_bytes": run.peak_rss_bytes,
                    "predicted_sql_sha256": sql_hash,
                    "error": run.error,
                }
            )

    doc = {
        "kind": "gguf_runtime_benchmark",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": deployment["base"],  # base GGUF (kept for compare tooling)
        "lora": deployment["lora"],
        "deployment": deployment,  # effective model: base + LoRA, incl. combined_size_bytes
        "settings": settings_dict(settings),
        "protocol": {"warmup_runs": warmup, "num_prompts": len(examples), "measured_repeats": repeats,
                     "example_ids": [e.example_id for e in examples]},  # fmt: skip
        "summary": summarize(measured, hashes),
        "measured_runs": measured,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    s = doc["summary"]
    print(f"Wrote {args.output}: {s['requests_ok']}/{s['requests_total']} ok, deterministic={s['deterministic_across_repeats']}")
    return 1 if s["requests_failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
