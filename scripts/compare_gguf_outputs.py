"""Quantization regression/sanity comparison of two `run_local_gguf.py`
result files (F16 base + LoRA vs Q4_K_M base + the SAME LoRA). Output agreement on a
small fixed sample -- NOT an accuracy benchmark.

    uv run python scripts/compare_gguf_outputs.py `
        --reference .artifacts\\phase7\\benchmarks\\sanity-f16-base-lora.json `
        --candidate .artifacts\\phase7\\benchmarks\\sanity-q4_k_m-base-lora.json `
        --output .artifacts\\phase7\\benchmarks\\quant-sanity-comparison.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from localsql.deploy.sanity import compare_generations  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    ref = json.loads(args.reference.read_text(encoding="utf-8"))
    cand = json.loads(args.candidate.read_text(encoding="utf-8"))
    try:
        report = compare_generations(ref, cand)
    except ValueError as e:
        print(f"BLOCKER: {e}")
        return 1
    print(
        f"Exact normalized agreement: {report['exact_normalized_agreement']}/{report['n']} "
        f"({len(report['changed_examples'])} changed)"
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
