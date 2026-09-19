"""Phase 7 GGUF conversion orchestration (llama.cpp tools supplied by path).

Subcommands: base | lora | quantize (primary path) | merge (OPTIONAL, needs llama-export-lora)
Each real run: preflight, run the tool (no shell), fail on nonzero exit,
write to <name>.partial.gguf then rename, record command + input/output
SHA256 + size in .artifacts/phase7/manifests/stage-<name>.json. A rerun that
would replace a different artifact is refused.

--dry-run prints the exact command(s) and preflight problems; runs nothing.
Flags of external tools are not assumed stable: check them with the tool's
--help and use --extra-arg to adapt.

    uv run python scripts/convert_phase7.py base --dry-run `
        --convert-script C:\\llama.cpp\\convert_hf_to_gguf.py --hf-dir D:\\models\\qwen3-4b
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from localsql.deploy import convert  # noqa: E402
from localsql.deploy.config import load_phase7_config  # noqa: E402
from localsql.deploy.manifest import default_artifact_paths, phase7_root  # noqa: E402
from localsql.deploy.provenance import ProvenanceError, validate_adapter, validate_base_snapshot  # noqa: E402


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "phase7.yaml")
    p.add_argument("--dry-run", action="store_true", help="Print commands; run nothing, write nothing.")
    p.add_argument("--extra-arg", action="append", default=[], help="Extra argument for the external tool (repeatable).")
    p.add_argument("--output", type=Path, default=None, help="Override the default output path.")
    p.add_argument("--timeout", type=float, default=None, help="Subprocess timeout in seconds (default: none).")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="stage", required=True)

    b = sub.add_parser("base", help="HF base -> base GGUF")
    _add_common(b)
    b.add_argument("--convert-script", type=Path, required=True, help="llama.cpp convert_hf_to_gguf.py")
    b.add_argument("--hf-dir", type=Path, required=True, help="Local HF snapshot of the frozen base revision.")
    b.add_argument("--python-exe", type=str, default=sys.executable)

    lo = sub.add_parser("lora", help="PEFT LoRA -> LoRA GGUF")
    _add_common(lo)
    lo.add_argument("--convert-script", type=Path, required=True, help="llama.cpp convert_lora_to_gguf.py")
    lo.add_argument("--hf-dir", type=Path, required=True)
    lo.add_argument("--adapter-dir", type=Path, default=None)
    lo.add_argument("--python-exe", type=str, default=sys.executable)

    m = sub.add_parser("merge", help="OPTIONAL: base GGUF + LoRA GGUF -> merged GGUF (not required)")
    _add_common(m)
    m.add_argument("--export-lora-exe", type=Path, required=True, help="llama.cpp llama-export-lora")
    m.add_argument("--base-gguf", type=Path, default=None, help="Default: base F16 GGUF.")
    m.add_argument("--lora-gguf", type=Path, default=None, help="Default: LoRA F16 GGUF.")

    q = sub.add_parser("quantize", help="base F16 GGUF -> base Q4_K_M GGUF")
    _add_common(q)
    q.add_argument("--quantize-exe", type=Path, required=True, help="llama.cpp llama-quantize")
    q.add_argument("--input-gguf", type=Path, default=None, help="Default: the base F16 GGUF.")
    return parser


def make_plan(args, cfg, paths: dict) -> convert.StagePlan:
    extra = args.extra_arg
    outtype = cfg.quantization.intermediate_outtype
    if args.stage == "base":
        return convert.plan_base_gguf(args.python_exe, args.convert_script, args.hf_dir,
                                      args.output or Path(paths["base_f16_gguf"]), outtype, extra)  # fmt: skip
    if args.stage == "lora":
        adapter_dir = args.adapter_dir or (REPO_ROOT / cfg.adapter.default_dir)
        return convert.plan_lora_gguf(args.python_exe, args.convert_script, adapter_dir, args.hf_dir,
                                      args.output or Path(paths["lora_f16_gguf"]), outtype, extra)  # fmt: skip
    if args.stage == "merge":
        return convert.plan_merge(args.export_lora_exe, args.base_gguf or Path(paths["base_f16_gguf"]),
                                  args.lora_gguf or Path(paths["lora_f16_gguf"]),
                                  args.output or Path(paths["optional_merged_f16_gguf"]), extra)  # fmt: skip
    return convert.plan_quantize(args.quantize_exe, args.input_gguf or Path(paths["base_f16_gguf"]),
                                 args.output or Path(paths["base_q4_k_m_gguf"]), cfg.quantization.target, extra)  # fmt: skip


def frozen_input_checks(args, cfg) -> dict:
    """Fail-closed provenance checks for stages that consume frozen inputs."""
    info: dict = {}
    if args.stage in ("base", "lora"):
        info["base_snapshot"] = validate_base_snapshot(args.hf_dir, cfg)
    if args.stage == "lora":
        adapter_dir = args.adapter_dir or (REPO_ROOT / cfg.adapter.default_dir)
        info["adapter"] = validate_adapter(adapter_dir, cfg).to_dict()
    return info


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_phase7_config(args.config)
    root = phase7_root(REPO_ROOT, cfg)
    plan = make_plan(args, cfg, default_artifact_paths(root, cfg))

    if args.dry_run:
        report = convert.dry_run_report(plan)
        try:
            frozen_input_checks(args, cfg)
            report["frozen_input_checks"] = "passed"
        except ProvenanceError as e:
            report["frozen_input_checks"] = f"FAILED: {e}"
        print(json.dumps(report, indent=2))
        print("Dry run OK -- nothing executed.")
        return 0

    try:
        info = frozen_input_checks(args, cfg)
        record = convert.execute_stage(plan, root / "manifests", timeout=args.timeout)
    except (ProvenanceError, convert.StageError) as e:
        print(f"BLOCKER: {e}")
        return 1
    if info and record["status"] == "completed":
        rec_path = convert.record_path(root / "manifests", plan)
        record = {**record, "frozen_inputs": info}
        rec_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"Stage '{plan.name}' {record['status']}: {record['output']['path']}")
    print(f"  size={record['output']['size_bytes']} sha256={record['output']['sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
