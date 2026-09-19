"""Write a `SHA256SUMS` next to the Q4_K_M base + LoRA GGUF so the cloud host (and the
model-server container) can verify the transferred artifacts.

    uv run python scripts/phase11_model_manifest.py --base PATH/base-Q4_K_M.gguf --lora PATH/lora-1518-f16.gguf --out-dir DEPLOY_DIR

Copies nothing and writes only the manifest (plain `sha256sum -c` format, file names relative to
`--out-dir`, so both files must be placed in the same directory on the host). Prints sizes and the
expected LoRA hash pinned in configs/phase7.yaml when the LoRA is the frozen checkpoint-1518 adapter.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--lora", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True, help="directory that will hold both GGUFs on the host (SHA256SUMS is written here)")
    args = ap.parse_args()
    lines, total = [], 0
    for p in (args.base, args.lora):
        if not p.is_file():
            print(f"missing file: {p.name}", file=sys.stderr)
            return 2
        digest = sha256(p)
        total += p.stat().st_size
        lines.append(f"{digest}  {p.name}")
        print(f"{digest}  {p.name}  ({p.stat().st_size / 1e6:.1f} MB)")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"combined size: {total / 1e9:.2f} GB; wrote {args.out_dir / 'SHA256SUMS'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
