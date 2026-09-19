"""Phase 7 preparation: validate the frozen checkpoint-1518 adapter and base
model contract, create .artifacts/phase7/, and emit phase7_manifest.json.

--dry-run validates and prints the manifest but creates NO directories/files.
Never downloads a model and never converts anything.

    uv run python scripts/prepare_phase7.py --dry-run
    uv run python scripts/prepare_phase7.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from localsql.deploy.config import load_phase7_config  # noqa: E402
from localsql.deploy.manifest import ManifestConflictError, build_manifest, phase7_root, write_manifest  # noqa: E402
from localsql.deploy.provenance import ProvenanceError, validate_adapter  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "phase7.yaml")
    parser.add_argument("--adapter-dir", type=Path, default=None, help="Default: adapter.default_dir from the config.")
    parser.add_argument("--dry-run", action="store_true", help="Validate + print; write nothing.")
    args = parser.parse_args()

    cfg = load_phase7_config(args.config)
    adapter_dir = args.adapter_dir or (REPO_ROOT / cfg.adapter.default_dir)
    try:
        adapter = validate_adapter(adapter_dir, cfg)
    except ProvenanceError as e:
        print(f"BLOCKER: {e}")
        return 1
    print(f"Adapter OK: {cfg.adapter.checkpoint} sha256={adapter.weights_sha256} r={adapter.r} alpha={adapter.lora_alpha}")
    print(f"Base model (frozen): {cfg.model.id} @ {cfg.model.revision}")

    root = phase7_root(REPO_ROOT, cfg)
    manifest = build_manifest(cfg, adapter, root)
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        print(f"Dry run OK -- would create {root} and write manifests/phase7_manifest.json. Nothing written.")
        return 0

    try:
        path = write_manifest(root, cfg, manifest)
    except ManifestConflictError as e:
        print(f"BLOCKER: {e}")
        return 1
    print(f"Manifest: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
