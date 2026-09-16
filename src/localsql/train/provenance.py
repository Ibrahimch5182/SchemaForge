"""Phase 5B source-revision provenance resolution.

`git archive` (and similar Kaggle-upload paths) strips `.git`, so a bare
`git rev-parse HEAD` silently loses the source commit on exactly the
machine this matters most for (an ephemeral Kaggle VM running a durable
checkpoint export). This module resolves a source revision from an
explicit override first, then a provenance file, then falls back to a
best-effort `git rev-parse` -- and returns which path was used, rather
than silently guessing. No network access is used or required.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

PROVENANCE_FILE_NAME = "SOURCE_REVISION"


def resolve_source_revision(explicit: str | None, repo_root: Path) -> dict:
    """Resolve a source-code revision string, in priority order:

    1. `explicit` (e.g. a `--source-revision` CLI flag) -- always wins.
       This is the escape hatch for environments (Kaggle uploads, `git
       archive` extracts) where `.git` is not present but the caller
       still knows the commit.
    2. A `SOURCE_REVISION` file at `repo_root` (first non-empty line) --
       for writing the value once, before packaging, in an environment
       that will later lose `.git`.
    3. `git rev-parse HEAD`, best-effort (silently returns unavailable if
       not a git repository, git isn't installed, or the command fails).

    Returns `{"source_revision": str | None, "source_revision_origin":
    "explicit_arg" | "provenance_file" | "git_rev_parse" | "unavailable"}`
    -- the value is never silently dropped without recording how (or
    whether) it was found.
    """
    if explicit:
        return {"source_revision": explicit, "source_revision_origin": "explicit_arg"}

    provenance_path = repo_root / PROVENANCE_FILE_NAME
    if provenance_path.exists():
        value = provenance_path.read_text(encoding="utf-8").strip()
        if value:
            return {"source_revision": value, "source_revision_origin": "provenance_file"}

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            value = result.stdout.strip()
            if value:
                return {"source_revision": value, "source_revision_origin": "git_rev_parse"}
    except (OSError, subprocess.SubprocessError):
        pass

    return {"source_revision": None, "source_revision_origin": "unavailable"}


def resolve_run_source_revision(
    run_config: dict, summary: dict | None, explicit: str | None, repo_root: Path
) -> dict:
    """Resolve the source revision to record for a durable export's RUN
    provenance -- i.e. what identifies the source code state that
    actually produced this checkpoint, as opposed to whatever the export
    ENVIRONMENT happens to resolve to right now.

    A run's `run_config.json` (and `summary.json`) already recorded a
    `source_revision` at TRAINING time (see `resolve_source_revision`,
    called from `scripts/run_qlora_smoke.py`) -- that recorded value must
    never be silently replaced by re-resolving from the export
    environment (which may be a different Kaggle session, a fresh clone
    missing `.git`, or simply invoked without `--source-revision`).

    Precedence:
    1. `run_config["source_revision"]`, if present and truthy -- the
       run's own recorded value, with its own recorded
       `source_revision_origin` (defaulting to `"run_config"` if that
       field is somehow missing).
    2. `summary["provenance"]["source_revision"]`, if `run_config` lacks
       one -- same reasoning, one artifact removed.
    3. `resolve_source_revision(explicit, repo_root)` (the existing
       explicit-arg / provenance-file / git-rev-parse fallback) -- used
       ONLY when NEITHER run artifact recorded a revision at all.

    Never mutates `run_config` or `summary`.
    """
    run_config_revision = run_config.get("source_revision")
    if run_config_revision:
        return {
            "source_revision": run_config_revision,
            "source_revision_origin": run_config.get("source_revision_origin") or "run_config",
        }

    summary_provenance = (summary or {}).get("provenance") or {}
    summary_revision = summary_provenance.get("source_revision")
    if summary_revision:
        return {
            "source_revision": summary_revision,
            "source_revision_origin": summary_provenance.get("source_revision_origin") or "summary_provenance",
        }

    return resolve_source_revision(explicit, repo_root)
