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
