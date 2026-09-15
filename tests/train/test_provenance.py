"""Contract tests for Phase 5B Task 4: source-revision provenance
resolution. CPU/offline, no network -- `git`, if invoked at all, is only
ever run against a synthetic tmp_path directory (never a network call).
"""

from pathlib import Path

from localsql.train.provenance import resolve_source_revision


def test_resolve_source_revision_explicit_arg_wins_over_everything(tmp_path):
    (tmp_path / "SOURCE_REVISION").write_text("file-value\n", encoding="utf-8")
    result = resolve_source_revision("explicit-value", tmp_path)
    assert result == {"source_revision": "explicit-value", "source_revision_origin": "explicit_arg"}


def test_resolve_source_revision_uses_provenance_file_when_no_explicit_value(tmp_path):
    (tmp_path / "SOURCE_REVISION").write_text("committed-from-file\n", encoding="utf-8")
    result = resolve_source_revision(None, tmp_path)
    assert result == {"source_revision": "committed-from-file", "source_revision_origin": "provenance_file"}


def test_resolve_source_revision_ignores_empty_provenance_file(tmp_path):
    (tmp_path / "SOURCE_REVISION").write_text("   \n", encoding="utf-8")
    result = resolve_source_revision(None, tmp_path)
    # Falls through to git (not a repo in tmp_path) -> unavailable.
    assert result["source_revision_origin"] in ("git_rev_parse", "unavailable")
    if result["source_revision_origin"] == "unavailable":
        assert result["source_revision"] is None


def test_resolve_source_revision_unavailable_when_nothing_present(tmp_path):
    result = resolve_source_revision(None, tmp_path)
    assert result["source_revision_origin"] in ("git_rev_parse", "unavailable")
    if result["source_revision_origin"] == "unavailable":
        assert result["source_revision"] is None


def test_resolve_source_revision_never_raises_when_git_missing_or_not_a_repo(tmp_path, monkeypatch):
    import subprocess

    def _boom(*args, **kwargs):
        raise FileNotFoundError("git not found")

    monkeypatch.setattr(subprocess, "run", _boom)
    result = resolve_source_revision(None, tmp_path)
    assert result == {"source_revision": None, "source_revision_origin": "unavailable"}


def test_resolve_source_revision_returns_dict_with_both_required_keys():
    result = resolve_source_revision("x", Path("."))
    assert set(result.keys()) == {"source_revision", "source_revision_origin"}
