"""Contract tests for Phase 5B Task 4: source-revision provenance
resolution. CPU/offline, no network -- `git`, if invoked at all, is only
ever run against a synthetic tmp_path directory (never a network call).
"""

from pathlib import Path

from localsql.train.provenance import resolve_run_source_revision, resolve_source_revision


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


# --- resolve_run_source_revision (durable-export provenance fix) ---


def test_resolve_run_source_revision_propagates_run_config_recorded_revision(tmp_path):
    """Regression: a certified run (data/runs/phase5b-resume-cert-v2)
    recorded source_revision=78b3dab4de70473b9afcab56d0156adb6daa7ab1 /
    source_revision_origin=explicit_arg in run_config.json at training
    time, but export reported 'unavailable' -- it must instead propagate
    the run's own recorded value."""
    run_config = {
        "source_revision": "78b3dab4de70473b9afcab56d0156adb6daa7ab1",
        "source_revision_origin": "explicit_arg",
    }
    result = resolve_run_source_revision(run_config, summary=None, explicit=None, repo_root=tmp_path)
    assert result == {
        "source_revision": "78b3dab4de70473b9afcab56d0156adb6daa7ab1",
        "source_revision_origin": "explicit_arg",
    }


def test_resolve_run_source_revision_run_config_takes_precedence_over_summary_and_fallback(tmp_path):
    """run_config's recorded revision wins even when summary.json has a
    different one AND an explicit --source-revision is passed at export
    time -- a recorded run revision is never replaced by recomputing from
    the export environment."""
    run_config = {"source_revision": "run-config-revision", "source_revision_origin": "explicit_arg"}
    summary = {"provenance": {"source_revision": "summary-revision", "source_revision_origin": "git_rev_parse"}}
    result = resolve_run_source_revision(run_config, summary, explicit="export-time-arg", repo_root=tmp_path)
    assert result == {"source_revision": "run-config-revision", "source_revision_origin": "explicit_arg"}


def test_resolve_run_source_revision_falls_back_to_summary_when_run_config_lacks_one(tmp_path):
    run_config = {}  # no source_revision recorded (e.g. an older run predating this field)
    summary = {"provenance": {"source_revision": "summary-revision", "source_revision_origin": "git_rev_parse"}}
    result = resolve_run_source_revision(run_config, summary, explicit=None, repo_root=tmp_path)
    assert result == {"source_revision": "summary-revision", "source_revision_origin": "git_rev_parse"}


def test_resolve_run_source_revision_falls_back_to_existing_resolution_when_neither_artifact_has_one(tmp_path):
    run_config = {}
    summary = {"provenance": {}}
    result = resolve_run_source_revision(run_config, summary, explicit="export-time-arg", repo_root=tmp_path)
    assert result == {"source_revision": "export-time-arg", "source_revision_origin": "explicit_arg"}


def test_resolve_run_source_revision_falls_back_when_summary_is_none_and_run_config_empty(tmp_path):
    result = resolve_run_source_revision({}, summary=None, explicit="export-time-arg", repo_root=tmp_path)
    assert result == {"source_revision": "export-time-arg", "source_revision_origin": "explicit_arg"}


def test_resolve_run_source_revision_defaults_origin_when_run_config_has_revision_but_no_origin(tmp_path):
    run_config = {"source_revision": "rev-only-no-origin-field"}
    result = resolve_run_source_revision(run_config, summary=None, explicit=None, repo_root=tmp_path)
    assert result == {"source_revision": "rev-only-no-origin-field", "source_revision_origin": "run_config"}


def test_resolve_run_source_revision_defaults_origin_when_summary_has_revision_but_no_origin(tmp_path):
    run_config = {}
    summary = {"provenance": {"source_revision": "rev-only-no-origin-field"}}
    result = resolve_run_source_revision(run_config, summary, explicit=None, repo_root=tmp_path)
    assert result == {"source_revision": "rev-only-no-origin-field", "source_revision_origin": "summary_provenance"}


def test_resolve_run_source_revision_never_mutates_run_config_or_summary(tmp_path):
    run_config = {"source_revision": "abc", "source_revision_origin": "explicit_arg"}
    summary = {"provenance": {"source_revision": "xyz"}}
    run_config_before = dict(run_config)
    summary_before = {"provenance": dict(summary["provenance"])}

    resolve_run_source_revision(run_config, summary, explicit="something-else", repo_root=tmp_path)

    assert run_config == run_config_before
    assert summary == summary_before
