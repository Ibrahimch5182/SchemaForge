from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "benchmark.yaml"


def _load():
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def test_locked_benchmark_variant():
    config = _load()
    assert config["benchmark_id"] == "bird_mini_dev_500"
    assert config["dialect"] == "sqlite"
    assert config["expected_examples"] == 500
    assert config["expected_databases"] == 11


def test_excludes_livesqlbench_v2_variant():
    config = _load()
    excluded = config["excluded_variant"]
    assert "livesqlbench" in excluded["name"]


def test_metrics_configuration():
    config = _load()
    assert config["metrics"]["primary"] == "execution_accuracy"
    assert config["metrics"]["secondary"] == "soft_f1"
    assert config["metrics"]["r_ves_deferred"] is True


def test_official_evaluator_revision_is_pinned():
    config = _load()
    revision = config["source"]["official_evaluator"]["revision"]
    assert len(revision) == 40  # full git commit SHA, not a floating branch name
