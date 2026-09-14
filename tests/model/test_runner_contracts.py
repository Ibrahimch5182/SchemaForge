"""Contract tests for scripts/run_baseline.py and the generation pipeline.

No model/CUDA/network involved -- these test the pure-Python plumbing:
manifest parsing, gold isolation, prediction-contract conformance, and
deterministic ordering.
"""

import importlib.util
import json
from pathlib import Path

from localsql.benchmark.models import GenerationExample
from localsql.benchmark.prediction import validate_predictions
from localsql.model.generation import generate_one_example

from tests.model.fixtures import FakeBackend, make_example

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_BASELINE_PATH = REPO_ROOT / "scripts" / "run_baseline.py"

# Symbols that would indicate the runner can see gold/evaluator data.
_FORBIDDEN_SYMBOLS = {"GradingExample", "load_gold_sql", "analyze_source_consistency", "GRADING"}


def _load_run_baseline_module():
    spec = importlib.util.spec_from_file_location("run_baseline_under_test", RUN_BASELINE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runner_module_has_no_gold_or_grading_symbols():
    module = _load_run_baseline_module()
    leaked = _FORBIDDEN_SYMBOLS & set(dir(module))
    assert not leaked, f"run_baseline.py imports gold/grading symbols: {leaked}"


def test_load_manifest_parses_generation_examples_only(tmp_path):
    module = _load_run_baseline_module()
    manifest_path = tmp_path / "manifest.jsonl"
    examples = [make_example(0), make_example(1)]
    with manifest_path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(ex.model_dump_json() + "\n")

    loaded = module.load_manifest(manifest_path)
    assert len(loaded) == 2
    assert all(isinstance(ex, GenerationExample) for ex in loaded)
    assert [ex.example_id for ex in loaded] == [ex.example_id for ex in examples]


def test_load_manifest_rejects_a_row_with_gold_sql_field(tmp_path):
    module = _load_run_baseline_module()
    manifest_path = tmp_path / "manifest.jsonl"
    bad_row = {**json.loads(make_example(0).model_dump_json()), "sql": "SELECT * FROM secret_gold"}
    manifest_path.write_text(json.dumps(bad_row) + "\n", encoding="utf-8")

    try:
        module.load_manifest(manifest_path)
    except ValueError as e:
        assert "gold-free" in str(e) or "gold" in str(e).lower()
    else:
        raise AssertionError("expected a row containing 'sql' to be rejected")


def test_generation_ordering_matches_manifest_order():
    manifest = [make_example(i) for i in range(5)]
    backend = FakeBackend()
    for ex in manifest:
        generate_one_example(ex, backend, model_revision="abc", context_mode="with_business_context")

    assert backend.calls == [ex.prompt for ex in manifest]


def test_generation_records_conform_to_phase2_prediction_contract():
    manifest = [make_example(i) for i in range(3)]
    backend = FakeBackend()
    records = [
        generate_one_example(ex, backend, model_revision="abc", context_mode="with_business_context")
        for ex in manifest
    ]
    raw_predictions = [r.to_prediction_dict() for r in records]

    validation = validate_predictions(raw_predictions, manifest)
    assert validation.is_complete
    assert len(validation.valid_records) == len(manifest)
