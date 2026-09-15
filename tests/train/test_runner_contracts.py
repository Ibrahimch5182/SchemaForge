"""Contract tests for scripts/run_qlora_smoke.py.

No model/CUDA/network involved. Proves the smoke runner only ever reads
Phase 1's prepared train.jsonl (reusing prompt/completion/evidence-dropout
verbatim) and never touches BIRD Mini-Dev (external eval data) or the
`localsql.benchmark` package at all.
"""

import importlib.util
from pathlib import Path

from localsql.train.config import load_train_config
from localsql.train.sft_data import load_prepared_examples

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = REPO_ROOT / "scripts" / "run_qlora_smoke.py"

_FORBIDDEN_SYMBOLS = {"GradingExample", "load_gold_sql", "analyze_source_consistency"}


def _load_runner_module():
    spec = importlib.util.spec_from_file_location("run_qlora_smoke_under_test", RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runner_module_never_imports_benchmark_package():
    source = RUNNER_PATH.read_text(encoding="utf-8")
    assert "localsql.benchmark" not in source
    assert "bird_mini_dev" not in source


def test_runner_module_has_no_grading_symbols():
    module = _load_runner_module()
    leaked = _FORBIDDEN_SYMBOLS & set(dir(module))
    assert not leaked, f"run_qlora_smoke.py imports Mini-Dev/grading symbols: {leaked}"


def test_load_examples_only_reads_configured_train_file(tmp_path):
    module = _load_runner_module()
    cfg = load_train_config(REPO_ROOT / "configs" / "train.yaml")
    assert cfg.data.train_file == "data/processed/train.jsonl"  # Phase 1 artifact, not Mini-Dev

    train_path = tmp_path / "train.jsonl"
    from tests.train.fixtures import make_prepared_example

    examples = [make_prepared_example(0), make_prepared_example(1, split="validation")]
    with train_path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(ex.model_dump_json() + "\n")

    class _FakeDataCfg:
        train_file = str(train_path.relative_to(tmp_path))

    class _FakeCfg:
        data = _FakeDataCfg()

    loaded, resolved_path = module.load_examples(_FakeCfg(), tmp_path, limit=None)
    # split == "validation" rows must be filtered out -- training only trains on split == "train".
    assert len(loaded) == 1
    assert loaded[0].split == "train"
    assert resolved_path == train_path


def test_real_phase1_train_file_loads_and_is_all_split_train():
    """Sanity check against the real committed Phase 1 output, if present
    locally (skips cleanly if not -- CI/dry environments need not have it)."""
    train_path = REPO_ROOT / "data" / "processed" / "train.jsonl"
    if not train_path.exists():
        return
    examples = load_prepared_examples(train_path)
    assert all(e.split == "train" for e in examples)
    assert all(e.completion.strip() for e in examples)
