"""Contract tests for scripts/run_qlora_smoke.py.

No model/CUDA/network involved. Proves the smoke runner only ever reads
Phase 1's prepared train.jsonl (reusing prompt/completion/evidence-dropout
verbatim) and never touches BIRD Mini-Dev (external eval data) or the
`localsql.benchmark` package at all.
"""

import importlib.util
import json
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


def test_resolve_train_examples_input_override_bypasses_default_train_file(tmp_path):
    """Regression for a real Kaggle failure: `--input
    data/certification/longest_16.jsonl` still resolved
    `configs/train.yaml`'s `data.train_file` and failed with 'training
    file not found: .../data/processed/train.jsonl' -- proof --input was
    honored only by --token-profile, never by the actual training path.
    `resolve_train_examples` must use --input for BOTH modes and never
    touch cfg.data.train_file when --input is given."""
    module = _load_runner_module()

    override_path = tmp_path / "certification.jsonl"
    override_path.write_text(
        json.dumps(
            {
                "example_id": "cert:0000",
                "prompt": "p",
                "completion": "c",
                "db_id": "d",
                "representation": "canonical_unchanged",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    class _FakeDataCfg:
        # Deliberately a path that does NOT exist -- if resolve_train_examples
        # ever touches this when --input is given, the test fails loudly
        # (FileNotFoundError/BLOCKER) instead of silently succeeding.
        train_file = "data/processed/does_not_exist_train.jsonl"

    class _FakeCfg:
        data = _FakeDataCfg()

    examples, effective_path = module.resolve_train_examples(_FakeCfg(), tmp_path, override_path, limit=None)

    assert effective_path == override_path
    assert len(examples) == 1
    assert examples[0].example_id == "cert:0000"


def test_resolve_train_examples_missing_explicit_input_fails_on_that_path_not_default(tmp_path, capsys):
    """A typo'd/missing --input must BLOCKER naming the --input path
    itself, never silently falling back to (or complaining about) the
    default config.train_file."""
    import pytest

    module = _load_runner_module()
    missing_path = tmp_path / "does_not_exist_certification.jsonl"

    class _FakeDataCfg:
        train_file = "data/processed/train.jsonl"

    class _FakeCfg:
        data = _FakeDataCfg()

    with pytest.raises(SystemExit):
        module.resolve_train_examples(_FakeCfg(), tmp_path, missing_path, limit=None)

    captured = capsys.readouterr()
    assert str(missing_path) in captured.out
    assert "data/processed/train.jsonl" not in captured.out


def test_resolve_train_examples_without_input_preserves_default_behavior(tmp_path):
    """No --input given: behavior is byte-for-byte the same as the
    pre-fix load_examples path (Phase 1 train.jsonl, split=='train'
    filter) -- the fix must not change default-path behavior at all."""
    module = _load_runner_module()
    from tests.train.fixtures import make_prepared_example

    train_path = tmp_path / "train.jsonl"
    examples = [make_prepared_example(0), make_prepared_example(1, split="validation")]
    with train_path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(ex.model_dump_json() + "\n")

    class _FakeDataCfg:
        train_file = str(train_path.relative_to(tmp_path))

    class _FakeCfg:
        data = _FakeDataCfg()

    loaded, resolved_path = module.resolve_train_examples(_FakeCfg(), tmp_path, None, limit=None)

    assert len(loaded) == 1
    assert loaded[0].split == "train"
    assert resolved_path == train_path


def test_main_uses_resolve_train_examples_for_both_token_profile_and_training_paths():
    """Structural regression guard: main() must call resolve_train_examples
    for BOTH the --token-profile branch and the actual training/dry-run
    branch -- the original bug was exactly that only one branch honored
    --input. Guards against a future edit reintroducing a bare
    load_examples(cfg, REPO_ROOT, ...) call in main() that bypasses the
    override."""
    source = RUNNER_PATH.read_text(encoding="utf-8")
    main_body = source.split("def main() -> None:", 1)[1]
    assert main_body.count("resolve_train_examples(cfg, REPO_ROOT, args.input, args.max_train_examples)") == 2
    assert "load_examples(cfg, REPO_ROOT, args.max_train_examples)" not in main_body


def test_run_config_train_file_fields_come_from_resolved_train_path_parameter():
    """run_config.json's train_file/train_file_sha256 -- and therefore the
    durable checkpoint export's training-data snapshot, which reads them
    -- must be derived from the `train_path` parameter passed into
    run_smoke_training (the resolved effective path from
    resolve_train_examples), never re-derived from cfg.data.train_file."""
    source = RUNNER_PATH.read_text(encoding="utf-8")
    assert '"train_file": str(train_path),' in source
    assert '"train_file_sha256": file_sha256(train_path),' in source
    training_fn_source = source.split("def run_smoke_training(", 1)[1].split("\ndef run_verify_adapter", 1)[0]
    assert "cfg.data.train_file" not in training_fn_source


def test_write_adapter_verification_persists_standalone_json_file(tmp_path):
    """Regression for the missing-artifact bug: a real successful smoke run
    (qlora-smoke-1-alloc-retry) embedded `adapter_verification` correctly
    inside `summary.json` but never wrote the promised standalone
    `adapter_verification.json`. CPU/offline -- no GPU run needed to prove
    the file-writing behavior itself."""
    module = _load_runner_module()
    verification = {
        "resolved_revision": "cdbee75f17c01a7cc42f958dc650907174af0554",
        "adapter_active": True,
        "adapter_names": ["default"],
        "sample_example_id": "birdsql/bird23-train-filtered:00000",
        "sample_raw_completion": "SELECT T1.director_name FROM movies AS T1 WHERE T1.movie_title = 'x'",
    }

    out_path = module.write_adapter_verification(tmp_path, verification)

    assert out_path == tmp_path / "adapter_verification.json"
    assert out_path.exists()
    assert json.loads(out_path.read_text(encoding="utf-8")) == verification


def test_both_verification_call_sites_persist_the_standalone_artifact():
    """Guards against the bug recurring: both the post-training sanity
    check (inside run_smoke_training) and the standalone --verify-adapter
    mode must call the shared writer, not just embed the result into
    summary.json."""
    source = RUNNER_PATH.read_text(encoding="utf-8")
    # Match call sites ("... = write_adapter_verification(run_dir, ...)"),
    # not the `def write_adapter_verification(...)` line itself.
    assert source.count("= write_adapter_verification(run_dir") == 2


def test_main_parser_exposes_checkpoint_and_resume_flags():
    """Phase 5B Task 4: save_steps/save_total_limit/resume_from_checkpoint
    must be explicit CLI options, not hardcoded or environment-inferred."""
    source = RUNNER_PATH.read_text(encoding="utf-8")
    assert '"--save-steps"' in source
    assert '"--save-total-limit"' in source
    assert '"--resume-from-checkpoint"' in source
    # Wired into the actual training call, not just declared and dropped.
    assert "save_steps=args.save_steps" in source
    assert "save_total_limit=args.save_total_limit" in source
    assert "resume_from_checkpoint=args.resume_from_checkpoint" in source


def test_run_smoke_training_rejects_nonexistent_resume_checkpoint_path(tmp_path):
    """Explicit resume source only -- an invalid/typo'd path must BLOCKER,
    never silently fall back to training from scratch. This exits before
    any CUDA/model loading, so it is safe to run on this CPU-only machine.
    """
    import pytest

    module = _load_runner_module()
    cfg = load_train_config(REPO_ROOT / "configs" / "train.yaml")
    from tests.train.fixtures import make_prepared_example

    examples = [make_prepared_example(0)]
    run_dir = tmp_path / "run"
    train_path = tmp_path / "train.jsonl"
    train_path.write_text("{}", encoding="utf-8")

    with pytest.raises(SystemExit):
        module.run_smoke_training(
            cfg,
            examples,
            run_dir,
            train_path,
            max_steps=1,
            skip_verify=True,
            resume_from_checkpoint=str(tmp_path / "does_not_exist" / "checkpoint-2"),
        )


def test_run_smoke_training_refuses_to_overwrite_completed_run_even_with_resume(tmp_path):
    """A completed run's summary.json is never overwritten -- resuming
    into an already-finished --run-id must still BLOCKER, matching the
    documented RUN A / RUN B pattern (two distinct --run-ids)."""
    import pytest

    module = _load_runner_module()
    cfg = load_train_config(REPO_ROOT / "configs" / "train.yaml")
    from tests.train.fixtures import make_prepared_example

    examples = [make_prepared_example(0)]
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "summary.json").write_text("{}", encoding="utf-8")
    train_path = tmp_path / "train.jsonl"
    train_path.write_text("{}", encoding="utf-8")

    with pytest.raises(SystemExit):
        module.run_smoke_training(
            cfg, examples, run_dir, train_path, max_steps=1, skip_verify=True, resume_from_checkpoint=None
        )


def test_docstring_resume_example_uses_two_distinct_run_ids():
    """Regression: an earlier draft of the RUN A / RUN B docstring example
    reused the same --run-id for both commands, which the blocker above
    would refuse (a completed run's summary.json can't be resumed into)."""
    source = RUNNER_PATH.read_text(encoding="utf-8")
    assert "--run-id phase5b-resume-a" in source
    assert "--run-id phase5b-resume-b" in source


def test_summary_dict_construction_records_resume_and_no_silent_truncation():
    """Structural guard on the summary.json field names Phase 5B requires:
    parent-checkpoint provenance and an explicit no-truncation guarantee."""
    source = RUNNER_PATH.read_text(encoding="utf-8")
    assert '"resumed_from_checkpoint": train_result["resumed_from_checkpoint"]' in source
    assert '"starting_global_step": train_result["starting_global_step"]' in source
    assert '"checkpoint_dir": str(checkpoint_dir)' in source
    assert '"any_sequence_truncated": False' in source
    assert '"skipped_exceeds_max_seq_length_example_ids": skipped_over_length' in source


def test_main_parser_exposes_source_revision_flag_and_wires_it_through():
    """Phase 5B reliability correction Task 4: source revision must be an
    explicit CLI option (git archive/Kaggle upload strips .git) and must
    actually reach run_smoke_training, not just be parsed and dropped."""
    source = RUNNER_PATH.read_text(encoding="utf-8")
    assert '"--source-revision"' in source
    assert "source_revision=args.source_revision" in source


def test_summary_provenance_records_accelerate_trl_and_source_revision():
    """Phase 5B reliability correction Task 4: environment/source
    provenance fields must be represented in summary.json's provenance
    block, not just torch/transformers/peft/bitsandbytes."""
    source = RUNNER_PATH.read_text(encoding="utf-8")
    assert '"accelerate_version": info.accelerate_version' in source
    assert '"trl_version": info.trl_version' in source
    assert '"source_revision": source_revision_info["source_revision"]' in source
    assert '"source_revision_origin": source_revision_info["source_revision_origin"]' in source


def test_run_config_records_source_revision():
    """run_config.json must carry source_revision too (not just
    summary.json), so an export/resume can find it even before a run
    completes and writes a summary."""
    source = RUNNER_PATH.read_text(encoding="utf-8")
    assert '"source_revision": source_revision_info["source_revision"],' in source
    assert '"source_revision_origin": source_revision_info["source_revision_origin"],' in source


def test_run_token_profile_writes_real_token_length_manifest():
    """Phase 5B reliability correction Task 1: the canonical throughput
    manifest must be written from the real tokenizer profiling loop, using
    the certification module's writer -- never the character estimator."""
    source = RUNNER_PATH.read_text(encoding="utf-8")
    assert "from localsql.train.certification import write_token_length_manifest" in source
    assert "write_token_length_manifest(token_lengths_path, token_length_records)" in source
    assert "token_estimate" not in source
    assert "estimate_tokens" not in source


def test_load_arbitrary_jsonl_for_profiling_captures_db_id_and_representation(tmp_path):
    """Required so the token-length manifest can carry db_id/representation
    per example, per the Phase 5B reliability correction's manifest schema."""
    module = _load_runner_module()
    path = tmp_path / "candidate.jsonl"
    path.write_text(
        json.dumps(
            {
                "example_id": "x:0000",
                "prompt": "p",
                "completion": "c",
                "db_id": "shop_db",
                "representation": "compact_full_schema",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    examples, _ = module.load_arbitrary_jsonl_for_profiling(path, limit=None)

    assert examples[0].db_id == "shop_db"
    assert examples[0].representation == "compact_full_schema"


def test_real_phase1_train_file_loads_and_is_all_split_train():
    """Sanity check against the real committed Phase 1 output, if present
    locally (skips cleanly if not -- CI/dry environments need not have it)."""
    train_path = REPO_ROOT / "data" / "processed" / "train.jsonl"
    if not train_path.exists():
        return
    examples = load_prepared_examples(train_path)
    assert all(e.split == "train" for e in examples)
    assert all(e.completion.strip() for e in examples)
