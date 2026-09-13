import json

from localsql.benchmark.official_adapter import MISSING_PREDICTION_SQL, build_diff_jsonl, build_official_prediction_file


def test_official_prediction_file_matches_expected_format(tmp_path):
    out_path = tmp_path / "pred.json"
    missing = build_official_prediction_file(
        predictions_by_id={"a": "SELECT 1", "b": "SELECT 2"},
        grading_ids_in_order=["a", "b"],
        grading_db_ids_in_order=["db1", "db2"],
        out_path=out_path,
    )
    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert list(data.keys()) == ["0", "1"]
    assert data["0"] == "SELECT 1\t----- bird -----\tdb1"
    assert data["1"] == "SELECT 2\t----- bird -----\tdb2"
    assert missing == []


def test_missing_predictions_get_sentinel_and_are_reported(tmp_path):
    out_path = tmp_path / "pred.json"
    missing = build_official_prediction_file(
        predictions_by_id={"a": "SELECT 1"},
        grading_ids_in_order=["a", "b"],
        grading_db_ids_in_order=["db1", "db2"],
        out_path=out_path,
    )
    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert missing == ["b"]
    assert MISSING_PREDICTION_SQL in data["1"]
    # Alignment is preserved -- index "1" still maps to db2, not shifted.
    assert data["1"].endswith("db2")


def test_diff_jsonl_preserves_order(tmp_path):
    out_path = tmp_path / "diff.jsonl"
    build_diff_jsonl(["simple", "moderate", None], out_path)
    lines = [json.loads(l) for l in out_path.read_text(encoding="utf-8").splitlines()]
    assert [l["difficulty"] for l in lines] == ["simple", "moderate", None]
