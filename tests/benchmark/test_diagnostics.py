from localsql.benchmark.diagnostics import EXEC_ERROR, EXEC_OK, PARSE_FAILED, PARSE_OK, diagnose_prediction

from tests.benchmark.fixtures import write_fixture_workspace


def test_parseable_and_executable_valid_sql(tmp_path):
    ws = write_fixture_workspace(tmp_path)
    diag = diagnose_prediction("ex-0", "SELECT COUNT(*) FROM customers", ws["db_path"])
    assert diag.parse_status == PARSE_OK
    assert diag.execution_status == EXEC_OK
    assert diag.runtime_ms is not None


def test_unparseable_sql_is_flagged():
    from localsql.benchmark.diagnostics import check_parse

    status, err_type, err_msg = check_parse("SELECT FROM WHERE")
    assert status == PARSE_FAILED
    assert err_type is not None


def test_parseable_but_not_executable_against_wrong_schema(tmp_path):
    ws = write_fixture_workspace(tmp_path)
    # Valid SQL syntax, but references a column that doesn't exist --
    # demonstrates parseable != executable.
    diag = diagnose_prediction("ex-1", "SELECT nonexistent_column FROM customers", ws["db_path"])
    assert diag.parse_status == PARSE_OK
    assert diag.execution_status == EXEC_ERROR
    assert diag.error_message is not None


def test_executable_result_is_not_a_correctness_claim(tmp_path):
    ws = write_fixture_workspace(tmp_path)
    # This executes successfully but returns the WRONG answer (no WHERE
    # clause) -- diagnostics must not claim correctness, only executability.
    diag = diagnose_prediction("ex-2", "SELECT COUNT(*) FROM customers", ws["db_path"])
    assert diag.execution_status == EXEC_OK
    assert not hasattr(diag, "official_correct")
