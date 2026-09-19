import sqlite3
import time

import pytest

from localsql.backend.errors import DatabaseUnavailableError, ExecutionError
from localsql.backend.executor import SQLiteReadOnlyExecutor
from localsql.backend.registry import RegisteredDatabase
from localsql.backend.sqlite_conn import open_readonly

from tests.backend.helpers import file_sha, make_registry


@pytest.fixture
def db(tmp_path):
    return make_registry(tmp_path).resolve("demo")


def count(db, table="employees"):
    conn = sqlite3.connect(db.path)
    try:
        return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    finally:
        conn.close()


def test_select_returns_columns_rows_and_metadata(db):
    res = SQLiteReadOnlyExecutor(5, 100).execute(db, "SELECT emp_id, name FROM employees ORDER BY emp_id LIMIT 3")
    assert res.columns == ["emp_id", "name"]
    assert res.rows == [[1, "Ada"], [2, "Grace"], [3, "Linus"]]
    assert res.returned_row_count == 3 and res.truncated is False and res.max_rows == 100
    assert res.elapsed_ms >= 0
    assert not hasattr(res, "total_rows")  # no total count is claimed


def test_json_safe_values(db):
    res = SQLiteReadOnlyExecutor(5, 10).execute(
        db, "SELECT NULL, 1, 2.5, 'x', x'DEADBEEF', 1e999, -1e999, 9223372036854775807"
    )
    import json

    row = res.rows[0]
    assert row[:4] == [None, 1, 2.5, "x"] and row[4] == "<blob:4 bytes>"
    assert row[5] == "inf" and row[6] == "-inf" and row[7] == 9223372036854775807
    json.dumps(res.model_dump(mode="json"), allow_nan=False)  # strictly JSON-serializable


def test_truncation_is_deterministic_with_max_rows_plus_one(db):
    ex = lambda n, sql="SELECT emp_id FROM employees ORDER BY emp_id": SQLiteReadOnlyExecutor(5, n).execute(db, sql)  # noqa: E731
    exact = ex(12)  # exactly 12 rows exist
    assert (exact.returned_row_count, exact.truncated) == (12, False)
    over = ex(11)
    assert (over.returned_row_count, over.truncated) == (11, True) and over.rows[-1] == [11]
    assert ex(11).rows == over.rows  # repeatable
    one = ex(1)
    assert one.rows == [[1]] and one.truncated is True


def test_timeout_interrupts_runaway_query(db):
    runaway = "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT COUNT(*) FROM c"
    start = time.monotonic()
    with pytest.raises(ExecutionError) as e:
        SQLiteReadOnlyExecutor(0.3, 10).execute(db, runaway)
    assert e.value.code == "timeout" and time.monotonic() - start < 10


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM employees",
        "UPDATE employees SET salary = 0",
        "INSERT INTO departments VALUES (99, 'x')",
        "DROP TABLE employees",
        "CREATE TABLE evil (a INT)",
        "ALTER TABLE employees ADD COLUMN z INT",
        "PRAGMA writable_schema = 1",
        "PRAGMA journal_mode = DELETE",
        "ATTACH DATABASE ':memory:' AS m",
        "VACUUM",
        "BEGIN",
        "SAVEPOINT s",
        "REINDEX",
        "SELECT load_extension('nope')",
        "SELECT * FROM pragma_table_info('employees')",
    ],
)
def test_executor_independently_blocks_everything_but_reads(db, sql):
    before = file_sha(db.path)
    with pytest.raises(ExecutionError):
        SQLiteReadOnlyExecutor(5, 10).execute(db, sql)
    assert file_sha(db.path) == before
    assert count(db) == 12 and count(db, "departments") == 3


def test_connection_level_read_only_without_authorizer(db):
    """mode=ro + query_only alone (no authorizer) already refuses writes."""
    conn = open_readonly(db.path)
    try:
        for sql in ["DELETE FROM employees", "INSERT INTO departments VALUES (9,'x')", "CREATE TABLE z(a)"]:
            with pytest.raises(sqlite3.OperationalError):
                conn.execute(sql)
    finally:
        conn.close()
    assert count(db) == 12


def test_multiple_statements_not_executed(db):
    with pytest.raises(ExecutionError) as e:
        SQLiteReadOnlyExecutor(5, 10).execute(db, "SELECT 1; DELETE FROM employees")
    assert e.value.code == "sql_error" and count(db) == 12


def test_sql_errors_are_structured_and_path_free(db):
    with pytest.raises(ExecutionError) as e:
        SQLiteReadOnlyExecutor(5, 10).execute(db, "SELECT nope FROM employees")
    assert e.value.code == "sql_error" and "nope" in e.value.message
    assert str(db.path.parent) not in e.value.message


def test_connection_is_closed_even_on_error(db):
    ex = SQLiteReadOnlyExecutor(5, 10)
    for sql in ["SELECT nope", "SELECT 1"]:
        try:
            ex.execute(db, sql)
        except ExecutionError:
            pass
    db.path.unlink()  # on Windows this fails if any handle leaked
    assert not db.path.exists()


def test_missing_database_file_is_unavailable(tmp_path):
    ghost = RegisteredDatabase("g", tmp_path / "ghost.sqlite", "sqlite")
    with pytest.raises((DatabaseUnavailableError, ExecutionError)):
        SQLiteReadOnlyExecutor(5, 10).execute(ghost, "SELECT 1")
    assert not (tmp_path / "ghost.sqlite").exists()  # mode=ro never creates files


def test_invalid_limits_rejected():
    with pytest.raises(ValueError):
        SQLiteReadOnlyExecutor(0, 10)
    with pytest.raises(ValueError):
        SQLiteReadOnlyExecutor(1, 0)
