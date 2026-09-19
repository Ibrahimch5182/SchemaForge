import pytest

from localsql.backend.safety import SQLSafetyPolicy

policy = SQLSafetyPolicy()

ALLOWED = [
    "SELECT 1",
    "select * from employees;",
    "SELECT name FROM employees WHERE salary > 100 ORDER BY name LIMIT 5",
    "WITH eng AS (SELECT * FROM employees WHERE dept_id = 1) SELECT COUNT(*) FROM eng",
    "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c WHERE x < 5) SELECT x FROM c",
    "SELECT a FROM t UNION SELECT b FROM u",
    "(SELECT 1) UNION ALL (SELECT 2)",
    "SELECT e.name, d.name FROM employees e JOIN departments d ON e.dept_id = d.dept_id",
    "SELECT * FROM (SELECT dept_id, AVG(salary) s FROM employees GROUP BY dept_id) WHERE s > 1",
    "SELECT 1 -- trailing comment",
    "SELECT 1; -- comment after semicolon",
    "SELECT 'a;b; DROP TABLE x' AS literal",  # semicolons/keywords inside string literals are fine
    "SELECT 1 /* DROP TABLE x */",
    "SELECT length(name), substr(name, 1, 2), CAST(salary AS INTEGER) FROM employees",
]


@pytest.mark.parametrize("sql", ALLOWED)
def test_read_only_queries_allowed(sql):
    d = policy.check(sql)
    assert d.allowed, d.reasons


REJECTED = [
    ("INSERT INTO t VALUES (1)", "not_read_only_query"),
    ("UPDATE t SET a = 1", "not_read_only_query"),
    ("DELETE FROM t", "not_read_only_query"),
    ("REPLACE INTO t VALUES (1)", "not_read_only_query"),
    ("DROP TABLE t", "not_read_only_query"),
    ("CREATE TABLE x (a INT)", "not_read_only_query"),
    ("CREATE TABLE x AS SELECT 1", "not_read_only_query"),
    ("ALTER TABLE t ADD COLUMN c INT", "not_read_only_query"),
    ("ATTACH DATABASE 'x.db' AS other", "not_read_only_query"),
    ("DETACH other", "not_read_only_query"),
    ("PRAGMA writable_schema = 1", "not_read_only_query"),
    ("PRAGMA table_info(t)", "not_read_only_query"),
    ("BEGIN", "not_read_only_query"),
    ("COMMIT", "not_read_only_query"),
    ("ROLLBACK", "not_read_only_query"),
    ("SAVEPOINT s", "not_read_only_query"),
    ("VACUUM", "not_read_only_query"),
    ("REINDEX", "not_read_only_query"),
    ("ANALYZE", "not_read_only_query"),
    ("EXPLAIN SELECT 1", "not_read_only_query"),
    ("VALUES (1)", "not_read_only_query"),
    ("SELECT 1; SELECT 2", "multiple_statements"),
    ("SELECT 1; DROP TABLE t", "multiple_statements"),
    ("SELECT 1;;DELETE FROM t", "multiple_statements"),
    ("", "empty_sql"),
    ("   \n ", "empty_sql"),
    ("-- only a comment", "empty_sql"),
    (None, "empty_sql"),
]


@pytest.mark.parametrize("sql,code", REJECTED)
def test_non_read_only_or_malformed_rejected(sql, code):
    d = policy.check(sql)
    assert not d.allowed
    assert d.reasons[0].code == code, d.reasons


def test_mutation_hidden_in_cte_is_rejected():
    for sql in [
        "WITH a AS (SELECT 1) INSERT INTO t SELECT * FROM a",
        "WITH a AS (SELECT 1) DELETE FROM t",
        "WITH a AS (DELETE FROM t RETURNING *) SELECT * FROM a",
        "WITH a AS (INSERT INTO t VALUES (1) RETURNING *) SELECT * FROM a",
        "WITH a AS (UPDATE t SET x = 1 RETURNING *) SELECT * FROM a",
    ]:
        assert not policy.check(sql).allowed, sql


def test_mutation_nested_in_subquery_or_set_operation_is_rejected():
    for sql in [
        "SELECT * FROM (DELETE FROM t RETURNING *)",
        "SELECT 1 UNION SELECT * FROM (INSERT INTO t VALUES (1) RETURNING *)",
        "SELECT (DELETE FROM t)",
        "SELECT 1 WHERE EXISTS (UPDATE t SET a = 1 RETURNING a)",
    ]:
        assert not policy.check(sql).allowed, sql


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT load_extension('evil')",
        "SELECT LOAD_EXTENSION('evil', 'entry')",
        "SELECT readfile('/etc/passwd')",
        "SELECT writefile('x', 'y')",
        "SELECT * FROM pragma_table_info('employees')",
        "SELECT name FROM pragma_database_list",
        "SELECT (SELECT load_extension('x'))",
        "WITH a AS (SELECT load_extension('x')) SELECT * FROM a",
    ],
)
def test_dangerous_functions_rejected(sql):
    d = policy.check(sql)
    assert not d.allowed and any(r.code == "denied_function" for r in d.reasons), d.reasons


def test_select_into_and_bound_parameters_rejected():
    assert not policy.check("SELECT * INTO newtable FROM t").allowed
    assert not policy.check("SELECT ?").allowed
    assert not policy.check("SELECT :a").allowed


def test_markdown_fences_are_not_repaired():
    d = policy.check("```sql\nSELECT 1\n```")
    assert not d.allowed


def test_limits_and_invalid_characters():
    assert policy.check("SELECT " + "1," * 20000 + "1").reasons[0].code == "too_long"
    assert policy.check("SELECT 1\x00; DROP TABLE t").reasons[0].code == "invalid_characters"
    assert SQLSafetyPolicy(max_sql_chars=10).check("SELECT 1234567890").reasons[0].code == "too_long"


def test_dialect_restricted_to_sqlite():
    with pytest.raises(ValueError):
        SQLSafetyPolicy(dialect="postgres")


# ---- regression: real smoke produced ": SELECT COUNT(emp_id) FROM employees".
# Root cause was a serving-prompt drift (fixed in the runtime, not by loosening
# safety or normalization), so leading label junk must STILL be rejected here.
@pytest.mark.parametrize(
    "sql",
    [
        ": SELECT COUNT(emp_id) FROM employees",
        "SQL: SELECT 1",
        ": DELETE FROM employees",
        "SQL: DROP TABLE employees",
        "SELECT FROM WHERE",
        "SELEC 1",
    ],
)
def test_label_artifacts_and_malformed_sql_stay_rejected(sql):
    assert not policy.check(sql).allowed


def test_normalization_stays_whitespace_only_and_valid_sql_is_untouched():
    from localsql.model.generation import normalize_predicted_sql

    assert normalize_predicted_sql(": SELECT 1") == ": SELECT 1"  # no label stripping
    assert normalize_predicted_sql("\r\n SELECT 1 \n") == "SELECT 1"
    ok = "SELECT SUM(T1.salary) FROM employees AS T1 INNER JOIN departments AS T2 ON T1.dept_id = T2.dept_id WHERE T2.name = 'Engineering'"
    assert normalize_predicted_sql(ok) == ok and policy.check(ok).allowed
