import os
import sqlite3

import pytest

from localsql.backend.errors import (
    ConfigError,
    DatabaseUnavailableError,
    SchemaIntrospectionError,
    UnknownDatabaseError,
)
from localsql.backend.introspection import SQLiteSchemaIntrospector
from localsql.backend.registry import DatabaseEntry, DatabaseRegistry
from localsql.data.schema_serializer import serialize_schema

from tests.backend.helpers import make_registry


# ---------------------------------------------------------------- registry
def test_resolve_registered_database(tmp_path):
    reg = make_registry(tmp_path)
    db = reg.resolve("demo")
    assert db.path == (tmp_path / "dbroot" / "demo.sqlite").resolve() and db.dialect == "sqlite"


@pytest.mark.parametrize("bad_id", ["nope", "../demo", "demo/../demo", "C:\\x.sqlite", "", None, 5])
def test_unknown_or_hostile_ids_are_unknown(tmp_path, bad_id):
    with pytest.raises(UnknownDatabaseError):
        make_registry(tmp_path).resolve(bad_id)


@pytest.mark.parametrize(
    "file",
    ["../outside.sqlite", "sub/../../outside.sqlite", "..\\outside.sqlite", "/etc/passwd", "\\abs.sqlite", "C:\\Windows\\x.db", "C:x.db", ""],
)
def test_config_rejects_traversal_and_absolute_paths(tmp_path, file):
    with pytest.raises(ConfigError):
        DatabaseRegistry(tmp_path, [DatabaseEntry("db1", file)])


def test_config_rejects_bad_id_duplicate_and_unsupported_dialect(tmp_path):
    with pytest.raises(ConfigError, match="invalid database id"):
        DatabaseRegistry(tmp_path, [DatabaseEntry("bad id!", "a.db")])
    with pytest.raises(ConfigError, match="duplicate"):
        DatabaseRegistry(tmp_path, [DatabaseEntry("a", "a.db"), DatabaseEntry("a", "b.db")])
    with pytest.raises(ConfigError, match="not supported"):
        DatabaseRegistry(tmp_path, [DatabaseEntry("pg", "a.db", dialect="postgresql")])


def test_missing_file_is_unavailable(tmp_path):
    with pytest.raises(DatabaseUnavailableError):
        make_registry(tmp_path, with_demo=False).resolve("demo")


def test_symlink_escaping_root_is_rejected(tmp_path):
    outside = tmp_path / "outside.sqlite"
    sqlite3.connect(outside).close()
    root = tmp_path / "root"
    root.mkdir()
    link = root / "link.sqlite"
    try:
        os.symlink(outside, link)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not permitted on this machine")
    reg = DatabaseRegistry(root, [DatabaseEntry("link", "link.sqlite")])
    with pytest.raises(DatabaseUnavailableError):
        reg.resolve("link")


def test_public_listing_never_exposes_paths(tmp_path):
    listing = make_registry(tmp_path).list()
    assert listing == [{"id": "demo", "dialect": "sqlite", "description": "demo db"}]
    assert "dbroot" not in str(listing)


# ------------------------------------------------------------ introspection
def _db(tmp_path, ddl):
    root = tmp_path / "r"
    root.mkdir(exist_ok=True)
    path = root / "t.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript(ddl)
    conn.commit()
    conn.close()
    return DatabaseRegistry(root, [DatabaseEntry("t", "t.sqlite")]).resolve("t")


def test_introspection_pk_fk_types_and_order(tmp_path):
    db = _db(
        tmp_path,
        """
        CREATE TABLE departments (dept_id INTEGER PRIMARY KEY, name text);
        CREATE TABLE employees (emp_id integer PRIMARY KEY, name TEXT, dept_id INTEGER REFERENCES departments(dept_id), pay);
        CREATE TABLE assign (emp_id INTEGER, project_id INTEGER, hours REAL,
            PRIMARY KEY (project_id, emp_id), FOREIGN KEY (emp_id) REFERENCES employees(emp_id));
        """,
    )
    schema = SQLiteSchemaIntrospector().introspect(db)
    assert schema.dialect == "sqlite" and schema.db_id == "t"
    assert [t.name for t in schema.tables] == ["departments", "employees", "assign"]  # creation order
    emp = {c.name: c for c in schema.tables[1].columns}
    assert emp["emp_id"].data_type == "INTEGER" and emp["emp_id"].is_primary_key
    assert emp["pay"].data_type is None  # undeclared type stays None, not invented
    assert (emp["dept_id"].foreign_key.table, emp["dept_id"].foreign_key.column) == ("departments", "dept_id")
    assign = {c.name: c for c in schema.tables[2].columns}
    assert assign["emp_id"].is_primary_key and assign["project_id"].is_primary_key and not assign["hours"].is_primary_key
    assert assign["emp_id"].foreign_key.table == "employees"


def test_implicit_fk_target_resolves_to_referenced_pk(tmp_path):
    db = _db(
        tmp_path,
        "CREATE TABLE a (id INTEGER PRIMARY KEY); CREATE TABLE b (x INTEGER REFERENCES a);",
    )
    b = {c.name: c for c in SQLiteSchemaIntrospector().introspect(db).tables[1].columns}
    assert (b["x"].foreign_key.table, b["x"].foreign_key.column) == ("a", "id")


def test_composite_fk_yields_reference_per_column(tmp_path):
    db = _db(
        tmp_path,
        """
        CREATE TABLE p (a INTEGER, b INTEGER, PRIMARY KEY (a, b));
        CREATE TABLE c (x INTEGER, y INTEGER, FOREIGN KEY (x, y) REFERENCES p (a, b));
        """,
    )
    c = {col.name: col for col in SQLiteSchemaIntrospector().introspect(db).tables[1].columns}
    assert c["x"].foreign_key.column == "a" and c["y"].foreign_key.column == "b"


def test_internal_tables_and_views_excluded(tmp_path):
    db = _db(
        tmp_path,
        "CREATE TABLE t (id INTEGER PRIMARY KEY AUTOINCREMENT, v TEXT); INSERT INTO t(v) VALUES ('x'); CREATE VIEW vw AS SELECT * FROM t;",
    )
    assert [t.name for t in SQLiteSchemaIntrospector().introspect(db).tables] == ["t"]  # no sqlite_sequence, no view


def test_output_feeds_canonical_serializer_deterministically(tmp_path):
    db = _db(tmp_path, "CREATE TABLE a (id INTEGER PRIMARY KEY); CREATE TABLE b (x INTEGER REFERENCES a(id), n TEXT);")
    intro = SQLiteSchemaIntrospector()
    text = serialize_schema(intro.introspect(db))
    assert text == "a(\n  id INTEGER PK\n)\n\nb(\n  x INTEGER FK->a.id,\n  n TEXT\n)"
    assert text == serialize_schema(intro.introspect(db))


def test_no_tables_and_non_database_file_fail_cleanly(tmp_path):
    empty = _db(tmp_path, "")
    with pytest.raises(SchemaIntrospectionError, match="no tables"):
        SQLiteSchemaIntrospector().introspect(empty)
    root = tmp_path / "junk"
    root.mkdir()
    (root / "j.sqlite").write_bytes(b"this is not a sqlite database" * 20)
    junk = DatabaseRegistry(root, [DatabaseEntry("j", "j.sqlite")]).resolve("j")
    with pytest.raises(SchemaIntrospectionError):
        SQLiteSchemaIntrospector().introspect(junk)


def test_introspection_does_not_modify_database(tmp_path):
    from tests.backend.helpers import file_sha

    db = _db(tmp_path, "CREATE TABLE t (id INTEGER PRIMARY KEY);")
    before = file_sha(db.path)
    SQLiteSchemaIntrospector().introspect(db)
    assert file_sha(db.path) == before
