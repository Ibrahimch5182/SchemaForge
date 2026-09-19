"""Deterministic demo SQLite database (used by scripts/smoke_phase8.py).

Small fixed schema exercising a single-column FK and a composite primary key.
Contains only synthetic data.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

_DDL = """
CREATE TABLE departments (
    dept_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL
);
CREATE TABLE employees (
    emp_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    dept_id INTEGER REFERENCES departments(dept_id),
    salary REAL,
    hire_date TEXT
);
CREATE TABLE project_assignments (
    emp_id INTEGER,
    project_id INTEGER,
    hours REAL,
    PRIMARY KEY (emp_id, project_id),
    FOREIGN KEY (emp_id) REFERENCES employees(emp_id)
);
"""

_DEPARTMENTS = [(1, "Engineering"), (2, "Sales"), (3, "Support")]
_EMPLOYEES = [
    (1, "Ada", 1, 120000.0, "2019-03-01"),
    (2, "Grace", 1, 130000.0, "2018-07-15"),
    (3, "Linus", 1, 110000.0, "2021-01-10"),
    (4, "Margaret", 1, 125000.0, "2020-05-20"),
    (5, "Don", 2, 90000.0, "2017-11-02"),
    (6, "Barbara", 2, 95000.0, "2019-09-09"),
    (7, "Ken", 2, 88000.0, "2022-02-14"),
    (8, "Dennis", 3, 70000.0, "2016-06-30"),
    (9, "Radia", 3, 72000.0, "2020-10-01"),
    (10, "Tim", 3, 68000.0, "2023-04-17"),
    (11, "Vint", 1, 140000.0, "2015-08-08"),
    (12, "Frances", 2, 92000.0, "2021-12-12"),
]
_ASSIGNMENTS = [(1, 10, 120.5), (1, 11, 40.0), (2, 10, 80.0), (3, 12, 200.0), (4, 11, 60.0), (5, 13, 15.0)]

DEMO_TABLES = ["departments", "employees", "project_assignments"]
DEMO_EMPLOYEE_COUNT = len(_EMPLOYEES)


def create_demo_database(path: Path) -> Path:
    """(Re)create the demo DB at `path` deterministically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    conn = sqlite3.connect(path)
    try:
        conn.executescript(_DDL)
        conn.executemany("INSERT INTO departments VALUES (?, ?)", _DEPARTMENTS)
        conn.executemany("INSERT INTO employees VALUES (?, ?, ?, ?, ?)", _EMPLOYEES)
        conn.executemany("INSERT INTO project_assignments VALUES (?, ?, ?)", _ASSIGNMENTS)
        conn.commit()
    finally:
        conn.close()
    return path
