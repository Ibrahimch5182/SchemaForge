"""Small synthetic Mini-Dev-shaped fixtures shared across benchmark tests.

Entirely synthetic (not copied from real BIRD Mini-Dev), so tests need no
network access and don't depend on the full 500-example benchmark.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

SCHEMA_ENTRIES = [
    {
        "db_id": "shop_db",
        "table_names_original": ["customers"],
        "table_names": ["customers"],
        "column_names_original": [[-1, "*"], [0, "customer_id"], [0, "country"]],
        "column_names": [[-1, "*"], [0, "customer_id"], [0, "country"]],
        "column_types": ["text", "integer", "text"],
        "primary_keys": [1],
        "foreign_keys": [],
    }
]

QUESTIONS = [
    {
        "question_id": 1,
        "db_id": "shop_db",
        "question": "How many customers are there?",
        "evidence": "",
        "SQL": "SELECT COUNT(*) FROM customers",
        "difficulty": "simple",
    },
    {
        "question_id": 2,
        "db_id": "shop_db",
        "question": "How many customers are from France?",
        "evidence": "France refers to country = 'France'",
        "SQL": "SELECT COUNT(*) FROM customers WHERE country = 'France'",
        "difficulty": "moderate",
    },
    {
        "question_id": 3,
        "db_id": "shop_db",
        "question": "List all customer ids.",
        "evidence": "",
        # HF's SQL field deliberately diverges from the archive gold SQL for
        # this row (see ARCHIVE_GOLD_SQL_OVERRIDES) -- mirrors the real
        # Phase 2A finding that HF and the archive disagree on some rows.
        # This is diagnostic-only and must NOT be used for grading.
        "SQL": "SELECT customer_id FROM customers",
        "difficulty": "simple",
    },
]

# Simulates the archive's mini_dev_sqlite_gold.sql containing different SQL
# than HF's questions file for index 2 -- the canonical grading source must
# use this value, not QUESTIONS[2]["SQL"].
ARCHIVE_GOLD_SQL_OVERRIDES = {2: "SELECT customer_id FROM customers ORDER BY customer_id"}


def write_fixture_workspace(tmp_path: Path) -> dict[str, Path]:
    raw_dir = tmp_path / "raw"
    databases_dir = tmp_path / "databases"
    raw_dir.mkdir(parents=True)
    databases_dir.mkdir(parents=True)

    schema_path = raw_dir / "dev_tables.json"
    schema_path.write_text(json.dumps(SCHEMA_ENTRIES), encoding="utf-8")

    gold_path = raw_dir / "mini_dev_sqlite_gold.sql"
    gold_lines = [
        f"{ARCHIVE_GOLD_SQL_OVERRIDES.get(i, q['SQL'])}\t{q['db_id']}"
        for i, q in enumerate(QUESTIONS)
    ]
    gold_path.write_text("\n".join(gold_lines) + "\n", encoding="utf-8")

    db_dir = databases_dir / "shop_db"
    db_dir.mkdir()
    db_path = db_dir / "shop_db.sqlite"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE customers (customer_id INTEGER PRIMARY KEY, country TEXT)")
    conn.executemany(
        "INSERT INTO customers (country) VALUES (?)",
        [("France",), ("France",), ("Germany",)],
    )
    conn.commit()
    conn.close()

    return {
        "raw_dir": raw_dir,
        "databases_dir": databases_dir,
        "schema_path": schema_path,
        "gold_path": gold_path,
        "db_path": db_path,
    }
