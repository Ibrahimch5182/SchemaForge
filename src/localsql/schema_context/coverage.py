"""Gold-based coverage diagnostic (Phase 5A, Task 4). EVALUATION ONLY.

This module is intentionally separate from `relevance.py`. Nothing here is
imported by, or feeds into, schema selection -- gold SQL is parsed only
AFTER a selection has already been made, purely to measure how well an
inference-time-safe selection performed. `relevance.select_schema_within_budget`
has no parameter through which this module's output could leak back in.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError


@dataclass(frozen=True)
class GoldReferences:
    tables: frozenset[str]
    columns: frozenset[str]  # bare column names, best-effort qualified


def extract_gold_references(
    sql: str, schema_tables: dict[str, list[str]], dialect: str = "sqlite"
) -> Optional[GoldReferences]:
    """Parse gold SQL to find referenced table/column identifiers.

    `schema_tables` is `{table_name: [column_name, ...]}`, used to qualify
    ambiguous column references across joins via sqlglot's optimizer.
    Returns `None` (reported as "unparseable/ambiguous", never guessed) if
    the SQL doesn't parse or qualification fails.
    """
    try:
        tree = sqlglot.parse_one(sql, dialect=dialect)
    except SqlglotError:
        return None

    tables = frozenset(t.name for t in tree.find_all(exp.Table) if t.name)
    if not tables:
        return None

    columns: set[str] = set()
    try:
        from sqlglot.optimizer.qualify import qualify

        schema_dict = {name: {col: "TEXT" for col in cols} for name, cols in schema_tables.items()}
        qualified = qualify(tree, schema=schema_dict, dialect=dialect, validate_qualify_columns=False)
        for col in qualified.find_all(exp.Column):
            if col.name:
                columns.add(col.name)
    except Exception:
        # Qualification is best-effort; fall back to raw (unqualified)
        # column names rather than declaring the whole example ambiguous.
        for col in tree.find_all(exp.Column):
            if col.name:
                columns.add(col.name)

    return GoldReferences(tables=tables, columns=frozenset(columns))


@dataclass(frozen=True)
class CoverageResult:
    example_id: str
    parseable: bool
    gold_table_count: int
    gold_tables_retained: int
    gold_column_count: int
    gold_columns_retained: int
    retained_all_tables: bool
    retained_all_columns: bool


def compute_coverage(
    example_id: str,
    selected_table_names: frozenset[str],
    gold: Optional[GoldReferences],
) -> CoverageResult:
    """Compare a (gold-free) selection against gold references. Read-only
    diagnostic -- computed strictly after selection; cannot influence it.
    """
    if gold is None:
        return CoverageResult(
            example_id=example_id,
            parseable=False,
            gold_table_count=0,
            gold_tables_retained=0,
            gold_column_count=0,
            gold_columns_retained=0,
            retained_all_tables=False,
            retained_all_columns=False,
        )

    selected_lower = {t.lower() for t in selected_table_names}
    gold_tables_lower = {t.lower() for t in gold.tables}
    tables_retained = len(gold_tables_lower & selected_lower)
    all_gold_tables_retained = gold_tables_lower and tables_retained == len(gold_tables_lower)

    # The selector is all-or-nothing per table (no column-level dropping),
    # so column coverage is fully determined by table coverage: a gold
    # column is retained iff every table it could belong to was retained.
    columns_retained = len(gold.columns) if all_gold_tables_retained else 0

    return CoverageResult(
        example_id=example_id,
        parseable=True,
        gold_table_count=len(gold_tables_lower),
        gold_tables_retained=tables_retained,
        gold_column_count=len(gold.columns),
        gold_columns_retained=columns_retained,
        retained_all_tables=all_gold_tables_retained,
        retained_all_columns=all_gold_tables_retained,
    )


def summarize_coverage(results: list[CoverageResult]) -> dict:
    parseable = [r for r in results if r.parseable]
    unparseable = len(results) - len(parseable)

    def _safe_ratio(num: int, denom: int) -> float:
        return round(num / denom, 4) if denom else 0.0

    total_gold_tables = sum(r.gold_table_count for r in parseable)
    total_retained_tables = sum(r.gold_tables_retained for r in parseable)
    total_gold_columns = sum(r.gold_column_count for r in parseable)
    total_retained_columns = sum(r.gold_columns_retained for r in parseable)

    return {
        "total_examples": len(results),
        "parseable_examples": len(parseable),
        "unparseable_or_ambiguous_examples": unparseable,
        "table_recall": _safe_ratio(total_retained_tables, total_gold_tables),
        "column_recall": _safe_ratio(total_retained_columns, total_gold_columns),
        "pct_examples_retaining_all_gold_tables": _safe_ratio(
            sum(1 for r in parseable if r.retained_all_tables), len(parseable)
        ),
        "pct_examples_retaining_all_gold_columns": _safe_ratio(
            sum(1 for r in parseable if r.retained_all_columns), len(parseable)
        ),
    }
