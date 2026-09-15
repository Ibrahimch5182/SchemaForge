"""Deterministic, question-conditioned schema relevance scoring and
budgeted schema serialization (Phase 5A, Task 3).

INFERENCE-TIME-SAFE BY CONSTRUCTION: `select_schema_within_budget`'s
signature has no parameter for gold SQL, gold tables, or gold columns.
Only information available at real inference time is accepted: the
natural-language question, optional business context, and the database
schema itself. This is enforced by the function signature, not just by
convention -- there is nothing to accidentally pass gold data into.

No embeddings, no vector DB, no LLM call, no RAG, no agents. Term-overlap
scoring + a foreign-key closure + greedy whole-table packing against a
caller-supplied length budget.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from localsql.data.models import DatabaseSchema, TableSchema
from localsql.schema_context.compact_serializer import serialize_schema_compact

_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_NON_ALNUM = re.compile(r"[^a-zA-Z0-9]+")
_WORD = re.compile(r"[a-zA-Z0-9]+")


def normalize_identifier(name: str) -> frozenset[str]:
    """Split an identifier into normalized (lowercased) tokens, splitting on
    snake_case/camelCase/non-alphanumeric boundaries. `customer_id` and
    `CustomerID` both normalize to `{"customer", "id"}`."""
    spaced = _CAMEL_BOUNDARY.sub("_", name)
    tokens = _NON_ALNUM.split(spaced)
    return frozenset(t.lower() for t in tokens if t)


def tokenize_text(text: str) -> frozenset[str]:
    """Lowercased word tokens from free text (question/business context)."""
    return frozenset(m.group(0).lower() for m in _WORD.finditer(text))


@dataclass(frozen=True)
class TableRelevance:
    table_name: str
    score: float
    matched_terms: frozenset[str]


def score_table_relevance(table: TableSchema, question_terms: frozenset[str]) -> TableRelevance:
    """Deterministic relevance score for one table against a term set.

    Prefers exact/normalized identifier matches (task requirement 2):
    a whole-table-name match scores highest; partial table-name term
    overlap and column-name term overlap add smaller, bounded amounts.
    Never depends on iteration/hash order -- pure function of
    (table, question_terms).
    """
    table_terms = normalize_identifier(table.name)
    matched = table_terms & question_terms

    score = 0.0
    if table_terms and table_terms <= question_terms:
        score += 2.0  # every token of the table name appears in the question
    elif matched:
        score += len(matched) / max(len(table_terms), 1)

    if table.columns:
        col_hits = sum(1 for c in table.columns if normalize_identifier(c.name) & question_terms)
        score += col_hits / len(table.columns)

    return TableRelevance(table_name=table.name, score=score, matched_terms=matched)


def add_fk_closure(schema: DatabaseSchema, selected: frozenset[str]) -> frozenset[str]:
    """Add any table that is the foreign-key TARGET of an already-selected
    table's column (task requirement 5), to a fixed point (handles FK
    chains). Deterministic: depends only on `schema` and `selected`.
    """
    tables_by_name = {t.name: t for t in schema.tables}
    result = set(selected)
    changed = True
    while changed:
        changed = False
        for name in sorted(result):  # sorted: deterministic iteration order
            table = tables_by_name.get(name)
            if table is None:
                continue
            for col in table.columns:
                if col.foreign_key and col.foreign_key.table not in result:
                    if col.foreign_key.table in tables_by_name:
                        result.add(col.foreign_key.table)
                        changed = True
    return frozenset(result)


@dataclass(frozen=True)
class SchemaSelectionResult:
    db_id: str
    selected_table_names: tuple[str, ...]  # in schema's original deterministic order
    dropped_table_names: tuple[str, ...]
    serialized_schema: str
    estimated_length: int
    budget: int
    within_budget: bool
    fell_back_to_single_table: bool


def select_schema_within_budget(
    schema: DatabaseSchema,
    question: str,
    business_context: str | None,
    budget: int,
    length_fn: Callable[[str], int] = len,
    serializer: Callable[[DatabaseSchema], str] = serialize_schema_compact,
) -> SchemaSelectionResult:
    """Deterministic, question-conditioned schema selection within a length
    budget. `length_fn` measures the serialized schema (default: character
    count for offline use; pass a real tokenizer-based counter on Kaggle
    for exact enforcement -- the algorithm itself never depends on which
    length metric is used).

    Algorithm:
    1. Score every table's relevance to (question, business_context).
    2. Visit tables in descending-score order (ties broken by the table's
       original schema position -- deterministic, no hashing).
    3. For each table, tentatively add it plus its FK closure (requirement
       5) to the running selection; keep the addition only if the
       resulting *whole* serialized schema still fits the budget --
       otherwise skip that table and keep trying lower-scored ones
       (never partially includes a table, so no identifier is ever cut
       mid-string -- requirement 8).
    4. If nothing fits (pathological tiny budget), fall back to the single
       highest-scored table alone, so a selection is never empty.

    Gold SQL/answers play no role anywhere in this function -- there is no
    parameter for them.
    """
    question_terms = tokenize_text(question)
    if business_context:
        question_terms = question_terms | tokenize_text(business_context)

    original_order = [t.name for t in schema.tables]
    tables_by_name = {t.name: t for t in schema.tables}

    scored = [score_table_relevance(t, question_terms) for t in schema.tables]
    # Deterministic order: score desc, then original schema position asc.
    visit_order = sorted(
        range(len(scored)),
        key=lambda i: (-scored[i].score, i),
    )

    def serialize_subset(names: frozenset[str]) -> str:
        subset_tables = [tables_by_name[n] for n in original_order if n in names]
        subset_schema = DatabaseSchema(db_id=schema.db_id, dialect=schema.dialect, tables=subset_tables)
        return serializer(subset_schema)

    selected: frozenset[str] = frozenset()
    for idx in visit_order:
        name = original_order[idx]
        if name in selected:
            continue
        candidate = add_fk_closure(schema, selected | {name})
        if length_fn(serialize_subset(candidate)) <= budget:
            selected = candidate

    fell_back = False
    if not selected:
        # Pathologically small budget: never return an empty schema.
        best_idx = max(range(len(scored)), key=lambda i: (scored[i].score, -i))
        selected = frozenset({original_order[best_idx]})
        fell_back = True

    final_text = serialize_subset(selected)
    final_length = length_fn(final_text)
    selected_ordered = tuple(n for n in original_order if n in selected)
    dropped_ordered = tuple(n for n in original_order if n not in selected)

    return SchemaSelectionResult(
        db_id=schema.db_id,
        selected_table_names=selected_ordered,
        dropped_table_names=dropped_ordered,
        serialized_schema=final_text,
        estimated_length=final_length,
        budget=budget,
        within_budget=final_length <= budget,
        fell_back_to_single_table=fell_back,
    )
