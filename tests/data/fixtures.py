"""Small synthetic fixtures shared across unit tests (no BIRD download needed)."""

from __future__ import annotations

from localsql.data.models import (
    ColumnSchema,
    DatabaseSchema,
    ForeignKeyRef,
    RawBirdExample,
    TableSchema,
)


def make_shop_schema() -> DatabaseSchema:
    customers = TableSchema(
        name="customers",
        columns=[
            ColumnSchema(name="customer_id", data_type="INTEGER", is_primary_key=True),
            ColumnSchema(name="country", data_type="TEXT"),
            ColumnSchema(name="segment", data_type="TEXT", description="Customer segment label."),
        ],
    )
    orders = TableSchema(
        name="orders",
        columns=[
            ColumnSchema(name="order_id", data_type="INTEGER", is_primary_key=True),
            ColumnSchema(
                name="customer_id",
                data_type="INTEGER",
                foreign_key=ForeignKeyRef(table="customers", column="customer_id"),
            ),
            ColumnSchema(name="order_date", data_type="DATE"),
            ColumnSchema(name="total", data_type="NUMERIC"),
        ],
    )
    return DatabaseSchema(db_id="shop", dialect="sqlite", tables=[customers, orders])


def make_raw_example(
    row_index: int = 0,
    db_id: str = "shop",
    question: str = "How many customers are there?",
    evidence: str | None = "customers refers to the customers table",
    sql: str = "SELECT COUNT(*) FROM customers",
) -> RawBirdExample:
    return RawBirdExample.model_validate(
        {
            "row_index": row_index,
            "db_id": db_id,
            "question": question,
            "evidence": evidence,
            "SQL": sql,
        }
    )
