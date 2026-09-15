"""Synthetic schema fixtures for Phase 5A `schema_context` tests.

Entirely synthetic (not real BIRD data), so tests need no network/CUDA and
don't depend on the full 6,067-example training set.
"""

from __future__ import annotations

from localsql.data.models import ColumnSchema, DatabaseSchema, ForeignKeyRef, TableSchema


def make_shop_schema() -> DatabaseSchema:
    customers = TableSchema(
        name="customers",
        columns=[
            ColumnSchema(name="customer_id", data_type="INTEGER", is_primary_key=True, description="PK"),
            ColumnSchema(name="name", data_type="TEXT", description="Customer full name."),
            ColumnSchema(name="country", data_type="TEXT", description="Customer country."),
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
                description="References the ordering customer.",
            ),
            ColumnSchema(name="order_date", data_type="DATE", description="Date the order was placed."),
        ],
    )
    products = TableSchema(
        name="products",
        columns=[
            ColumnSchema(name="product_id", data_type="INTEGER", is_primary_key=True),
            ColumnSchema(name="product_name", data_type="TEXT", description="Name of the product."),
            ColumnSchema(name="price", data_type="NUMERIC", description="Unit price."),
        ],
    )
    order_items = TableSchema(
        name="order_items",
        columns=[
            ColumnSchema(name="item_id", data_type="INTEGER", is_primary_key=True),
            ColumnSchema(
                name="order_id",
                data_type="INTEGER",
                foreign_key=ForeignKeyRef(table="orders", column="order_id"),
            ),
            ColumnSchema(
                name="product_id",
                data_type="INTEGER",
                foreign_key=ForeignKeyRef(table="products", column="product_id"),
            ),
            ColumnSchema(name="quantity", data_type="INTEGER", description="Number of units ordered."),
        ],
    )
    warehouse_logs = TableSchema(
        name="warehouse_logs",
        columns=[
            ColumnSchema(name="log_id", data_type="INTEGER", is_primary_key=True),
            ColumnSchema(name="note", data_type="TEXT", description="Free-text warehouse note."),
        ],
    )
    return DatabaseSchema(
        db_id="shop_db",
        dialect="sqlite",
        tables=[customers, orders, products, order_items, warehouse_logs],
    )
