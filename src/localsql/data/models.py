"""Canonical typed data models for the LocalSQL training-data pipeline.

These models are the internal contract between raw BIRD data, database
schema metadata, and the final prepared text-to-SQL training examples.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class ForeignKeyRef(BaseModel):
    """A reference from a column to another table's column."""

    model_config = ConfigDict(frozen=True)

    table: str
    column: str


class ColumnSchema(BaseModel):
    """A single column within a table."""

    model_config = ConfigDict(frozen=True)

    name: str
    data_type: Optional[str] = None
    is_primary_key: bool = False
    foreign_key: Optional[ForeignKeyRef] = None
    description: Optional[str] = None


class TableSchema(BaseModel):
    """A single table within a database."""

    model_config = ConfigDict(frozen=True)

    name: str
    columns: list[ColumnSchema]


class DatabaseSchema(BaseModel):
    """The full schema of one database, as sourced from official BIRD metadata."""

    model_config = ConfigDict(frozen=True)

    db_id: str
    dialect: str
    tables: list[TableSchema]


class RawBirdExample(BaseModel):
    """One row exactly as it appears in birdsql/bird23-train-filtered.

    Source columns are db_id, question, evidence, SQL.
    """

    model_config = ConfigDict(frozen=True)

    row_index: int
    db_id: str
    question: str
    evidence: Optional[str] = None
    sql: str = Field(alias="SQL")


class SourceMetadata(BaseModel):
    """Provenance information for a prepared example."""

    model_config = ConfigDict(frozen=True)

    dataset_repo_id: str
    dataset_revision: Optional[str] = None
    row_index: int
    evidence_available: bool
    business_context_kept: bool


class PreparedTextToSQLExample(BaseModel):
    """A fully prepared, model-ready text-to-SQL training example."""

    model_config = ConfigDict(frozen=True)

    example_id: str
    db_id: str
    dialect: str
    question: str
    business_context: Optional[str] = None
    schema_serialized: str
    prompt: str
    completion: str
    source: SourceMetadata
    split: Literal["train", "validation"]
