"""Typed models for the BIRD Mini-Dev benchmark: generation/grading isolation.

`GenerationExample` is what future model-inference code may see. It must
never carry gold SQL. `GradingExample` is the isolated counterpart used only
by evaluation code.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict

# Fields that must never appear on a GenerationExample -- enforced both by
# the model's own field set and by an explicit leakage test.
FORBIDDEN_GENERATION_FIELDS = {"sql", "gold_sql", "completion", "target", "target_sql", "answer"}


class GenerationExample(BaseModel):
    """Gold-free input for future model generation. No answer fields."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    example_id: str
    db_id: str
    dialect: str
    question: str
    business_context: Optional[str] = None
    serialized_schema: str
    prompt: str
    difficulty: Optional[str] = None


class GradingExample(BaseModel):
    """Isolated grading reference. Only evaluation code should load this."""

    model_config = ConfigDict(frozen=True)

    example_id: str
    db_id: str
    dialect: str
    sql: str
    difficulty: Optional[str] = None
    evidence: Optional[str] = None


class PredictionRecord(BaseModel):
    """One LocalSQL benchmark prediction. `predicted_sql` is required;
    everything else is optional metadata the official grader ignores.
    """

    model_config = ConfigDict(frozen=True, extra="ignore")

    example_id: str
    db_id: str
    predicted_sql: str
    latency_ms: Optional[float] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    model_id: Optional[str] = None
    context_mode: Optional[str] = None
