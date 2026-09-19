"""Service-level typed request/response models (no FastAPI dependency)."""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

QueryStatus = Literal["ok", "model_error", "unsafe_sql", "validation_error", "execution_error", "schema_error", "cancelled"]
Stage = Literal["schema", "model", "safety", "preflight", "execution"]
StageState = Literal["passed", "failed", "not_run"]


class QueryRequest(BaseModel):
    """Caller input. Deliberately has NO path/connection field: databases are
    addressed only by registered logical id."""

    model_config = ConfigDict(extra="forbid")

    database_id: str = Field(min_length=1, max_length=64)
    question: str = Field(min_length=1, max_length=2000)
    business_context: Optional[str] = Field(default=None, max_length=4000)

    @field_validator("question")
    @classmethod
    def _question_not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("question must not be blank")
        return v

    @field_validator("business_context")
    @classmethod
    def _blank_context_is_none(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        v = v.strip()
        return v or None


class SafetyReason(BaseModel):
    code: str
    message: str


class SafetyDecision(BaseModel):
    allowed: bool
    reasons: list[SafetyReason] = Field(default_factory=list)


class ExecutionResult(BaseModel):
    """Rows are capped at `max_rows`; `truncated` is True iff at least one more
    row existed. No total row count is reported because none is known."""

    columns: list[str]
    rows: list[list[Any]]
    returned_row_count: int
    truncated: bool
    max_rows: int
    elapsed_ms: float


class ErrorInfo(BaseModel):
    stage: Stage
    code: str
    message: str
    #: Short identifier-only hint (e.g. the missing column name). Never a raw engine message.
    detail: Optional[str] = None


class Timings(BaseModel):
    schema_ms: Optional[float] = None
    model_ms: Optional[float] = None
    safety_ms: Optional[float] = None
    preflight_ms: Optional[float] = None
    execution_ms: Optional[float] = None
    total_ms: float = 0.0


class ReliabilityInfo(BaseModel):
    """What was and was NOT verified about this answer.

    The deterministic stages verify safety, validity for the database and
    successful execution. None of that establishes that the SQL answers the
    question, and no calibrated correctness probability exists, so
    `semantic_correctness` is always "not_verified" and `confidence` is always
    null. Never invent a score here.
    """

    safety: StageState = "not_run"
    preflight: StageState = "not_run"
    execution: StageState = "not_run"
    semantic_correctness: Literal["not_verified"] = "not_verified"
    confidence: None = None
    note: str = (
        "Safety, database validation and successful execution do not guarantee the answer is correct. "
        "No calibrated confidence score is available."
    )


class QueryResponse(BaseModel):
    request_id: str
    database_id: str
    status: QueryStatus
    generated_sql: Optional[str] = None
    safety: Optional[SafetyDecision] = None
    result: Optional[ExecutionResult] = None
    error: Optional[ErrorInfo] = None
    timings: Timings
    model: dict[str, Any] = Field(default_factory=dict)
    prompt_sha256: Optional[str] = None
    dialect: Optional[str] = None
    reliability: ReliabilityInfo = Field(default_factory=ReliabilityInfo)
