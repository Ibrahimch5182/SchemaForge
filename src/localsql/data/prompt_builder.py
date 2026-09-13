"""The single canonical prompt/completion contract for LocalSQL.

This builder is reused, unmodified, across training, validation, baseline
inference, fine-tuned inference, and production inference. Do not create
alternate prompt templates elsewhere.
"""

from __future__ import annotations

SYSTEM_INSTRUCTIONS = (
    "You are a text-to-SQL model.\n"
    "Generate exactly one read-only SQL query that answers the question "
    "using only the provided database schema.\n"
    "Return SQL only.\n"
    "Do not use markdown."
)


def build_prompt(
    schema_serialized: str,
    dialect: str,
    question: str,
    business_context: str | None = None,
) -> str:
    """Build the canonical prompt. Omits the BUSINESS CONTEXT section if absent."""
    sections = [
        f"SYSTEM:\n{SYSTEM_INSTRUCTIONS}",
        f"DIALECT:\n{dialect}",
        f"SCHEMA:\n{schema_serialized}",
    ]
    if business_context:
        sections.append(f"BUSINESS CONTEXT:\n{business_context}")
    sections.append(f"QUESTION:\n{question}")
    return "\n\n".join(sections)


def build_completion(sql: str) -> str:
    """Normalize gold SQL into the SQL-only completion contract.

    Only leading/trailing whitespace is trimmed. Internal whitespace is left
    untouched, since collapsing it could alter the semantics of string
    literals embedded in the query.
    """
    return sql.strip()
