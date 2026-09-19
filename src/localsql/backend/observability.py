"""Lightweight structured (JSON-line) logging.

Fields are restricted to identifiers, stages, latencies and error/safety codes.
Never pass result rows, questions, SQL text, prompts, or secrets to `log_event`.
"""

from __future__ import annotations

import json
import logging
from typing import Any

LOGGER_NAME = "schemaforge.backend"

# Defensive: values under these keys are dropped even if a caller passes them.
_FORBIDDEN_KEYS = frozenset({"rows", "result", "question", "sql", "generated_sql", "prompt", "business_context", "password", "token", "secret"})


def get_logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)


def configure_logging(level: str = "INFO") -> None:
    logger = get_logger()
    logger.setLevel(level.upper())
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.propagate = False


def log_event(logger: logging.Logger, event: str, level: int = logging.INFO, **fields: Any) -> None:
    safe = {k: v for k, v in fields.items() if k not in _FORBIDDEN_KEYS}
    logger.log(level, json.dumps({"event": event, **safe}, default=str, sort_keys=True))
