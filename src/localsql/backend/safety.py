"""AST-based SQL safety policy (layer 1 of defense in depth).

Allows exactly ONE read-only statement whose root is SELECT / WITH...SELECT /
a set operation (UNION/INTERSECT/EXCEPT) of selects. The whole tree is walked,
so a mutation hidden in a CTE or subquery is rejected. This is an allowlist on
the root plus a denylist on nested constructs; anything sqlglot cannot parse as
a plain query (it falls back to `Command`) is rejected.

This layer is NOT the only barrier: the executor independently enforces
read-only access, so a parser differential here cannot mutate data.
"""

from __future__ import annotations

from typing import Optional

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

from localsql.backend.models import SafetyDecision, SafetyReason

DEFAULT_MAX_SQL_CHARS = 20_000

# Functions that read/write files, load code, or expose internals.
DENIED_FUNCTIONS = frozenset(
    {"load_extension", "readfile", "writefile", "edit", "fts3_tokenizer", "zipfile", "sqlite_dbpage"}
)


def _existing(*names: str) -> tuple[type, ...]:
    return tuple(getattr(exp, n) for n in names if hasattr(exp, n))


_ROOT_ALLOWED = _existing("Select", "SetOperation", "Subquery")
# Any of these ANYWHERE in the tree is a rejection (covers nested/CTE mutations).
_DENIED_NODES: dict[str, tuple[type, ...]] = {
    "mutation": _existing("Insert", "Update", "Delete", "Merge", "Copy", "LoadData", "TruncateTable"),
    "ddl": _existing("Create", "Drop", "Alter", "AlterColumn", "AlterRename", "Comment", "Analyze"),
    "attach_detach": _existing("Attach", "Detach"),
    "pragma": _existing("Pragma"),
    "transaction": _existing("Transaction", "Commit", "Rollback"),
    "session_command": _existing("Set", "Use", "Command", "Kill", "Lock", "Cache", "Uncache", "Refresh"),
    "select_into": _existing("Into"),
    "parameters": _existing("Placeholder", "Parameter"),
}


def _reject(*pairs: tuple[str, str]) -> SafetyDecision:
    return SafetyDecision(allowed=False, reasons=[SafetyReason(code=c, message=m) for c, m in pairs])


def _function_name(node: exp.Expression) -> Optional[str]:
    if isinstance(node, exp.Anonymous):
        return str(node.name).lower()
    if isinstance(node, exp.Func):
        return node.sql_name().lower()
    return None


class SQLSafetyPolicy:
    def __init__(self, max_sql_chars: int = DEFAULT_MAX_SQL_CHARS, dialect: str = "sqlite"):
        if dialect != "sqlite":
            raise ValueError("only the sqlite dialect is supported in Phase 8")
        self._max_sql_chars = max_sql_chars
        self._dialect = dialect

    def check(self, sql: Optional[str]) -> SafetyDecision:
        if sql is None or not sql.strip():
            return _reject(("empty_sql", "The model produced no SQL."))
        if len(sql) > self._max_sql_chars:
            return _reject(("too_long", f"SQL exceeds {self._max_sql_chars} characters."))
        if "\x00" in sql:
            return _reject(("invalid_characters", "SQL contains a NUL character."))

        try:
            parsed = sqlglot.parse(sql, read=self._dialect)
        except (SqlglotError, RecursionError):
            return _reject(("parse_error", "SQL could not be parsed."))
        # A trailing `;` or `; -- comment` yields None / Semicolon placeholders.
        statements = [s for s in parsed if s is not None and not isinstance(s, exp.Semicolon)]
        if len(statements) != 1:
            if not statements:
                return _reject(("empty_sql", "The model produced no SQL."))
            return _reject(("multiple_statements", f"Exactly one statement is allowed; found {len(statements)}."))

        root = statements[0]
        if not isinstance(root, _ROOT_ALLOWED):
            return _reject(
                ("not_read_only_query", f"Only SELECT / WITH queries are allowed; got {type(root).__name__}.")
            )

        reasons: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for node in root.walk():
            for code, types in _DENIED_NODES.items():
                if isinstance(node, types):
                    key = (code, type(node).__name__)
                    if key not in seen:
                        seen.add(key)
                        reasons.append((code, f"Disallowed construct in query: {type(node).__name__}."))
            if isinstance(node, exp.Table) and str(node.name).lower().startswith("pragma_"):
                if ("denied_function", "pragma_table") not in seen:
                    seen.add(("denied_function", "pragma_table"))
                    reasons.append(("denied_function", "SQLite pragma table-valued functions are not allowed."))
            name = _function_name(node)
            if name is not None and (name in DENIED_FUNCTIONS or name.startswith("pragma_")):
                key = ("denied_function", name)
                if key not in seen:
                    seen.add(key)
                    reasons.append(("denied_function", f"Function '{name}' is not allowed."))
        if reasons:
            return _reject(*reasons)
        return SafetyDecision(allowed=True)
