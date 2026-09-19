"""Logical database ids -> vetted filesystem paths.

API callers only ever supply a registered logical id. Paths come from
configuration, are relative to one allowed root, and are re-resolved (symlinks
followed) and re-checked to lie inside that root on every `resolve()`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Iterable, Optional

from localsql.backend.errors import ConfigError, DatabaseUnavailableError, UnknownDatabaseError

SUPPORTED_DIALECTS = frozenset({"sqlite"})  # PostgreSQL is designed-for, not implemented (Phase 8)
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


@dataclass(frozen=True)
class DatabaseEntry:
    id: str
    file: str  # relative to the registry root
    dialect: str = "sqlite"
    description: Optional[str] = None


@dataclass(frozen=True)
class RegisteredDatabase:
    id: str
    path: Path  # fully resolved, verified inside the root
    dialect: str
    description: Optional[str] = None


def _validate_relative_file(entry: DatabaseEntry) -> None:
    raw = entry.file
    if not raw or "\x00" in raw:
        raise ConfigError(f"database '{entry.id}': empty or invalid file")
    pure_win, pure_posix = PureWindowsPath(raw), PurePosixPath(raw)
    if pure_win.is_absolute() or pure_posix.is_absolute() or pure_win.drive or raw.startswith(("/", "\\")):
        raise ConfigError(f"database '{entry.id}': file must be relative to the database root")
    if ".." in pure_win.parts or ".." in pure_posix.parts:
        raise ConfigError(f"database '{entry.id}': path traversal ('..') is not allowed")


class DatabaseRegistry:
    def __init__(self, root: Path, entries: Iterable[DatabaseEntry]):
        self._root = Path(root).resolve()
        self._entries: dict[str, DatabaseEntry] = {}
        for e in entries:
            if not _ID_RE.match(e.id):
                raise ConfigError(f"invalid database id {e.id!r} (allowed: letters, digits, '_' and '-')")
            if e.id in self._entries:
                raise ConfigError(f"duplicate database id '{e.id}'")
            if e.dialect not in SUPPORTED_DIALECTS:
                raise ConfigError(
                    f"database '{e.id}': dialect '{e.dialect}' is not supported "
                    f"(supported: {sorted(SUPPORTED_DIALECTS)})"
                )
            _validate_relative_file(e)
            self._entries[e.id] = e

    @property
    def root(self) -> Path:
        return self._root

    def resolve(self, database_id: str) -> RegisteredDatabase:
        entry = self._entries.get(database_id) if isinstance(database_id, str) else None
        if entry is None:
            raise UnknownDatabaseError("Unknown database id.")
        candidate = (self._root / entry.file).resolve(strict=False)
        if not candidate.is_relative_to(self._root):
            # e.g. a symlink inside the root pointing outside it
            raise DatabaseUnavailableError("Database is not available.")
        if not candidate.is_file():
            raise DatabaseUnavailableError("Database is not available.")
        return RegisteredDatabase(entry.id, candidate, entry.dialect, entry.description)

    def list(self) -> list[dict]:
        """Public summaries -- never includes filesystem paths."""
        return [
            {"id": e.id, "dialect": e.dialect, "description": e.description}
            for e in sorted(self._entries.values(), key=lambda e: e.id)
        ]
