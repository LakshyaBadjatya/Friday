"""Persistent long-term memory behind a ``LongTermStore`` protocol (Phase 4).

This module provides FRIDAY's durable memory layer: facts the assistant has been
told to remember, a history of executed tasks, and an audit trail of orchestrator
steps. The contract is the :class:`LongTermStore` protocol; the concrete,
gate-required backend is :class:`SQLiteLongTermStore`, a local-first, zero-server
adapter built on the stdlib :mod:`sqlite3` module.

Design rules (binding, from the Phase-4 plan):

* **Local-first, zero-server.** SQLite is the concrete backend for both
  ``":memory:"`` (tests) and a file path (production). No daemon, no network.
* **Parametrized SQL only.** Every value reaches SQLite through a ``?``
  placeholder — user/agent text is never interpolated into a statement — so the
  store is injection-safe by construction.
* **Typed rows.** Reads return pydantic v2 models (:class:`Fact`,
  :class:`TaskRow`, :class:`AuditRow`), never raw tuples, so callers get a stable
  schema and validation at the boundary.
* **Postgres is a flagged adapter swap.** :class:`PostgresLongTermStore` exists
  as a documented stub that **lazy-imports** ``psycopg`` and raises a clear
  "configure Postgres (Phase 6)" error if used. ``psycopg`` is *not* a project
  dependency; the stub never runs in the gate.

Retrieval (:meth:`SQLiteLongTermStore.query_facts`) is a simple case-insensitive
substring match (SQL ``LIKE``) — deterministic, dependency-free, and good enough
for the long-term recall path. Semantic similarity lives in the vector store
(Phase-4 Stage 1B), not here.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from friday.errors import FridayError

# In-memory SQLite database identifier — the default, used by every test.
_MEMORY_PATH = ":memory:"

# Default page size for history/query reads when a caller does not specify one.
DEFAULT_LIMIT = 10

# Guidance surfaced when the Postgres adapter is used before Phase 6 wires it up.
_POSTGRES_PHASE6_NOTE = (
    "PostgresLongTermStore is not configured: the Postgres/pgvector backend is a "
    "Phase 6 adapter swap. Use SQLiteLongTermStore (the local-first default) for "
    "now, or configure Postgres (Phase 6) — install `psycopg` and provision the "
    "database — before selecting this backend."
)


def _utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string (second precision).

    A string column keeps the schema portable and the value human-readable in
    ``sqlite3`` dumps; callers that need a ``datetime`` can parse it back.
    """
    return datetime.now(UTC).isoformat()


# --------------------------------------------------------------------------- #
# Typed row models
# --------------------------------------------------------------------------- #
class Fact(BaseModel):
    """A durable fact the assistant has been asked to remember.

    ``source_id`` ties the fact back to where it came from (a document, a chat
    turn, an agent) so a grounded answer can cite it. ``sensitive`` marks data
    that must never be auto-persisted without explicit owner confirmation
    (the write-consent policy enforced upstream in the orchestrator).
    """

    id: int
    text: str
    source_id: str
    sensitive: bool = False
    created_at: str


class TaskRow(BaseModel):
    """A record of a task the assistant executed, for history/recall."""

    id: int
    intent: str
    summary: str
    ok: bool
    created_at: str


class AuditRow(BaseModel):
    """A single audit-trail entry for one orchestrator/agent step."""

    id: int
    step: str
    ok: bool
    detail: str
    created_at: str


# --------------------------------------------------------------------------- #
# Protocol
# --------------------------------------------------------------------------- #
@runtime_checkable
class LongTermStore(Protocol):
    """Structural contract for FRIDAY's durable memory backend.

    Implementations persist facts, tasks, and audit records and support a
    ``forget`` operation that removes facts matching a query (the user-facing
    "forget what you know about X" command).
    """

    def add_fact(
        self, text: str, source_id: str, sensitive: bool = False
    ) -> Fact:
        """Persist a fact and return the stored row (with its assigned id)."""
        ...

    def query_facts(self, query: str, limit: int = DEFAULT_LIMIT) -> list[Fact]:
        """Return up to ``limit`` facts whose text contains ``query`` (CI)."""
        ...

    def all_facts(self, limit: int = DEFAULT_LIMIT) -> list[Fact]:
        """Return up to ``limit`` facts, newest first (for bulk export)."""
        ...

    def add_task(self, intent: str, summary: str, ok: bool) -> TaskRow:
        """Persist a task record and return the stored row."""
        ...

    def task_history(self, limit: int = DEFAULT_LIMIT) -> list[TaskRow]:
        """Return up to ``limit`` task records, most recent first."""
        ...

    def add_audit(self, step: str, ok: bool, detail: str) -> AuditRow:
        """Persist an audit record and return the stored row."""
        ...

    def forget(self, query: str) -> int:
        """Remove every fact whose text contains ``query`` (CI); return count."""
        ...


# --------------------------------------------------------------------------- #
# SQLite implementation
# --------------------------------------------------------------------------- #
class SQLiteLongTermStore:
    """Local-first long-term store backed by stdlib :mod:`sqlite3`.

    A single long-lived connection is held for the life of the instance. It is
    opened with ``check_same_thread=False`` so the store can be shared across the
    threads the test suite and the app's executor use; writes are serialised by
    SQLite's own locking, which is sufficient for FRIDAY's low write volume.

    Tables are created idempotently in :meth:`init_schema` (``CREATE TABLE IF NOT
    EXISTS``), so constructing a store over an existing file is safe and never
    clobbers prior data.

    Args:
        path: A filesystem path for a durable database, or ``":memory:"``
            (the default) for an ephemeral in-process database used by tests.
    """

    def __init__(self, path: str = _MEMORY_PATH) -> None:
        self._path = path
        # ``check_same_thread=False``: the connection may be touched from more
        # than one thread (test fixtures, the app executor). SQLite serialises
        # access internally; we never share a cursor across threads.
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self.init_schema()

    def init_schema(self) -> None:
        """Create the ``facts``, ``tasks``, and ``audit`` tables if absent.

        Idempotent: safe to call repeatedly and safe against a database file
        that already holds these tables. ``sensitive``/``ok`` are stored as
        ``INTEGER`` (0/1) — SQLite has no native boolean — and converted back to
        ``bool`` when rows are read.
        """
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                source_id TEXT NOT NULL,
                sensitive INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                intent TEXT NOT NULL,
                summary TEXT NOT NULL,
                ok INTEGER NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS audit (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                step TEXT NOT NULL,
                ok INTEGER NOT NULL,
                detail TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )
        self._conn.commit()

    # -- facts -------------------------------------------------------------- #
    def add_fact(
        self, text: str, source_id: str, sensitive: bool = False
    ) -> Fact:
        """Insert a fact (parametrized) and return the stored :class:`Fact`."""
        created_at = _utc_now_iso()
        cursor = self._conn.execute(
            "INSERT INTO facts (text, source_id, sensitive, created_at) "
            "VALUES (?, ?, ?, ?)",
            (text, source_id, int(sensitive), created_at),
        )
        self._conn.commit()
        fact_id = int(cursor.lastrowid or 0)
        return Fact(
            id=fact_id,
            text=text,
            source_id=source_id,
            sensitive=sensitive,
            created_at=created_at,
        )

    def query_facts(self, query: str, limit: int = DEFAULT_LIMIT) -> list[Fact]:
        """Return up to ``limit`` facts whose text contains ``query`` (CI).

        Matching uses a parametrized SQL ``LIKE`` with the query wrapped in
        ``%`` wildcards, so ``query`` is always a literal substring — SQL
        metacharacters in it have no effect. ``LIKE`` is case-insensitive for
        ASCII in SQLite. Results are newest-first. A non-positive ``limit``
        yields an empty list.
        """
        if limit <= 0:
            return []
        like = self._contains(query)
        rows = self._conn.execute(
            "SELECT id, text, source_id, sensitive, created_at "
            "FROM facts WHERE text LIKE ? ESCAPE '\\' "
            "ORDER BY id DESC LIMIT ?",
            (like, limit),
        ).fetchall()
        return [self._row_to_fact(row) for row in rows]

    def all_facts(self, limit: int = DEFAULT_LIMIT) -> list[Fact]:
        """Return up to ``limit`` facts, newest first (a non-positive limit -> []).

        Unfiltered bulk read for second-brain export; results are newest-first to
        match :meth:`query_facts`.
        """
        if limit <= 0:
            return []
        rows = self._conn.execute(
            "SELECT id, text, source_id, sensitive, created_at "
            "FROM facts ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [self._row_to_fact(row) for row in rows]

    def forget(self, query: str) -> int:
        """Delete every fact whose text contains ``query`` (CI); return count.

        Parametrized like :meth:`query_facts`, so ``query`` is a literal
        substring. Returns the number of rows removed (0 when nothing matched).
        """
        like = self._contains(query)
        cursor = self._conn.execute(
            "DELETE FROM facts WHERE text LIKE ? ESCAPE '\\'",
            (like,),
        )
        self._conn.commit()
        return cursor.rowcount

    # -- tasks -------------------------------------------------------------- #
    def add_task(self, intent: str, summary: str, ok: bool) -> TaskRow:
        """Insert a task record (parametrized) and return the stored row."""
        created_at = _utc_now_iso()
        cursor = self._conn.execute(
            "INSERT INTO tasks (intent, summary, ok, created_at) "
            "VALUES (?, ?, ?, ?)",
            (intent, summary, int(ok), created_at),
        )
        self._conn.commit()
        return TaskRow(
            id=int(cursor.lastrowid or 0),
            intent=intent,
            summary=summary,
            ok=ok,
            created_at=created_at,
        )

    def task_history(self, limit: int = DEFAULT_LIMIT) -> list[TaskRow]:
        """Return up to ``limit`` task records, most recent first."""
        if limit <= 0:
            return []
        rows = self._conn.execute(
            "SELECT id, intent, summary, ok, created_at "
            "FROM tasks ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [self._row_to_task(row) for row in rows]

    # -- audit -------------------------------------------------------------- #
    def add_audit(self, step: str, ok: bool, detail: str) -> AuditRow:
        """Insert an audit record (parametrized) and return the stored row."""
        created_at = _utc_now_iso()
        cursor = self._conn.execute(
            "INSERT INTO audit (step, ok, detail, created_at) "
            "VALUES (?, ?, ?, ?)",
            (step, int(ok), detail, created_at),
        )
        self._conn.commit()
        return AuditRow(
            id=int(cursor.lastrowid or 0),
            step=step,
            ok=ok,
            detail=detail,
            created_at=created_at,
        )

    def audit_history(self, limit: int = DEFAULT_LIMIT) -> list[AuditRow]:
        """Return up to ``limit`` audit records, most recent first."""
        if limit <= 0:
            return []
        rows = self._conn.execute(
            "SELECT id, step, ok, detail, created_at "
            "FROM audit ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [self._row_to_audit(row) for row in rows]

    # -- helpers ------------------------------------------------------------ #
    @staticmethod
    def _contains(query: str) -> str:
        """Wrap ``query`` as a ``LIKE`` substring pattern, escaping wildcards.

        ``%`` and ``_`` (and the escape char ``\\``) in ``query`` are escaped so
        they match literally rather than acting as wildcards; the result is then
        surrounded by ``%`` so the pattern means "contains ``query``". Used with
        ``ESCAPE '\\'`` in the SQL.
        """
        escaped = (
            query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        )
        return f"%{escaped}%"

    @staticmethod
    def _row_to_fact(row: sqlite3.Row) -> Fact:
        return Fact(
            id=int(row["id"]),
            text=str(row["text"]),
            source_id=str(row["source_id"]),
            sensitive=bool(row["sensitive"]),
            created_at=str(row["created_at"]),
        )

    @staticmethod
    def _row_to_task(row: sqlite3.Row) -> TaskRow:
        return TaskRow(
            id=int(row["id"]),
            intent=str(row["intent"]),
            summary=str(row["summary"]),
            ok=bool(row["ok"]),
            created_at=str(row["created_at"]),
        )

    @staticmethod
    def _row_to_audit(row: sqlite3.Row) -> AuditRow:
        return AuditRow(
            id=int(row["id"]),
            step=str(row["step"]),
            ok=bool(row["ok"]),
            detail=str(row["detail"]),
            created_at=str(row["created_at"]),
        )


# --------------------------------------------------------------------------- #
# Postgres adapter stub (Phase 6)
# --------------------------------------------------------------------------- #
class PostgresLongTermStore:
    """Flagged, not-yet-wired Postgres adapter — a documented Phase 6 swap.

    This class exists to make the adapter seam explicit and to fail loudly if a
    misconfigured deployment selects the Postgres backend before it is wired up.
    The ``psycopg`` driver is **lazy-imported** inside :meth:`_connect` and is
    *not* a project dependency; every public method raises a clear
    :class:`~friday.errors.FridayError` pointing at Phase 6.

    Constructing the object is cheap and never imports ``psycopg`` — only an
    actual operation attempts the (deferred) connection, so importing this module
    stays dependency-free for the gate.
    """

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn

    def _connect(self) -> object:
        """Lazy-import ``psycopg`` and raise the Phase-6 guidance error.

        ``psycopg`` is imported here (not at module top) so the optional driver
        is never required to import this module. Whether or not the import
        succeeds, the backend is not provisioned, so we always raise the clear
        Phase-6 error rather than half-connecting.
        """
        try:
            import psycopg  # type: ignore[import-not-found]  # noqa: F401, PLC0415
        except ImportError as exc:  # pragma: no cover - psycopg not installed
            raise FridayError(_POSTGRES_PHASE6_NOTE) from exc
        raise FridayError(_POSTGRES_PHASE6_NOTE)

    def add_fact(
        self, text: str, source_id: str, sensitive: bool = False
    ) -> Fact:
        self._connect()
        raise FridayError(_POSTGRES_PHASE6_NOTE)

    def query_facts(self, query: str, limit: int = DEFAULT_LIMIT) -> list[Fact]:
        self._connect()
        raise FridayError(_POSTGRES_PHASE6_NOTE)

    def all_facts(self, limit: int = DEFAULT_LIMIT) -> list[Fact]:
        self._connect()
        raise FridayError(_POSTGRES_PHASE6_NOTE)

    def add_task(self, intent: str, summary: str, ok: bool) -> TaskRow:
        self._connect()
        raise FridayError(_POSTGRES_PHASE6_NOTE)

    def task_history(self, limit: int = DEFAULT_LIMIT) -> list[TaskRow]:
        self._connect()
        raise FridayError(_POSTGRES_PHASE6_NOTE)

    def add_audit(self, step: str, ok: bool, detail: str) -> AuditRow:
        self._connect()
        raise FridayError(_POSTGRES_PHASE6_NOTE)

    def forget(self, query: str) -> int:
        self._connect()
        raise FridayError(_POSTGRES_PHASE6_NOTE)
