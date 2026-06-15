"""Local-first trigger store behind a SQLite backend (Tier 1 scheduler).

FRIDAY's durable scheduler layer: time-based triggers the assistant fires to run
named actions. The concrete, gate-required backend is :class:`SQLiteTriggerStore`,
a local-first, zero-server adapter built on the stdlib :mod:`sqlite3` module
(mirroring :class:`friday.reminders.store.SQLiteReminderStore`).

Design rules (binding):

* **Local-first, zero-server.** SQLite is the concrete backend for both
  ``":memory:"`` (tests) and a file path (production). No daemon, no network.
* **Parametrized SQL only.** Every value reaches SQLite through a ``?``
  placeholder, so the store is injection-safe by construction.
* **Idempotent schema.** ``CREATE TABLE IF NOT EXISTS`` so constructing a store
  over an existing file never clobbers data.
* **Thread-safe by construction (file paths).** A *connection-per-call* is opened
  for a filesystem-backed database (shareable across threads); an in-memory
  database keeps a single shared connection (a new connection would otherwise see
  an empty database).
* **Clock injectable.** :meth:`due` is driven entirely by the ``now`` datetime
  the caller passes — the store never reads the wall clock.

The ``spec`` column's meaning depends on ``kind``: ``interval`` -> seconds (int
as str); ``once`` -> ISO datetime; ``daily`` -> ``"HH:MM"``; ``weekly`` ->
``"DOW HH:MM"`` (``DOW`` in ``MON``..``SUN``). The store keeps ``spec`` opaque;
the :mod:`friday.scheduler.engine` module interprets it.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from typing import Literal

from pydantic import BaseModel

# In-memory SQLite database identifier — the ephemeral default for some callers.
_MEMORY_PATH = ":memory:"

TriggerKind = Literal["interval", "once", "daily", "weekly"]


# --------------------------------------------------------------------------- #
# Typed row model
# --------------------------------------------------------------------------- #
class Trigger(BaseModel):
    """A durable scheduled-trigger row.

    ``spec`` is interpreted per ``kind`` (see module docstring). ``next_run`` is
    the ISO timestamp of the next time the trigger should fire (``None`` for a
    spent ``once`` or an un-computable spec); ``last_run`` is the ISO timestamp of
    the most recent fire (``None`` until first fired). ``enabled`` gates whether
    the trigger participates in :meth:`SQLiteTriggerStore.due`.
    """

    id: int
    name: str
    kind: TriggerKind
    spec: str
    action: str
    enabled: bool = True
    next_run: str | None = None
    last_run: str | None = None


# --------------------------------------------------------------------------- #
# SQLite implementation
# --------------------------------------------------------------------------- #
class SQLiteTriggerStore:
    """Local-first trigger store backed by stdlib :mod:`sqlite3`.

    Args:
        path: A filesystem path for a durable database, or ``":memory:"`` for an
            ephemeral in-process database. For a file path a fresh connection is
            opened per call (thread-safe); ``":memory:"`` keeps one shared
            connection for the life of the instance.
    """

    def __init__(self, path: str = _MEMORY_PATH) -> None:
        self._path = path
        # An in-memory database cannot use connection-per-call: each new
        # connection would open a *separate* empty database. A file path can.
        self._shared: sqlite3.Connection | None = None
        if path == _MEMORY_PATH:
            self._shared = sqlite3.connect(path, check_same_thread=False)
            self._shared.row_factory = sqlite3.Row
        self.init_schema()

    # -- connection management --------------------------------------------- #
    def _connect(self) -> sqlite3.Connection:
        """Return the shared (memory) connection or a fresh per-call one (file)."""
        if self._shared is not None:
            return self._shared
        conn = sqlite3.connect(self._path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_schema(self) -> None:
        """Create the ``triggers`` table if absent (idempotent)."""
        conn = self._connect()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS triggers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    spec TEXT NOT NULL,
                    action TEXT NOT NULL,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    next_run TEXT,
                    last_run TEXT
                )
                """
            )
            conn.commit()
        finally:
            self._close(conn)

    # -- writes ------------------------------------------------------------ #
    def add(
        self,
        *,
        name: str,
        kind: TriggerKind,
        spec: str,
        action: str,
        enabled: bool = True,
        next_run: str | None = None,
        last_run: str | None = None,
    ) -> Trigger:
        """Insert a trigger (parametrized) and return the stored row."""
        conn = self._connect()
        try:
            cursor = conn.execute(
                "INSERT INTO triggers "
                "(name, kind, spec, action, enabled, next_run, last_run) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (name, kind, spec, action, 1 if enabled else 0, next_run, last_run),
            )
            conn.commit()
            trigger_id = int(cursor.lastrowid or 0)
        finally:
            self._close(conn)
        return Trigger(
            id=trigger_id,
            name=name,
            kind=kind,
            spec=spec,
            action=action,
            enabled=enabled,
            next_run=next_run,
            last_run=last_run,
        )

    def update(self, trigger: Trigger) -> bool:
        """Persist ``enabled``/``next_run``/``last_run`` (and name/spec/action).

        Returns ``True`` when a row with ``trigger.id`` existed and was updated,
        ``False`` otherwise.
        """
        conn = self._connect()
        try:
            cursor = conn.execute(
                "UPDATE triggers SET name = ?, kind = ?, spec = ?, action = ?, "
                "enabled = ?, next_run = ?, last_run = ? WHERE id = ?",
                (
                    trigger.name,
                    trigger.kind,
                    trigger.spec,
                    trigger.action,
                    1 if trigger.enabled else 0,
                    trigger.next_run,
                    trigger.last_run,
                    trigger.id,
                ),
            )
            conn.commit()
            return cursor.rowcount > 0
        finally:
            self._close(conn)

    def set_enabled(self, trigger_id: int, enabled: bool) -> bool:
        """Toggle a trigger's ``enabled`` flag; ``False`` when no such row."""
        conn = self._connect()
        try:
            cursor = conn.execute(
                "UPDATE triggers SET enabled = ? WHERE id = ?",
                (1 if enabled else 0, trigger_id),
            )
            conn.commit()
            return cursor.rowcount > 0
        finally:
            self._close(conn)

    def delete(self, trigger_id: int) -> int:
        """Delete a trigger by id; return the number of rows removed (0 or 1)."""
        conn = self._connect()
        try:
            cursor = conn.execute(
                "DELETE FROM triggers WHERE id = ?", (trigger_id,)
            )
            conn.commit()
            return cursor.rowcount
        finally:
            self._close(conn)

    # -- reads ------------------------------------------------------------- #
    def get(self, trigger_id: int) -> Trigger | None:
        """Return the trigger with ``trigger_id`` or ``None`` when absent."""
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT id, name, kind, spec, action, enabled, next_run, last_run "
                "FROM triggers WHERE id = ?",
                (trigger_id,),
            ).fetchone()
            return None if row is None else self._row_to_trigger(row)
        finally:
            self._close(conn)

    def list_triggers(self) -> list[Trigger]:
        """Return every trigger in insertion (ascending id) order."""
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT id, name, kind, spec, action, enabled, next_run, last_run "
                "FROM triggers ORDER BY id ASC"
            ).fetchall()
            return [self._row_to_trigger(row) for row in rows]
        finally:
            self._close(conn)

    def due(self, now: datetime) -> list[Trigger]:
        """Return enabled triggers whose ``next_run`` is at or before ``now``.

        Driven entirely by the passed ``now`` datetime — the store never reads
        the wall clock. ``now`` is normalized to an ISO string for the comparison;
        ISO-8601 timestamps compare correctly as lexicographic strings when
        uniformly formatted. Triggers with a ``NULL`` ``next_run`` (e.g. a spent
        ``once``) are never due.
        """
        now_iso = now.isoformat()
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT id, name, kind, spec, action, enabled, next_run, last_run "
                "FROM triggers "
                "WHERE enabled = 1 AND next_run IS NOT NULL AND next_run <= ? "
                "ORDER BY next_run ASC, id ASC",
                (now_iso,),
            ).fetchall()
            return [self._row_to_trigger(row) for row in rows]
        finally:
            self._close(conn)

    # -- helpers ----------------------------------------------------------- #
    @staticmethod
    def _row_to_trigger(row: sqlite3.Row) -> Trigger:
        # The column is constrained to the four literals on write; pydantic
        # re-validates ``kind`` against ``TriggerKind`` at construction, so an
        # out-of-band stored value surfaces as a validation error (a bug).
        return Trigger(
            id=int(row["id"]),
            name=str(row["name"]),
            kind=str(row["kind"]),
            spec=str(row["spec"]),
            action=str(row["action"]),
            enabled=bool(row["enabled"]),
            next_run=None if row["next_run"] is None else str(row["next_run"]),
            last_run=None if row["last_run"] is None else str(row["last_run"]),
        )

    def _close(self, conn: sqlite3.Connection) -> None:
        """Close a per-call connection; leave the shared (memory) one open."""
        if conn is not self._shared:
            conn.close()
