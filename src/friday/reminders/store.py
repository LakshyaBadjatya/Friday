"""Local-first reminder store behind a ``ReminderStore`` protocol (Tier 1).

FRIDAY's durable reminder layer: reminders the assistant has been asked to keep,
each with optional due date and simple recurrence. The contract is the
:class:`ReminderStore` protocol; the concrete, gate-required backend is
:class:`SQLiteReminderStore`, a local-first, zero-server adapter built on the
stdlib :mod:`sqlite3` module (mirroring
:class:`friday.memory.long_term.SQLiteLongTermStore`).

Design rules (binding):

* **Local-first, zero-server.** SQLite is the concrete backend for both
  ``":memory:"`` (tests) and a file path (production). No daemon, no network.
* **Parametrized SQL only.** Every value reaches SQLite through a ``?``
  placeholder — user/agent text is never interpolated into a statement — so the
  store is injection-safe by construction.
* **Idempotent schema.** The table is created with ``CREATE TABLE IF NOT
  EXISTS`` so constructing a store over an existing file never clobbers data.
* **Thread-safe by construction (file paths).** A *connection-per-call* is opened
  for a filesystem-backed database, so the store may be shared across the threads
  the test suite and the app executor use without a pinned, single-thread
  connection. An in-memory database (``":memory:"``) keeps a single shared
  connection because each new connection would otherwise see an empty database.
* **Clock injectable.** ``created_at`` is sourced from an injected
  ``clock() -> float`` (epoch seconds), never the wall clock, so timestamps are
  deterministic in tests. ``due()`` is driven entirely by the ISO timestamp the
  caller passes, again never reading the wall clock.

**Recurrence rule (kept deliberately simple).** A reminder may declare a
``recurrence`` of ``"daily"`` or ``"weekly"``. On :meth:`complete`, instead of
flipping to ``done`` the reminder *stays open* and its ``due_at`` is rolled
forward by exactly one period (``+1 day`` / ``+7 days``) from its current
``due_at``. This is a fixed-step roll (it does not skip missed occurrences to
"catch up" to now) — the simplest rule that keeps a recurring reminder alive. A
recurring reminder with **no** ``due_at`` anchor cannot be rolled forward, so it
falls back to completing as a one-shot (``done``).
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel

# In-memory SQLite database identifier — the ephemeral default for some callers.
_MEMORY_PATH = ":memory:"

# Recurrence period -> the time step a completed occurrence rolls forward by.
_RECURRENCE_STEP: dict[str, timedelta] = {
    "daily": timedelta(days=1),
    "weekly": timedelta(weeks=1),
}

ReminderStatus = Literal["open", "done"]


# --------------------------------------------------------------------------- #
# Typed row model
# --------------------------------------------------------------------------- #
class Reminder(BaseModel):
    """A durable reminder/task row.

    ``due_at`` and ``recurrence`` are optional: an undated reminder is a plain
    "remember to" note that never fires through :meth:`SQLiteReminderStore.due`,
    and a ``None`` recurrence is a one-shot. ``status`` is ``"open"`` until the
    reminder is completed (a recurring reminder rolls forward and stays open).
    ``created_at`` is an ISO-8601 string derived from the store's injected clock.
    """

    id: int
    text: str
    due_at: str | None = None
    recurrence: str | None = None
    status: ReminderStatus = "open"
    created_at: str


# --------------------------------------------------------------------------- #
# Protocol
# --------------------------------------------------------------------------- #
@runtime_checkable
class ReminderStore(Protocol):
    """Structural contract for FRIDAY's durable reminder backend."""

    def add(
        self,
        text: str,
        due_at: str | None = None,
        recurrence: str | None = None,
    ) -> Reminder:
        """Persist a reminder and return the stored row (with its assigned id)."""
        ...

    def list_reminders(
        self, status: Literal["open", "all"] = "open"
    ) -> list[Reminder]:
        """Return reminders soonest-due first; ``"all"`` includes completed."""
        ...

    def due(self, now_iso: str) -> list[Reminder]:
        """Return open reminders whose ``due_at`` is at or before ``now_iso``."""
        ...

    def complete(self, reminder_id: int) -> bool:
        """Complete a reminder (recurring -> roll forward); ``False`` if absent."""
        ...

    def delete(self, reminder_id: int) -> int:
        """Delete a reminder by id; return the number of rows removed (0/1)."""
        ...


# --------------------------------------------------------------------------- #
# SQLite implementation
# --------------------------------------------------------------------------- #
class SQLiteReminderStore:
    """Local-first reminder store backed by stdlib :mod:`sqlite3`.

    Args:
        path: A filesystem path for a durable database, or ``":memory:"`` for an
            ephemeral in-process database. For a file path a fresh connection is
            opened per call (thread-safe); ``":memory:"`` keeps one shared
            connection for the life of the instance (a new connection would see
            an empty database).
        clock: A zero-arg callable returning the current time as epoch seconds.
            Used only for ``created_at``; injected so timestamps are
            deterministic in tests. Defaults to the system wall clock.
    """

    def __init__(
        self,
        path: str = _MEMORY_PATH,
        *,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._path = path
        if clock is None:
            import time as _time  # local import: keep the wall clock off the

            self._clock: Callable[[], float] = _time.time  # tested default path
        else:
            self._clock = clock
        # An in-memory database cannot use connection-per-call: each new
        # connection would open a *separate* empty database. A file path can, so
        # the store is shareable across threads without a pinned connection.
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
        """Create the ``reminders`` table if absent (idempotent).

        ``status`` is stored as text (``"open"``/``"done"``); ``due_at`` and
        ``recurrence`` are nullable. Safe to call repeatedly and against an
        existing file.
        """
        conn = self._connect()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS reminders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT NOT NULL,
                    due_at TEXT,
                    recurrence TEXT,
                    status TEXT NOT NULL DEFAULT 'open',
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.commit()
        finally:
            self._close(conn)

    # -- writes ------------------------------------------------------------ #
    def add(
        self,
        text: str,
        due_at: str | None = None,
        recurrence: str | None = None,
    ) -> Reminder:
        """Insert a reminder (parametrized) and return the stored row."""
        created_at = self._now_iso()
        conn = self._connect()
        try:
            cursor = conn.execute(
                "INSERT INTO reminders (text, due_at, recurrence, status, created_at) "
                "VALUES (?, ?, ?, 'open', ?)",
                (text, due_at, recurrence, created_at),
            )
            conn.commit()
            reminder_id = int(cursor.lastrowid or 0)
        finally:
            self._close(conn)
        return Reminder(
            id=reminder_id,
            text=text,
            due_at=due_at,
            recurrence=recurrence,
            status="open",
            created_at=created_at,
        )

    def complete(self, reminder_id: int) -> bool:
        """Complete a reminder: one-shot -> ``done``; recurring -> roll forward.

        Returns ``True`` when a reminder with ``reminder_id`` existed and was
        updated, ``False`` when no such (open) row was found. A recurring
        reminder with a ``due_at`` anchor stays ``open`` and its ``due_at``
        advances by one period; a recurring reminder without a ``due_at`` (no
        anchor to roll) falls back to completing as a one-shot.
        """
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT due_at, recurrence FROM reminders "
                "WHERE id = ? AND status = 'open'",
                (reminder_id,),
            ).fetchone()
            if row is None:
                return False

            due_at = row["due_at"]
            recurrence = row["recurrence"]
            next_due = self._roll_forward(due_at, recurrence)
            if next_due is not None:
                # Recurring + anchored: keep it open, advance the due date.
                conn.execute(
                    "UPDATE reminders SET due_at = ? WHERE id = ?",
                    (next_due, reminder_id),
                )
            else:
                # One-shot, or recurring-but-undated: mark it done.
                conn.execute(
                    "UPDATE reminders SET status = 'done' WHERE id = ?",
                    (reminder_id,),
                )
            conn.commit()
            return True
        finally:
            self._close(conn)

    def delete(self, reminder_id: int) -> int:
        """Delete a reminder by id; return the number of rows removed (0 or 1)."""
        conn = self._connect()
        try:
            cursor = conn.execute(
                "DELETE FROM reminders WHERE id = ?", (reminder_id,)
            )
            conn.commit()
            return cursor.rowcount
        finally:
            self._close(conn)

    # -- reads ------------------------------------------------------------- #
    def list_reminders(
        self, status: Literal["open", "all"] = "open"
    ) -> list[Reminder]:
        """Return reminders soonest-due first, then by creation order.

        ``status="open"`` (the default) returns only open reminders;
        ``status="all"`` includes completed ones. Ordering: reminders with a
        ``due_at`` sort ahead of undated ones (``due_at IS NULL`` last), earliest
        ``due_at`` first; ties and undated rows fall back to insertion order
        (ascending id), which is creation order.
        """
        conn = self._connect()
        try:
            if status == "all":
                rows = conn.execute(
                    "SELECT id, text, due_at, recurrence, status, created_at "
                    "FROM reminders "
                    "ORDER BY due_at IS NULL, due_at ASC, id ASC"
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT id, text, due_at, recurrence, status, created_at "
                    "FROM reminders WHERE status = 'open' "
                    "ORDER BY due_at IS NULL, due_at ASC, id ASC"
                ).fetchall()
            return [self._row_to_reminder(row) for row in rows]
        finally:
            self._close(conn)

    def due(self, now_iso: str) -> list[Reminder]:
        """Return open reminders whose ``due_at`` is at or before ``now_iso``.

        Driven entirely by the passed ``now_iso`` timestamp — the store never
        reads the wall clock here. ISO-8601 timestamps compare correctly as
        lexicographic strings when uniformly formatted, so the filter is a
        parametrized string comparison. Undated reminders are never due.
        """
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT id, text, due_at, recurrence, status, created_at "
                "FROM reminders "
                "WHERE status = 'open' AND due_at IS NOT NULL AND due_at <= ? "
                "ORDER BY due_at ASC, id ASC",
                (now_iso,),
            ).fetchall()
            return [self._row_to_reminder(row) for row in rows]
        finally:
            self._close(conn)

    # -- helpers ----------------------------------------------------------- #
    def _now_iso(self) -> str:
        """Current time as an ISO-8601 UTC string, from the injected clock."""
        from datetime import UTC

        return datetime.fromtimestamp(self._clock(), tz=UTC).isoformat()

    @staticmethod
    def _roll_forward(due_at: str | None, recurrence: str | None) -> str | None:
        """Compute the next ``due_at`` for a recurring reminder, or ``None``.

        Returns ``None`` (meaning "do not roll; complete as one-shot") when there
        is no recurrence, no anchor ``due_at``, or an unrecognized recurrence
        keyword. Otherwise advances ``due_at`` by exactly one period and returns
        the new ISO timestamp (preserving the original offset/format).
        """
        if recurrence is None or due_at is None:
            return None
        step = _RECURRENCE_STEP.get(recurrence.strip().lower())
        if step is None:
            return None
        try:
            anchor = datetime.fromisoformat(due_at)
        except ValueError:
            return None
        return (anchor + step).isoformat()

    @staticmethod
    def _row_to_reminder(row: sqlite3.Row) -> Reminder:
        return Reminder(
            id=int(row["id"]),
            text=str(row["text"]),
            due_at=None if row["due_at"] is None else str(row["due_at"]),
            recurrence=(
                None if row["recurrence"] is None else str(row["recurrence"])
            ),
            status="done" if str(row["status"]) == "done" else "open",
            created_at=str(row["created_at"]),
        )

    def _close(self, conn: sqlite3.Connection) -> None:
        """Close a per-call connection; leave the shared (memory) one open."""
        if conn is not self._shared:
            conn.close()
