"""Local-first meeting-notes store behind a SQLite backend (Tier 1 meetings).

FRIDAY's durable meeting-notes layer: the structured :class:`MeetingNotes`
produced by :class:`~friday.meetings.capture.MeetingCapture`, persisted so a
captured meeting is listable, retrievable, and deletable. The concrete,
gate-required backend is :class:`SQLiteMeetingStore`, a local-first, zero-server
adapter built on the stdlib :mod:`sqlite3` module (mirroring
:class:`friday.protocols.store.SQLiteProtocolStore` and
:class:`friday.reminders.store.SQLiteReminderStore`).

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
* **Action items as JSON.** The ``action_items`` list is serialized to a single
  JSON ``TEXT`` column, keeping the schema flat while preserving the ordered list
  on read.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Protocol, runtime_checkable

from friday.meetings.capture import MeetingNotes

# In-memory SQLite database identifier — the ephemeral default for some callers.
_MEMORY_PATH = ":memory:"


@runtime_checkable
class MeetingStore(Protocol):
    """Structural contract for FRIDAY's durable meeting-notes backend."""

    def add(self, notes: MeetingNotes) -> MeetingNotes:
        """Persist ``notes`` and return the stored row (with its assigned id)."""
        ...

    def list_meetings(self, limit: int = ...) -> list[MeetingNotes]:
        """Return stored meetings, most-recent first, capped at ``limit``."""
        ...

    def get(self, meeting_id: int) -> MeetingNotes | None:
        """Return the meeting with ``meeting_id`` or ``None`` when absent."""
        ...

    def delete(self, meeting_id: int) -> int:
        """Delete a meeting by id; return the number of rows removed (0 or 1)."""
        ...


class SQLiteMeetingStore:
    """Local-first meeting-notes store backed by stdlib :mod:`sqlite3`.

    Args:
        path: A filesystem path for a durable database, or ``":memory:"`` for an
            ephemeral in-process database. For a file path a fresh connection is
            opened per call (thread-safe); ``":memory:"`` keeps one shared
            connection for the life of the instance (a new connection would see
            an empty database).
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
        """Create the ``meetings`` table if absent (idempotent).

        ``action_items`` holds the JSON-serialized list; the rest are plain text.
        Safe to call repeatedly and against an existing file.
        """
        conn = self._connect()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS meetings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    transcript TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    action_items TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.commit()
        finally:
            self._close(conn)

    # -- writes ------------------------------------------------------------ #
    def add(self, notes: MeetingNotes) -> MeetingNotes:
        """Insert meeting notes (parametrized; action items as JSON).

        Returns a copy of ``notes`` with the store-assigned ``id`` populated; the
        passed ``notes.id`` is ignored (the store owns id assignment).
        """
        action_items_json = json.dumps(list(notes.action_items))
        conn = self._connect()
        try:
            cursor = conn.execute(
                "INSERT INTO meetings "
                "(title, transcript, summary, action_items, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    notes.title,
                    notes.transcript,
                    notes.summary,
                    action_items_json,
                    notes.created_at,
                ),
            )
            conn.commit()
            meeting_id = int(cursor.lastrowid or 0)
        finally:
            self._close(conn)
        return notes.model_copy(update={"id": meeting_id})

    def delete(self, meeting_id: int) -> int:
        """Delete a meeting by id; return the number of rows removed (0 or 1)."""
        conn = self._connect()
        try:
            cursor = conn.execute(
                "DELETE FROM meetings WHERE id = ?", (meeting_id,)
            )
            conn.commit()
            return cursor.rowcount
        finally:
            self._close(conn)

    # -- reads ------------------------------------------------------------- #
    def get(self, meeting_id: int) -> MeetingNotes | None:
        """Return the meeting with ``meeting_id`` or ``None`` when absent."""
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT id, title, transcript, summary, action_items, created_at "
                "FROM meetings WHERE id = ?",
                (meeting_id,),
            ).fetchone()
            return None if row is None else self._row_to_notes(row)
        finally:
            self._close(conn)

    def list_meetings(self, limit: int = 50) -> list[MeetingNotes]:
        """Return stored meetings, most-recent first (descending id), up to ``limit``."""
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT id, title, transcript, summary, action_items, created_at "
                "FROM meetings ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [self._row_to_notes(row) for row in rows]
        finally:
            self._close(conn)

    # -- helpers ----------------------------------------------------------- #
    @staticmethod
    def _load_action_items(raw: str) -> list[str]:
        """Parse the JSON action-items column back into a list of strings."""
        data = json.loads(raw)
        return [str(item) for item in data]

    def _row_to_notes(self, row: sqlite3.Row) -> MeetingNotes:
        return MeetingNotes(
            id=int(row["id"]),
            title=str(row["title"]),
            transcript=str(row["transcript"]),
            summary=str(row["summary"]),
            action_items=self._load_action_items(str(row["action_items"])),
            created_at=str(row["created_at"]),
        )

    def _close(self, conn: sqlite3.Connection) -> None:
        """Close a per-call connection; leave the shared (memory) one open."""
        if conn is not self._shared:
            conn.close()
