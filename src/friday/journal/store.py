"""Local-first journal store behind a SQLite backend (Tier 2 journaling).

FRIDAY's durable journal layer: the structured :class:`JournalEntry` produced by
:class:`~friday.journal.service.JournalService`, persisted so a built day is
listable and retrievable. The concrete backend is :class:`SQLiteJournalStore`, a
local-first, zero-server adapter built on the stdlib :mod:`sqlite3` module
(mirroring :class:`friday.meetings.store.SQLiteMeetingStore` and
:class:`friday.reminders.store.SQLiteReminderStore`).

Design rules (binding):

* **Local-first, zero-server.** SQLite is the concrete backend for both
  ``":memory:"`` (tests) and a file path (production). No daemon, no network.
* **Parametrized SQL only.** Every value reaches SQLite through a ``?``
  placeholder, so the store is injection-safe by construction.
* **Idempotent schema.** ``CREATE TABLE IF NOT EXISTS`` so constructing a store
  over an existing file never clobbers data.
* **Upsert by date.** ``date`` is the primary key; :meth:`save` is an upsert, so
  re-building a day overwrites that day's entry rather than duplicating it.
* **Thread-safe by construction (file paths).** A *connection-per-call* is opened
  for a filesystem-backed database (shareable across threads); an in-memory
  database keeps a single shared connection (a new connection would otherwise see
  an empty database).
* **Highlights as JSON.** The ``highlights`` list is serialized to a single JSON
  ``TEXT`` column, keeping the schema flat while preserving the ordered list on
  read.
"""

from __future__ import annotations

import json
import sqlite3

from friday.journal.service import JournalEntry

# In-memory SQLite database identifier — the ephemeral default for some callers.
_MEMORY_PATH = ":memory:"


class SQLiteJournalStore:
    """Local-first journal store backed by stdlib :mod:`sqlite3`.

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
        """Create the ``journal`` table if absent (idempotent).

        ``date`` is the primary key (one entry per calendar day);
        ``highlights`` holds the JSON-serialized list. Safe to call repeatedly and
        against an existing file.
        """
        conn = self._connect()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS journal (
                    date TEXT PRIMARY KEY,
                    summary TEXT NOT NULL,
                    highlights TEXT NOT NULL,
                    event_count INTEGER NOT NULL
                )
                """
            )
            conn.commit()
        finally:
            self._close(conn)

    # -- writes ------------------------------------------------------------ #
    def save(self, entry: JournalEntry) -> JournalEntry:
        """Upsert ``entry`` keyed on its ``date``; return the stored entry.

        Re-saving a date overwrites that day's summary / highlights / event count
        (an ``ON CONFLICT(date) DO UPDATE`` upsert), so re-building a day never
        duplicates a row.
        """
        highlights_json = json.dumps(list(entry.highlights))
        conn = self._connect()
        try:
            conn.execute(
                "INSERT INTO journal (date, summary, highlights, event_count) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(date) DO UPDATE SET "
                "summary = excluded.summary, "
                "highlights = excluded.highlights, "
                "event_count = excluded.event_count",
                (
                    entry.date,
                    entry.summary,
                    highlights_json,
                    entry.event_count,
                ),
            )
            conn.commit()
        finally:
            self._close(conn)
        return entry

    # -- reads ------------------------------------------------------------- #
    def get(self, date: str) -> JournalEntry | None:
        """Return the entry for ``date`` (``YYYY-MM-DD``) or ``None`` when absent."""
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT date, summary, highlights, event_count "
                "FROM journal WHERE date = ?",
                (date,),
            ).fetchone()
            return None if row is None else self._row_to_entry(row)
        finally:
            self._close(conn)

    def list_entries(self, limit: int = 30) -> list[JournalEntry]:
        """Return stored entries, most-recent date first, capped at ``limit``."""
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT date, summary, highlights, event_count "
                "FROM journal ORDER BY date DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [self._row_to_entry(row) for row in rows]
        finally:
            self._close(conn)

    # -- helpers ----------------------------------------------------------- #
    @staticmethod
    def _load_highlights(raw: str) -> list[str]:
        """Parse the JSON highlights column back into a list of strings."""
        data = json.loads(raw)
        return [str(item) for item in data]

    def _row_to_entry(self, row: sqlite3.Row) -> JournalEntry:
        return JournalEntry(
            date=str(row["date"]),
            summary=str(row["summary"]),
            highlights=self._load_highlights(str(row["highlights"])),
            event_count=int(row["event_count"]),
        )

    def _close(self, conn: sqlite3.Connection) -> None:
        """Close a per-call connection; leave the shared (memory) one open."""
        if conn is not self._shared:
            conn.close()
