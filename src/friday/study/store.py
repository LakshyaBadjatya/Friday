"""Local-first study store behind a SQLite backend (Tier 2 study module).

FRIDAY's durable study layer: spaced-repetition flashcards (scheduled by the pure
SM-2 core in :mod:`friday.study.srs`) and logged study sessions. The concrete,
gate-required backend is :class:`SQLiteStudyStore`, a local-first, zero-server
adapter built on the stdlib :mod:`sqlite3` module (mirroring
:class:`friday.reminders.store.SQLiteReminderStore` and
:class:`friday.scheduler.store.SQLiteTriggerStore`).

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
* **Clock injectable.** The "now" used to stamp a session's ``at`` and to
  reschedule a reviewed card's ``due_at`` comes from an injected ``clock() ->
  datetime``, never the wall clock, so timestamps are deterministic in tests.
  :meth:`due_cards` is driven entirely by the ``now`` the caller passes.

A flashcard's spaced-repetition state (``ease``/``interval_days``/``reps``) is the
SM-2 :class:`~friday.study.srs.ReviewState`; ``due_at`` is the ISO timestamp of
the card's next review (``None`` for a never-reviewed card, which is always due).
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

from pydantic import BaseModel

from friday.study.srs import ReviewState, sm2

# In-memory SQLite database identifier — the ephemeral default for some callers.
_MEMORY_PATH = ":memory:"


# --------------------------------------------------------------------------- #
# Typed row models
# --------------------------------------------------------------------------- #
class Flashcard(BaseModel):
    """A spaced-repetition flashcard row.

    ``deck`` groups cards; ``front``/``back`` are the prompt and answer. The
    SM-2 scheduling state is ``ease``/``interval_days``/``reps`` (see
    :class:`~friday.study.srs.ReviewState`). ``due_at`` is the ISO timestamp of
    the next review, or ``None`` for a never-reviewed card (always due).
    """

    id: int
    deck: str
    front: str
    back: str
    ease: float = 2.5
    interval_days: int = 0
    reps: int = 0
    due_at: str | None = None


class StudySession(BaseModel):
    """A logged study session: ``minutes`` spent on ``topic`` at ``at`` (ISO)."""

    id: int
    topic: str
    minutes: int
    at: str


# --------------------------------------------------------------------------- #
# SQLite implementation
# --------------------------------------------------------------------------- #
class SQLiteStudyStore:
    """Local-first study store backed by stdlib :mod:`sqlite3`.

    Args:
        path: A filesystem path for a durable database, or ``":memory:"`` for an
            ephemeral in-process database. For a file path a fresh connection is
            opened per call (thread-safe); ``":memory:"`` keeps one shared
            connection for the life of the instance (a new connection would see
            an empty database).
        clock: A zero-arg callable returning the current time as a
            :class:`~datetime.datetime`. Used to stamp a session's ``at`` and to
            reschedule a reviewed card's ``due_at``; injected so timestamps are
            deterministic in tests. Defaults to the system wall clock (UTC).
    """

    def __init__(
        self,
        path: str = _MEMORY_PATH,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._path = path
        self._clock: Callable[[], datetime] = (
            clock if clock is not None else (lambda: datetime.now(UTC))
        )
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
        """Create the ``flashcards`` + ``study_sessions`` tables if absent.

        Idempotent (``CREATE TABLE IF NOT EXISTS``); safe to call repeatedly and
        against an existing file.
        """
        conn = self._connect()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS flashcards (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    deck TEXT NOT NULL,
                    front TEXT NOT NULL,
                    back TEXT NOT NULL,
                    ease REAL NOT NULL DEFAULT 2.5,
                    interval_days INTEGER NOT NULL DEFAULT 0,
                    reps INTEGER NOT NULL DEFAULT 0,
                    due_at TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS study_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    topic TEXT NOT NULL,
                    minutes INTEGER NOT NULL,
                    at TEXT NOT NULL
                )
                """
            )
            conn.commit()
        finally:
            self._close(conn)

    # -- card writes ------------------------------------------------------- #
    def add_card(self, deck: str, front: str, back: str) -> Flashcard:
        """Insert a flashcard (parametrized) at the SM-2 defaults; return it.

        A new card starts at ease 2.5, interval 0, reps 0, with ``due_at`` unset
        (``None``) so it is immediately due for its first review.
        """
        conn = self._connect()
        try:
            cursor = conn.execute(
                "INSERT INTO flashcards (deck, front, back, ease, interval_days, reps, due_at) "
                "VALUES (?, ?, ?, 2.5, 0, 0, NULL)",
                (deck, front, back),
            )
            conn.commit()
            card_id = int(cursor.lastrowid or 0)
        finally:
            self._close(conn)
        return Flashcard(id=card_id, deck=deck, front=front, back=back)

    def review_card(self, card_id: int, grade: int) -> Flashcard | None:
        """Apply SM-2 for ``grade`` and reschedule the card; ``None`` if absent.

        Loads the card's current SM-2 state, runs :func:`~friday.study.srs.sm2`,
        persists the new ease/interval/reps, and sets ``due_at`` to ``now +
        interval_days`` where ``now`` comes from the injected clock. Returns the
        updated :class:`Flashcard`, or ``None`` when no card has ``card_id``.

        Raises:
            ValueError: if ``grade`` is outside ``0..5`` (propagated from
                :func:`~friday.study.srs.sm2`).
        """
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT id, deck, front, back, ease, interval_days, reps, due_at "
                "FROM flashcards WHERE id = ?",
                (card_id,),
            ).fetchone()
            if row is None:
                return None

            state = ReviewState(
                ease=float(row["ease"]),
                interval_days=int(row["interval_days"]),
                reps=int(row["reps"]),
            )
            next_state = sm2(state, grade)
            due_at = (self._clock() + timedelta(days=next_state.interval_days)).isoformat()
            conn.execute(
                "UPDATE flashcards SET ease = ?, interval_days = ?, reps = ?, due_at = ? "
                "WHERE id = ?",
                (
                    next_state.ease,
                    next_state.interval_days,
                    next_state.reps,
                    due_at,
                    card_id,
                ),
            )
            conn.commit()
            return Flashcard(
                id=int(row["id"]),
                deck=str(row["deck"]),
                front=str(row["front"]),
                back=str(row["back"]),
                ease=next_state.ease,
                interval_days=next_state.interval_days,
                reps=next_state.reps,
                due_at=due_at,
            )
        finally:
            self._close(conn)

    def delete_card(self, card_id: int) -> int:
        """Delete a flashcard by id; return the number of rows removed (0 or 1)."""
        conn = self._connect()
        try:
            cursor = conn.execute(
                "DELETE FROM flashcards WHERE id = ?", (card_id,)
            )
            conn.commit()
            return cursor.rowcount
        finally:
            self._close(conn)

    # -- card reads -------------------------------------------------------- #
    def list_cards(self, deck: str | None = None) -> list[Flashcard]:
        """Return flashcards in insertion order, optionally filtered by ``deck``."""
        conn = self._connect()
        try:
            if deck is None:
                rows = conn.execute(
                    "SELECT id, deck, front, back, ease, interval_days, reps, due_at "
                    "FROM flashcards ORDER BY id ASC"
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT id, deck, front, back, ease, interval_days, reps, due_at "
                    "FROM flashcards WHERE deck = ? ORDER BY id ASC",
                    (deck,),
                ).fetchall()
            return [self._row_to_card(row) for row in rows]
        finally:
            self._close(conn)

    def due_cards(self, now: datetime) -> list[Flashcard]:
        """Return cards due at or before ``now`` (and never-reviewed cards).

        Driven entirely by the passed ``now`` datetime — the store never reads
        the wall clock here. A never-reviewed card (``due_at IS NULL``) is always
        due; a scheduled card is due when its ``due_at`` is at or before ``now``.
        ISO-8601 timestamps compare correctly as lexicographic strings when
        uniformly formatted, so the filter is a parametrized string comparison.
        """
        now_iso = now.isoformat()
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT id, deck, front, back, ease, interval_days, reps, due_at "
                "FROM flashcards "
                "WHERE due_at IS NULL OR due_at <= ? "
                "ORDER BY due_at IS NOT NULL, due_at ASC, id ASC",
                (now_iso,),
            ).fetchall()
            return [self._row_to_card(row) for row in rows]
        finally:
            self._close(conn)

    # -- session writes/reads ---------------------------------------------- #
    def add_session(self, topic: str, minutes: int) -> StudySession:
        """Log a study session (``at`` stamped from the injected clock)."""
        at = self._clock().isoformat()
        conn = self._connect()
        try:
            cursor = conn.execute(
                "INSERT INTO study_sessions (topic, minutes, at) VALUES (?, ?, ?)",
                (topic, minutes, at),
            )
            conn.commit()
            session_id = int(cursor.lastrowid or 0)
        finally:
            self._close(conn)
        return StudySession(id=session_id, topic=topic, minutes=minutes, at=at)

    def list_sessions(self, limit: int = 20) -> list[StudySession]:
        """Return logged sessions most-recent first, capped at ``limit``."""
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT id, topic, minutes, at FROM study_sessions "
                "ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [
                StudySession(
                    id=int(row["id"]),
                    topic=str(row["topic"]),
                    minutes=int(row["minutes"]),
                    at=str(row["at"]),
                )
                for row in rows
            ]
        finally:
            self._close(conn)

    # -- helpers ----------------------------------------------------------- #
    @staticmethod
    def _row_to_card(row: sqlite3.Row) -> Flashcard:
        return Flashcard(
            id=int(row["id"]),
            deck=str(row["deck"]),
            front=str(row["front"]),
            back=str(row["back"]),
            ease=float(row["ease"]),
            interval_days=int(row["interval_days"]),
            reps=int(row["reps"]),
            due_at=None if row["due_at"] is None else str(row["due_at"]),
        )

    def _close(self, conn: sqlite3.Connection) -> None:
        """Close a per-call connection; leave the shared (memory) one open."""
        if conn is not self._shared:
            conn.close()
