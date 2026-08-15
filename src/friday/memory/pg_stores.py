"""The rest of the durable stores: journal, flashcards, protocols.

Companion to :mod:`friday.reminders.pg_store`, which covers reminders and
scheduler triggers. Together they move everything a person would actually mourn
off the container's disk, which Render's free tier wipes on every deploy.

What is deliberately *not* here, and why:

* **Flows** are in-flight workflow runs — losing a half-finished run on deploy is
  a restart, not data loss.
* **The knowledge graph** is derived from long-term facts, which are already on
  Postgres; it rebuilds.
* **Circle/family** state lives in Firestore already.

Each class mirrors its SQLite counterpart signature for signature so the app
cannot tell which one it was handed, and ``psycopg`` is imported lazily inside
the shared connection helper so a build without the driver still starts.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any

from friday.journal.service import JournalEntry
from friday.logging import get_logger
from friday.protocols.store import Protocol, ProtocolStep
from friday.reminders.pg_store import _connect
from friday.study.srs import ReviewState, sm2
from friday.study.store import Flashcard, StudySession

logger = get_logger("friday.memory.pg_stores")


class PostgresJournalStore:
    """:class:`SQLiteJournalStore`'s contract, stored durably.

    ``highlights`` is a JSON array in a text column, exactly as SQLite held it —
    keeping the encoding identical means an entry written by either backend reads
    back the same from the other.
    """

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn

    def init_schema(self) -> None:
        with _connect(self._dsn) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS journal (
                    date        TEXT PRIMARY KEY,
                    summary     TEXT NOT NULL,
                    highlights  TEXT NOT NULL DEFAULT '[]',
                    event_count INTEGER NOT NULL DEFAULT 0
                )
                """
            )

    def save(self, entry: JournalEntry) -> JournalEntry:
        """Insert or replace one day's entry (the date is the key)."""
        with _connect(self._dsn) as conn:
            conn.execute(
                "INSERT INTO journal (date, summary, highlights, event_count) "
                "VALUES (%s, %s, %s, %s) "
                "ON CONFLICT (date) DO UPDATE SET summary = EXCLUDED.summary, "
                "highlights = EXCLUDED.highlights, event_count = EXCLUDED.event_count",
                (
                    entry.date,
                    entry.summary,
                    json.dumps(list(entry.highlights)),
                    entry.event_count,
                ),
            )
        return entry

    def get(self, date: str) -> JournalEntry | None:
        with _connect(self._dsn) as conn:
            row = conn.execute(
                "SELECT date, summary, highlights, event_count FROM journal "
                "WHERE date = %s",
                (date,),
            ).fetchone()
        return _entry(row) if row is not None else None

    def list_entries(self, limit: int = 30) -> list[JournalEntry]:
        with _connect(self._dsn) as conn:
            rows = conn.execute(
                "SELECT date, summary, highlights, event_count FROM journal "
                "ORDER BY date DESC LIMIT %s",
                (limit,),
            ).fetchall()
        return [_entry(r) for r in rows]


class PostgresStudyStore:
    """:class:`SQLiteStudyStore`'s contract, stored durably.

    The spaced-repetition schedule is the part worth persisting: a wiped deck is
    annoying, but a wiped *schedule* silently resets months of review history and
    starts showing long-learned cards every day again.
    """

    def __init__(self, dsn: str, clock: Any = None) -> None:
        self._dsn = dsn
        self._clock = clock or (lambda: datetime.now().astimezone())

    def init_schema(self) -> None:
        with _connect(self._dsn) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS flashcards (
                    id            SERIAL PRIMARY KEY,
                    deck          TEXT NOT NULL,
                    front         TEXT NOT NULL,
                    back          TEXT NOT NULL,
                    ease          DOUBLE PRECISION NOT NULL DEFAULT 2.5,
                    interval_days INTEGER NOT NULL DEFAULT 0,
                    reps          INTEGER NOT NULL DEFAULT 0,
                    due_at        TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS study_sessions (
                    id      SERIAL PRIMARY KEY,
                    topic   TEXT NOT NULL,
                    minutes INTEGER NOT NULL,
                    at      TEXT NOT NULL
                )
                """
            )

    def add_card(self, deck: str, front: str, back: str) -> Flashcard:
        due_at = self._clock().isoformat()
        with _connect(self._dsn) as conn:
            row = conn.execute(
                "INSERT INTO flashcards (deck, front, back, ease, interval_days, "
                "reps, due_at) VALUES (%s, %s, %s, 2.5, 0, 0, %s) RETURNING id",
                (deck, front, back, due_at),
            ).fetchone()
        return Flashcard(
            id=int(row[0]), deck=deck, front=front, back=back,
            ease=2.5, interval_days=0, reps=0, due_at=due_at,
        )

    def review_card(self, card_id: int, grade: int) -> Flashcard | None:
        """Apply SM-2 and reschedule; ``None`` when no card has ``card_id``."""
        with _connect(self._dsn) as conn:
            row = conn.execute(
                "SELECT id, deck, front, back, ease, interval_days, reps, due_at "
                "FROM flashcards WHERE id = %s",
                (card_id,),
            ).fetchone()
            if row is None:
                return None
            # sm2 validates the grade and raises ValueError outside 0..5; that
            # propagates deliberately so the route can answer 422 as it does on
            # SQLite rather than silently storing a nonsense review.
            nxt = sm2(
                ReviewState(ease=float(row[4]), interval_days=int(row[5]),
                            reps=int(row[6])),
                grade,
            )
            due_at = (self._clock() + timedelta(days=nxt.interval_days)).isoformat()
            conn.execute(
                "UPDATE flashcards SET ease = %s, interval_days = %s, reps = %s, "
                "due_at = %s WHERE id = %s",
                (nxt.ease, nxt.interval_days, nxt.reps, due_at, card_id),
            )
        return Flashcard(
            id=int(row[0]), deck=row[1], front=row[2], back=row[3],
            ease=nxt.ease, interval_days=nxt.interval_days, reps=nxt.reps,
            due_at=due_at,
        )

    def delete_card(self, card_id: int) -> int:
        with _connect(self._dsn) as conn:
            row = conn.execute(
                "DELETE FROM flashcards WHERE id = %s RETURNING id", (card_id,)
            ).fetchone()
        return 1 if row is not None else 0

    def list_cards(self, deck: str | None = None) -> list[Flashcard]:
        sql = ("SELECT id, deck, front, back, ease, interval_days, reps, due_at "
               "FROM flashcards")
        params: tuple[Any, ...] = ()
        if deck is not None:
            sql += " WHERE deck = %s"
            params = (deck,)
        sql += " ORDER BY id"
        with _connect(self._dsn) as conn:
            rows = conn.execute(sql, params).fetchall()
        return [_card(r) for r in rows]

    def due_cards(self, now: datetime) -> list[Flashcard]:
        with _connect(self._dsn) as conn:
            rows = conn.execute(
                "SELECT id, deck, front, back, ease, interval_days, reps, due_at "
                "FROM flashcards WHERE due_at IS NOT NULL AND due_at <= %s "
                "ORDER BY due_at",
                (now.isoformat(),),
            ).fetchall()
        return [_card(r) for r in rows]

    def add_session(self, topic: str, minutes: int) -> StudySession:
        at = self._clock().isoformat()
        with _connect(self._dsn) as conn:
            row = conn.execute(
                "INSERT INTO study_sessions (topic, minutes, at) "
                "VALUES (%s, %s, %s) RETURNING id",
                (topic, minutes, at),
            ).fetchone()
        return StudySession(id=int(row[0]), topic=topic, minutes=minutes, at=at)

    def list_sessions(self, limit: int = 20) -> list[StudySession]:
        with _connect(self._dsn) as conn:
            rows = conn.execute(
                "SELECT id, topic, minutes, at FROM study_sessions "
                "ORDER BY id DESC LIMIT %s",
                (limit,),
            ).fetchall()
        return [
            StudySession(id=int(r[0]), topic=r[1], minutes=int(r[2]), at=r[3])
            for r in rows
        ]


class PostgresProtocolStore:
    """:class:`SQLiteProtocolStore`'s contract, stored durably."""

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn

    def init_schema(self) -> None:
        with _connect(self._dsn) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS protocols (
                    id             SERIAL PRIMARY KEY,
                    name           TEXT NOT NULL,
                    trigger_phrase TEXT NOT NULL,
                    steps          TEXT NOT NULL,
                    enabled        BOOLEAN NOT NULL DEFAULT TRUE
                )
                """
            )

    def add(
        self,
        *,
        name: str,
        trigger_phrase: str,
        steps: list[ProtocolStep],
        enabled: bool = True,
    ) -> Protocol:
        with _connect(self._dsn) as conn:
            row = conn.execute(
                "INSERT INTO protocols (name, trigger_phrase, steps, enabled) "
                "VALUES (%s, %s, %s, %s) RETURNING id",
                (name, trigger_phrase, _dump_steps(steps), enabled),
            ).fetchone()
        return Protocol(
            id=int(row[0]), name=name, trigger_phrase=trigger_phrase,
            steps=steps, enabled=enabled,
        )

    def update(self, protocol: Protocol) -> bool:
        with _connect(self._dsn) as conn:
            row = conn.execute(
                "UPDATE protocols SET name = %s, trigger_phrase = %s, steps = %s, "
                "enabled = %s WHERE id = %s RETURNING id",
                (
                    protocol.name, protocol.trigger_phrase,
                    _dump_steps(protocol.steps), protocol.enabled, protocol.id,
                ),
            ).fetchone()
        return row is not None

    def set_enabled(self, protocol_id: int, enabled: bool) -> bool:
        with _connect(self._dsn) as conn:
            row = conn.execute(
                "UPDATE protocols SET enabled = %s WHERE id = %s RETURNING id",
                (enabled, protocol_id),
            ).fetchone()
        return row is not None

    def delete(self, protocol_id: int) -> int:
        with _connect(self._dsn) as conn:
            row = conn.execute(
                "DELETE FROM protocols WHERE id = %s RETURNING id", (protocol_id,)
            ).fetchone()
        return 1 if row is not None else 0

    def get(self, protocol_id: int) -> Protocol | None:
        with _connect(self._dsn) as conn:
            row = conn.execute(
                "SELECT id, name, trigger_phrase, steps, enabled FROM protocols "
                "WHERE id = %s",
                (protocol_id,),
            ).fetchone()
        return _protocol(row) if row is not None else None

    def get_by_name(self, name: str) -> Protocol | None:
        with _connect(self._dsn) as conn:
            row = conn.execute(
                "SELECT id, name, trigger_phrase, steps, enabled FROM protocols "
                "WHERE lower(name) = lower(%s)",
                (name,),
            ).fetchone()
        return _protocol(row) if row is not None else None

    def list_protocols(self) -> list[Protocol]:
        with _connect(self._dsn) as conn:
            rows = conn.execute(
                "SELECT id, name, trigger_phrase, steps, enabled FROM protocols "
                "ORDER BY id"
            ).fetchall()
        return [_protocol(r) for r in rows]


def _dump_steps(steps: list[ProtocolStep]) -> str:
    return json.dumps([s.model_dump() for s in steps])


def _load_steps(raw: str) -> list[ProtocolStep]:
    try:
        return [ProtocolStep(**s) for s in json.loads(raw or "[]")]
    except (ValueError, TypeError):
        logger.warning("protocol steps unreadable; treating as empty")
        return []


def _entry(row: Any) -> JournalEntry:
    try:
        highlights = list(json.loads(row[2] or "[]"))
    except (ValueError, TypeError):
        highlights = []
    return JournalEntry(
        date=row[0], summary=row[1], highlights=highlights,
        event_count=int(row[3] or 0),
    )


def _card(row: Any) -> Flashcard:
    return Flashcard(
        id=int(row[0]), deck=row[1], front=row[2], back=row[3],
        ease=float(row[4]), interval_days=int(row[5]), reps=int(row[6]),
        due_at=row[7],
    )


def _protocol(row: Any) -> Protocol:
    return Protocol(
        id=int(row[0]), name=row[1], trigger_phrase=row[2],
        steps=_load_steps(row[3]), enabled=bool(row[4]),
    )
