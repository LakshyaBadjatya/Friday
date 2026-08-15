"""Reminders and triggers that survive a deploy.

Render's free tier hands each container an ephemeral filesystem, so every deploy
starts on a clean disk. The SQLite stores were erasing themselves several times an
hour: a reminder set at 4:38 was gone by 4:40, and the trigger that *fires*
reminders went with it, so nothing fired at all. The bug was never in the reminder
logic — it was in where the file lived.

These are the same two stores against Postgres, which outlives the container. They
mirror :class:`~friday.reminders.store.SQLiteReminderStore` and
:class:`~friday.scheduler.store.SQLiteTriggerStore` signature for signature, so
nothing downstream can tell which one it was handed.

Following :mod:`friday.memory.pg`, ``psycopg`` is imported lazily inside the
connection helper: a build without the driver still starts, and the caller falls
back to SQLite rather than the app failing at import time.

Timestamps are stored as ISO-8601 ``text``, not ``timestamptz``. That looks lazy
and is deliberate — ``due()`` compares them lexicographically, which is correct
for uniformly-formatted ISO strings, and it keeps both backends byte-identical.
Switching to a native type would quietly change comparison semantics between the
two backends, which is exactly the kind of difference that hides for weeks.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from friday.logging import get_logger
from friday.reminders.store import Reminder
from friday.scheduler.store import Trigger, TriggerKind

logger = get_logger("friday.reminders.pg_store")


def _connect(dsn: str) -> Any:
    """Open an autocommit Postgres connection, importing the driver lazily."""
    import psycopg  # noqa: PLC0415

    return psycopg.connect(dsn, autocommit=True)


class PostgresReminderStore:
    """:class:`SQLiteReminderStore`'s contract, stored durably."""

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn

    def init_schema(self) -> None:
        with _connect(self._dsn) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS reminders (
                    id         SERIAL PRIMARY KEY,
                    text       TEXT NOT NULL,
                    due_at     TEXT,
                    recurrence TEXT,
                    status     TEXT NOT NULL DEFAULT 'open',
                    created_at TEXT NOT NULL
                )
                """
            )

    def add(
        self, text: str, due_at: str | None = None, recurrence: str | None = None
    ) -> Reminder:
        created_at = datetime.now(UTC).isoformat()
        with _connect(self._dsn) as conn:
            row = conn.execute(
                "INSERT INTO reminders (text, due_at, recurrence, status, created_at) "
                "VALUES (%s, %s, %s, 'open', %s) RETURNING id",
                (text, due_at, recurrence, created_at),
            ).fetchone()
        return Reminder(
            id=int(row[0]), text=text, due_at=due_at, recurrence=recurrence,
            status="open", created_at=created_at,
        )

    def list_reminders(
        self, status: Literal["open", "all"] = "open"
    ) -> list[Reminder]:
        sql = "SELECT id, text, due_at, recurrence, status, created_at FROM reminders"
        params: tuple[Any, ...] = ()
        if status == "open":
            sql += " WHERE status = %s"
            params = ("open",)
        sql += " ORDER BY id"
        with _connect(self._dsn) as conn:
            rows = conn.execute(sql, params).fetchall()
        return [_reminder(r) for r in rows]

    def due(self, now_iso: str) -> list[Reminder]:
        with _connect(self._dsn) as conn:
            rows = conn.execute(
                "SELECT id, text, due_at, recurrence, status, created_at FROM reminders "
                "WHERE status = 'open' AND due_at IS NOT NULL AND due_at <= %s "
                "ORDER BY due_at",
                (now_iso,),
            ).fetchall()
        return [_reminder(r) for r in rows]

    def complete(self, reminder_id: int) -> bool:
        """Close a reminder; a recurring one rolls forward instead of closing."""
        with _connect(self._dsn) as conn:
            row = conn.execute(
                "SELECT due_at, recurrence FROM reminders "
                "WHERE id = %s AND status = 'open'",
                (reminder_id,),
            ).fetchone()
            if row is None:
                return False
            due_at, recurrence = row[0], row[1]
            rolled = _roll_forward(due_at, recurrence)
            if rolled is not None:
                conn.execute(
                    "UPDATE reminders SET due_at = %s WHERE id = %s",
                    (rolled, reminder_id),
                )
            else:
                conn.execute(
                    "UPDATE reminders SET status = 'done' WHERE id = %s", (reminder_id,)
                )
        return True

    def delete(self, reminder_id: int) -> int:
        with _connect(self._dsn) as conn:
            row = conn.execute(
                "DELETE FROM reminders WHERE id = %s RETURNING id", (reminder_id,)
            ).fetchone()
        return 1 if row is not None else 0


class PostgresTriggerStore:
    """:class:`SQLiteTriggerStore`'s contract, stored durably.

    Persisting this matters as much as the reminders: the trigger is what fires
    them. Wiped by a deploy, the reminders survived in name only — sitting open
    forever with nothing scheduled to look at them.
    """

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn

    def init_schema(self) -> None:
        with _connect(self._dsn) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS triggers (
                    id       SERIAL PRIMARY KEY,
                    name     TEXT NOT NULL,
                    kind     TEXT NOT NULL,
                    spec     TEXT NOT NULL,
                    action   TEXT NOT NULL,
                    enabled  BOOLEAN NOT NULL DEFAULT TRUE,
                    next_run TEXT,
                    last_run TEXT
                )
                """
            )

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
        with _connect(self._dsn) as conn:
            row = conn.execute(
                "INSERT INTO triggers (name, kind, spec, action, enabled, next_run, "
                "last_run) VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id",
                (name, kind, spec, action, enabled, next_run, last_run),
            ).fetchone()
        return Trigger(
            id=int(row[0]), name=name, kind=kind, spec=spec, action=action,
            enabled=enabled, next_run=next_run, last_run=last_run,
        )

    def update(self, trigger: Trigger) -> bool:
        with _connect(self._dsn) as conn:
            row = conn.execute(
                "UPDATE triggers SET name = %s, kind = %s, spec = %s, action = %s, "
                "enabled = %s, next_run = %s, last_run = %s WHERE id = %s RETURNING id",
                (
                    trigger.name, trigger.kind, trigger.spec, trigger.action,
                    trigger.enabled, trigger.next_run, trigger.last_run, trigger.id,
                ),
            ).fetchone()
        return row is not None

    def set_enabled(self, trigger_id: int, enabled: bool) -> bool:
        with _connect(self._dsn) as conn:
            row = conn.execute(
                "UPDATE triggers SET enabled = %s WHERE id = %s RETURNING id",
                (enabled, trigger_id),
            ).fetchone()
        return row is not None

    def delete(self, trigger_id: int) -> int:
        with _connect(self._dsn) as conn:
            row = conn.execute(
                "DELETE FROM triggers WHERE id = %s RETURNING id", (trigger_id,)
            ).fetchone()
        return 1 if row is not None else 0

    def get(self, trigger_id: int) -> Trigger | None:
        with _connect(self._dsn) as conn:
            row = conn.execute(
                "SELECT id, name, kind, spec, action, enabled, next_run, last_run "
                "FROM triggers WHERE id = %s",
                (trigger_id,),
            ).fetchone()
        return _trigger(row) if row is not None else None

    def list_triggers(self) -> list[Trigger]:
        with _connect(self._dsn) as conn:
            rows = conn.execute(
                "SELECT id, name, kind, spec, action, enabled, next_run, last_run "
                "FROM triggers ORDER BY id"
            ).fetchall()
        return [_trigger(r) for r in rows]

    def due(self, now: datetime) -> list[Trigger]:
        """Enabled triggers whose ``next_run`` is at or before ``now``."""
        with _connect(self._dsn) as conn:
            rows = conn.execute(
                "SELECT id, name, kind, spec, action, enabled, next_run, last_run "
                "FROM triggers WHERE enabled = TRUE AND next_run IS NOT NULL "
                "AND next_run <= %s ORDER BY next_run",
                (now.isoformat(),),
            ).fetchall()
        return [_trigger(r) for r in rows]


def _roll_forward(due_at: str | None, recurrence: str | None) -> str | None:
    """The next occurrence of a recurring reminder, or ``None`` if it is one-off."""
    if not due_at or not recurrence:
        return None
    from datetime import timedelta  # noqa: PLC0415

    steps = {"daily": timedelta(days=1), "weekly": timedelta(weeks=1),
             "monthly": timedelta(days=30)}
    step = steps.get(recurrence)
    if step is None:
        return None
    try:
        return (datetime.fromisoformat(due_at) + step).isoformat()
    except ValueError:
        return None


def _reminder(row: Any) -> Reminder:
    return Reminder(
        id=int(row[0]), text=row[1], due_at=row[2], recurrence=row[3],
        status=row[4], created_at=row[5],
    )


def _trigger(row: Any) -> Trigger:
    return Trigger(
        id=int(row[0]), name=row[1], kind=row[2], spec=row[3], action=row[4],
        enabled=bool(row[5]), next_run=row[6], last_run=row[7],
    )
