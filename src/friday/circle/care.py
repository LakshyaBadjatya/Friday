"""Care: cross-person reminders (scheduled in the target's timezone) and SOS alerts.

Reminders let one person nudge another ("take your meds at 9") with the time
resolved in the *recipient's* timezone, so a long-distance reminder fires at the
right local moment. SOS alerts are a panic signal visible to everyone who shares a
group with the person who raised one. Both are consent-gated by
:meth:`CircleService.shares_group`. Storage is behind :class:`CareStore`; the
in-memory implementation backs tests and local runs.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta
from typing import Protocol
from uuid import uuid4
from zoneinfo import ZoneInfo

from pydantic import BaseModel

from friday.circle.service import CircleService
from friday.errors import PermissionError


class Reminder(BaseModel):
    """A reminder one member sets for another, fired at a UTC instant."""

    id: str
    creator_uid: str
    target_uid: str
    text: str
    fire_at: datetime
    created_at: datetime
    done: bool = False


class SosAlert(BaseModel):
    """A panic alert raised by a member, visible to their circle until resolved."""

    id: str
    uid: str
    message: str | None = None
    place: str | None = None
    raised_at: datetime
    resolved: bool = False


class CareStore(Protocol):
    """Persistence for reminders and SOS alerts."""

    def save_reminder(self, reminder: Reminder) -> None: ...

    def get_reminder(self, reminder_id: str) -> Reminder | None: ...

    def list_reminders(self) -> list[Reminder]: ...

    def save_alert(self, alert: SosAlert) -> None: ...

    def get_alert(self, alert_id: str) -> SosAlert | None: ...

    def list_alerts(self) -> list[SosAlert]: ...


class InMemoryCareStore:
    """A dict-backed :class:`CareStore` for tests and local use."""

    def __init__(self) -> None:
        self._reminders: dict[str, Reminder] = {}
        self._alerts: dict[str, SosAlert] = {}

    def save_reminder(self, reminder: Reminder) -> None:
        self._reminders[reminder.id] = reminder

    def get_reminder(self, reminder_id: str) -> Reminder | None:
        return self._reminders.get(reminder_id)

    def list_reminders(self) -> list[Reminder]:
        return list(self._reminders.values())

    def save_alert(self, alert: SosAlert) -> None:
        self._alerts[alert.id] = alert

    def get_alert(self, alert_id: str) -> SosAlert | None:
        return self._alerts.get(alert_id)

    def list_alerts(self) -> list[SosAlert]:
        return list(self._alerts.values())


def _next_local_time_utc(
    local_t: time, tz_name: str, now_utc: datetime, on_date: date | None
) -> datetime:
    """Resolve a wall-clock ``local_t`` in ``tz_name`` to a UTC instant.

    With ``on_date`` given, that date is used; otherwise the next occurrence (today
    if still ahead, else tomorrow) is chosen relative to ``now_utc``.
    """
    tz = ZoneInfo(tz_name)
    now_local = now_utc.astimezone(tz)
    target_date = on_date or now_local.date()
    candidate = datetime.combine(target_date, local_t, tzinfo=tz)
    if on_date is None and candidate <= now_local:
        candidate += timedelta(days=1)
    return candidate.astimezone(UTC)


class CareService:
    """Reminders + SOS, with consent and timezone-correct scheduling."""

    def __init__(self, circle: CircleService, store: CareStore) -> None:
        self._circle = circle
        self._store = store

    def remind(
        self,
        *,
        creator_uid: str,
        target_uid: str,
        text: str,
        local_time: time,
        now: datetime,
        on_date: date | None = None,
        reminder_id: str | None = None,
    ) -> Reminder:
        """Set a reminder for ``target_uid`` at ``local_time`` in *their* timezone."""
        if not self._circle.shares_group(creator_uid, target_uid):
            raise PermissionError(
                f"{creator_uid!r} cannot remind {target_uid!r} (no shared group)"
            )
        member = self._circle.find_member(target_uid)
        tz = member.tz if member else "UTC"
        reminder = Reminder(
            id=reminder_id or uuid4().hex,
            creator_uid=creator_uid,
            target_uid=target_uid,
            text=text,
            fire_at=_next_local_time_utc(local_time, tz, now, on_date),
            created_at=now,
        )
        self._store.save_reminder(reminder)
        return reminder

    def reminders_for(self, target_uid: str) -> list[Reminder]:
        return [
            r
            for r in self._store.list_reminders()
            if r.target_uid == target_uid and not r.done
        ]

    def due_reminders(self, *, now: datetime) -> list[Reminder]:
        return [
            r for r in self._store.list_reminders() if not r.done and r.fire_at <= now
        ]

    def complete_reminder(self, reminder_id: str) -> bool:
        reminder = self._store.get_reminder(reminder_id)
        if reminder is None or reminder.done:
            return False
        self._store.save_reminder(reminder.model_copy(update={"done": True}))
        return True

    def raise_sos(
        self,
        *,
        uid: str,
        message: str | None = None,
        place: str | None = None,
        now: datetime,
        alert_id: str | None = None,
    ) -> SosAlert:
        alert = SosAlert(
            id=alert_id or uuid4().hex,
            uid=uid,
            message=message,
            place=place,
            raised_at=now,
        )
        self._store.save_alert(alert)
        return alert

    def alerts_for(self, viewer_uid: str) -> list[SosAlert]:
        """Unresolved SOS alerts from people the viewer shares a group with."""
        return [
            a
            for a in self._store.list_alerts()
            if not a.resolved and self._circle.shares_group(viewer_uid, a.uid)
        ]

    def resolve_sos(self, alert_id: str) -> bool:
        alert = self._store.get_alert(alert_id)
        if alert is None or alert.resolved:
            return False
        self._store.save_alert(alert.model_copy(update={"resolved": True}))
        return True
