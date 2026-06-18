"""Unit tests for circle care: cross-person reminders + SOS alerts.

Reminders are scheduled in the *target's* timezone (so "remind her at 9pm" fires
at 9pm where she is) and gated by consent (you can only remind someone you share a
group with). SOS alerts are visible to everyone who shares a group with the person
who raised one. All offline; the reference instant is passed in.
"""

from __future__ import annotations

from datetime import UTC, datetime, time, timedelta
from zoneinfo import ZoneInfo

import pytest

from friday import errors
from friday.circle.care import CareService, InMemoryCareStore
from friday.circle.service import CircleService
from friday.circle.store import InMemoryCircleStore

NOW = datetime(2026, 6, 18, 17, 0, tzinfo=UTC)  # 1pm New York / 10:30pm Kolkata
NY = ZoneInfo("America/New_York")


def _circle() -> CircleService:
    circle = CircleService(InMemoryCircleStore())
    circle.create_group(
        name="Us",
        admin_uid="u-india",
        admin_display_name="Me",
        admin_tz="Asia/Kolkata",
        now=NOW,
        group_id="g1",
    )
    circle.accept_invite(
        code=circle.invite(group_id="g1", by_uid="u-india", now=NOW).code,
        uid="u-us",
        display_name="Bestie",
        tz="America/New_York",
        now=NOW,
    )
    return circle


def _care(circle: CircleService) -> CareService:
    return CareService(circle, InMemoryCareStore())


def test_reminder_is_scheduled_in_the_targets_timezone() -> None:
    care = _care(_circle())
    reminder = care.remind(
        creator_uid="u-india",
        target_uid="u-us",
        text="take your meds",
        local_time=time(21, 0),
        now=NOW,
    )
    # 9pm should be 9pm in New York, whatever that is in UTC.
    assert reminder.fire_at.astimezone(NY).hour == 21
    assert reminder.target_uid == "u-us"


def test_reminding_someone_you_dont_share_a_group_with_is_denied() -> None:
    care = _care(_circle())
    with pytest.raises(errors.PermissionError):
        care.remind(
            creator_uid="u-india",
            target_uid="u-stranger",
            text="hi",
            local_time=time(9, 0),
            now=NOW,
        )


def test_due_reminders_and_completion() -> None:
    circle = _circle()
    care = _care(circle)
    reminder = care.remind(
        creator_uid="u-india",
        target_uid="u-us",
        text="meds",
        local_time=time(21, 0),
        now=NOW,
    )
    before = reminder.fire_at - timedelta(minutes=1)
    after = reminder.fire_at + timedelta(minutes=1)
    assert care.due_reminders(now=before) == []
    assert [r.id for r in care.due_reminders(now=after)] == [reminder.id]
    assert care.complete_reminder(reminder.id) is True
    assert care.due_reminders(now=after) == []


def test_sos_is_visible_to_the_circle_but_not_strangers() -> None:
    circle = _circle()
    care = _care(circle)
    alert = care.raise_sos(uid="u-us", message="need help", place="downtown", now=NOW)
    assert alert.message == "need help"
    # The person who shares her group sees it; a stranger sees nothing.
    seen = care.alerts_for("u-india")
    assert [a.id for a in seen] == [alert.id]
    assert care.alerts_for("u-stranger") == []


def test_resolving_sos_clears_it() -> None:
    circle = _circle()
    care = _care(circle)
    alert = care.raise_sos(uid="u-us", now=NOW)
    assert care.resolve_sos(alert.id) is True
    assert care.alerts_for("u-india") == []
