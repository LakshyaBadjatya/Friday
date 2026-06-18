"""Unit tests for circle status: set/read with consent + timezone-aware describe.

Offline against the in-memory stores. A two-person circle (one in India, one in
the US) exercises the long-distance describe path; the reference instant is passed
in so "set N ago" and local-time are deterministic.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from friday import errors
from friday.circle.service import CircleService
from friday.circle.status import InMemoryStatusStore, StatusService, humanize_ago
from friday.circle.store import InMemoryCircleStore

NOW = datetime(2026, 6, 18, 17, 0, tzinfo=UTC)  # 1pm New York / 10:30pm Kolkata


def _circle_with_two() -> CircleService:
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


def _status(circle: CircleService) -> StatusService:
    return StatusService(circle, InMemoryStatusStore())


def test_set_and_read_status_with_consent() -> None:
    circle = _circle_with_two()
    svc = _status(circle)
    svc.set_status("u-us", text="having lunch", now=NOW)
    got = svc.get_status("u-india", "u-us")
    assert got is not None
    assert got.text == "having lunch"


def test_reading_without_a_shared_group_is_denied() -> None:
    circle = _circle_with_two()
    svc = _status(circle)
    svc.set_status("u-us", text="having lunch", now=NOW)
    with pytest.raises(errors.PermissionError):
        svc.get_status("u-stranger", "u-us")


def test_set_status_merges_fields() -> None:
    circle = _circle_with_two()
    svc = _status(circle)
    svc.set_status("u-us", text="working", now=NOW)
    svc.set_status("u-us", mood="focused", now=NOW + timedelta(minutes=5))
    got = svc.get_status("u-india", "u-us")
    assert got is not None
    assert got.text == "working"  # not wiped by the later mood-only update
    assert got.mood == "focused"


def test_describe_is_timezone_aware_and_dated() -> None:
    circle = _circle_with_two()
    svc = _status(circle)
    svc.set_status("u-us", text="having lunch", now=NOW - timedelta(minutes=20))
    line = svc.describe("u-india", "u-us", now=NOW)
    assert "Bestie" in line
    assert "having lunch" in line
    assert "PM" in line  # 1:00 PM in New York
    assert "20 minutes ago" in line


def test_describe_flags_when_the_friend_is_probably_asleep() -> None:
    circle = _circle_with_two()
    svc = _status(circle)
    asleep_now = datetime(2026, 6, 18, 7, 0, tzinfo=UTC)  # 3am New York
    svc.set_status("u-us", text="resting", now=asleep_now - timedelta(hours=2))
    line = svc.describe("u-india", "u-us", now=asleep_now)
    assert "asleep" in line
    assert "AM" in line


def test_describe_when_no_status_set() -> None:
    circle = _circle_with_two()
    svc = _status(circle)
    line = svc.describe("u-india", "u-us", now=NOW)
    assert "Bestie" in line
    assert "hasn't set a status" in line


def test_describe_denied_without_consent() -> None:
    circle = _circle_with_two()
    svc = _status(circle)
    with pytest.raises(errors.PermissionError):
        svc.describe("u-stranger", "u-us", now=NOW)


def test_place_and_safe_arrival_are_captured_and_spoken() -> None:
    circle = _circle_with_two()
    svc = _status(circle)
    svc.set_status("u-us", place="home", arrived_safe=True, now=NOW)
    got = svc.get_status("u-india", "u-us")
    assert got is not None
    assert got.place == "home"
    assert got.arrived_safe is True
    line = svc.describe("u-india", "u-us", now=NOW)
    assert "home" in line
    assert "safe" in line


def test_humanize_ago_buckets() -> None:
    assert humanize_ago(timedelta(seconds=10)) == "just now"
    assert humanize_ago(timedelta(seconds=90)) == "1 minute ago"
    assert humanize_ago(timedelta(minutes=20)) == "20 minutes ago"
    assert humanize_ago(timedelta(hours=1)) == "1 hour ago"
    assert humanize_ago(timedelta(hours=25)) == "yesterday"
    assert humanize_ago(timedelta(days=3)) == "3 days ago"
