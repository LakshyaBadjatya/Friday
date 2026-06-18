"""Unit tests for :mod:`friday.circle.friend_time`.

The long-distance core: given a reference UTC instant (passed in, so these are
deterministic with no clock), tell a viewer what their friend's local time is,
whether they're probably asleep, and whether now is a good time to reach them.
"""

from __future__ import annotations

from datetime import UTC, datetime

from friday.circle.friend_time import (
    describe_local,
    good_to_reach,
    is_probably_asleep,
    part_of_day,
    time_diff_hours,
)


def test_part_of_day_buckets() -> None:
    assert part_of_day(8) == "morning"
    assert part_of_day(14) == "afternoon"
    assert part_of_day(19) == "evening"
    assert part_of_day(2) == "night"
    assert part_of_day(23) == "night"


def test_india_is_9_5h_ahead_of_us_eastern_in_summer() -> None:
    # 2026-06-18 is DST in the US -> Eastern is EDT (UTC-4); IST is UTC+5:30.
    now = datetime(2026, 6, 18, 12, 0, tzinfo=UTC)
    diff = time_diff_hours("America/New_York", "Asia/Kolkata", now)
    assert round(diff, 1) == 9.5


def test_3am_in_new_york_reads_as_asleep_and_night() -> None:
    now = datetime(2026, 6, 18, 7, 0, tzinfo=UTC)  # 07:00 UTC = 03:00 EDT
    view = describe_local("America/New_York", now)
    assert view.asleep is True
    assert view.part_of_day == "night"
    assert ":" in view.local_time


def test_good_to_reach_is_false_when_the_friend_is_asleep() -> None:
    now = datetime(2026, 6, 18, 7, 0, tzinfo=UTC)  # 03:00 EDT / 12:30 IST
    # Viewer in India (awake) reaching friend in the US (asleep) -> bad time.
    assert good_to_reach("Asia/Kolkata", "America/New_York", now) is False
    # The reverse: friend in India is mid-day -> good time.
    assert good_to_reach("America/New_York", "Asia/Kolkata", now) is True


def test_is_probably_asleep_window() -> None:
    midnight = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)  # 00:00 in UTC zone
    noon = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    assert is_probably_asleep(describe_local("UTC", midnight)._dt) is True
    assert is_probably_asleep(describe_local("UTC", noon)._dt) is False


def test_describe_local_exposes_spoken_fields() -> None:
    now = datetime(2026, 6, 18, 16, 30, tzinfo=UTC)  # 22:00 IST
    view = describe_local("Asia/Kolkata", now)
    assert view.tz == "Asia/Kolkata"
    assert view.part_of_day == "night"
    assert view.asleep is False  # 10pm is late but not yet the sleep window
