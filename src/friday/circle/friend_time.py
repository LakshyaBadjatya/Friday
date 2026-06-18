"""Timezone-aware helpers for a long-distance circle.

Given a reference UTC instant (always passed in by the caller, never read from a
clock here, so every result is deterministic and unit-testable), answer the
questions a long-distance feature needs: what is the friend's *local* time, is it
morning/afternoon/evening/night, are they probably asleep, and is now a good time
to reach them. All timezones are IANA names (e.g. ``"Asia/Kolkata"``,
``"America/New_York"``) resolved via the stdlib :mod:`zoneinfo`, so DST is handled
correctly for the given instant.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

#: Inclusive-exclusive "probably asleep" window in local hours: [23:00, 07:00).
_SLEEP_START_HOUR = 23
_SLEEP_END_HOUR = 7


def _to_local(tz_name: str, utc_now: datetime) -> datetime:
    """Convert ``utc_now`` into ``tz_name`` local time.

    A naive ``utc_now`` is assumed to be UTC; an aware one is honoured as given.
    """
    if utc_now.tzinfo is None:
        utc_now = utc_now.replace(tzinfo=UTC)
    return utc_now.astimezone(ZoneInfo(tz_name))


def part_of_day(hour: int) -> str:
    """Bucket a local 24h ``hour`` into morning/afternoon/evening/night."""
    if 5 <= hour < 12:
        return "morning"
    if 12 <= hour < 17:
        return "afternoon"
    if 17 <= hour < 21:
        return "evening"
    return "night"


def is_probably_asleep(local_dt: datetime) -> bool:
    """Whether a local datetime falls in the nominal sleep window [23:00, 07:00)."""
    hour = local_dt.hour
    return hour >= _SLEEP_START_HOUR or hour < _SLEEP_END_HOUR


def time_diff_hours(viewer_tz: str, friend_tz: str, utc_now: datetime) -> float:
    """Hours the ``friend_tz`` is *ahead of* ``viewer_tz`` at ``utc_now``.

    Positive means the friend is ahead (later in the day). DST-correct because the
    offsets are evaluated at the given instant.
    """
    viewer_offset = _to_local(viewer_tz, utc_now).utcoffset()
    friend_offset = _to_local(friend_tz, utc_now).utcoffset()
    # utcoffset() is never None for a zoneinfo-aware datetime.
    assert viewer_offset is not None and friend_offset is not None
    return (friend_offset - viewer_offset).total_seconds() / 3600.0


def good_to_reach(viewer_tz: str, friend_tz: str, utc_now: datetime) -> bool:
    """Whether now is a sociable time to reach the friend (they're likely awake).

    ``viewer_tz`` is accepted for symmetry/future use (e.g. mutual-overlap logic);
    the current rule is simply that the friend is not in their sleep window.
    """
    return not is_probably_asleep(_to_local(friend_tz, utc_now))


@dataclass(frozen=True)
class LocalView:
    """A spoken-ready snapshot of a person's local time."""

    tz: str
    local_time: str  # e.g. "3:04 AM"
    part_of_day: str
    asleep: bool
    _dt: datetime  # the underlying local datetime (for further computation)


def describe_local(tz_name: str, utc_now: datetime) -> LocalView:
    """Build a :class:`LocalView` for ``tz_name`` at ``utc_now``."""
    local_dt = _to_local(tz_name, utc_now)
    # %-I (no leading zero) is glibc-specific; fall back to stripping it so the
    # spoken time reads "3:04 AM" on any platform.
    hour12 = local_dt.strftime("%I").lstrip("0") or "12"
    local_time = f"{hour12}:{local_dt.strftime('%M %p')}"
    return LocalView(
        tz=tz_name,
        local_time=local_time,
        part_of_day=part_of_day(local_dt.hour),
        asleep=is_probably_asleep(local_dt),
        _dt=local_dt,
    )
