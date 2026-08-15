"""Turn a spoken time into a stored timestamp.

"Remind me to call mum at six" carries a time a person *means*: six in the
evening, where they are standing, today unless that has already passed. The
reminder store holds UTC ISO-8601 strings. This module is the translation between
the two, and it is its own file because that translation is where the interesting
mistakes live — an off-by-one on "tomorrow", a bare "6" read as 6 AM when nobody
means 6 AM, a naive datetime compared against an aware one.

Distinct from :mod:`friday.circle.friend_time`, which converts a *known* instant
between zones; nothing there turns words into an instant.

Rules encoded here, each chosen to match how people actually speak:

* **Bare hours lean waking.** "at 6" means 18:00 — an unqualified 1-6 reads as
  PM, 7-11 as AM, because "meet me at 8" is breakfast and "at 4" is tea.
* **Times already past roll forward.** "at 9" said at 10 PM means 9 tomorrow; a
  reminder for a moment that has gone is never what was meant.
* **No new dependency.** ``dateutil`` is not in this project and one parser does
  not justify adding it, so this is stdlib :mod:`zoneinfo` plus regex.

:func:`parse_when` returns ``(utc_iso, spoken_description)`` so the caller can
confirm aloud what it understood — "tomorrow at 6 PM" — which is the only way the
user catches a misparse before the reminder silently fires at the wrong time.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

#: Weekday names -> ``datetime.weekday()`` index, for "on friday".
_WEEKDAYS = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
    "mon": 0, "tue": 1, "tues": 1, "wed": 2, "thu": 3, "thur": 3, "thurs": 3,
    "fri": 4, "sat": 5, "sun": 6,
}
#: Spoken number words for hours and small durations.
_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "fifteen": 15, "twenty": 20, "thirty": 30, "half": 30, "an": 1, "a": 1,
}
#: Implied hour for a part of the day said on its own ("tomorrow morning").
_PARTS_OF_DAY = {
    "midnight": 0, "morning": 9, "noon": 12, "midday": 12,
    "afternoon": 15, "evening": 19, "tonight": 20, "night": 21,
}

_HOUR_WORDS = "|".join(sorted(_NUMBER_WORDS, key=len, reverse=True))
_DAY_WORDS = "|".join(sorted(_WEEKDAYS, key=len, reverse=True))

#: ``at 6``, ``at 6:30``, ``at six``, each optionally ``am``/``pm``/``o'clock``.
_AT_TIME = re.compile(
    rf"\bat\s+(?P<hour>\d{{1,2}}|{_HOUR_WORDS})"
    r"(?:[:.](?P<minute>\d{2}))?\s*(?P<meridiem>a\.?m\.?|p\.?m\.?|o'?clock)?\b",
    re.IGNORECASE,
)
#: A bare ``6pm`` / ``6:30 am`` with no leading "at".
_BARE_TIME = re.compile(
    r"\b(?P<hour>\d{1,2})(?:[:.](?P<minute>\d{2}))?\s*"
    r"(?P<meridiem>a\.?m\.?|p\.?m\.?)\b",
    re.IGNORECASE,
)
#: ``in 10 minutes`` / ``in an hour`` / ``in half an hour`` / ``in 2 days``.
_IN_DURATION = re.compile(
    rf"\bin\s+(?P<count>\d+|{_HOUR_WORDS})\s*(?:an?\s+)?"
    r"(?P<unit>min(?:ute)?s?|hours?|hrs?|days?|weeks?)\b",
    re.IGNORECASE,
)
#: ``on friday`` / ``next monday`` / ``this saturday``.
_WEEKDAY_RE = re.compile(rf"\b(?:on|next|this)?\s*(?P<day>{_DAY_WORDS})\b", re.IGNORECASE)
#: Time phrasing stripped from the reminder's own text (it lives in ``due_at``).
_STRIP = re.compile(
    r"\b(?:at\s+\d{1,2}(?:[:.]\d{2})?\s*(?:a\.?m\.?|p\.?m\.?|o'?clock)?"
    rf"|at\s+(?:{_HOUR_WORDS})(?:\s*o'?clock)?"
    r"|in\s+(?:\d+|an?|half)\s*(?:an?\s+)?(?:min(?:ute)?s?|hours?|hrs?|days?|weeks?)"
    rf"|tomorrow|today|tonight|next\s+(?:{_DAY_WORDS}|week)|this\s+(?:{_DAY_WORDS})"
    rf"|on\s+(?:{_DAY_WORDS})"
    r"|\d{1,2}(?:[:.]\d{2})?\s*(?:a\.?m\.?|p\.?m\.?)"
    r"|in\s+the\s+(?:morning|afternoon|evening)"
    r"|morning|afternoon|evening|night|noon|midday|midnight)\b",
    re.IGNORECASE,
)


def zone(name: str) -> ZoneInfo:
    """The configured IANA zone, or UTC when the name is unusable.

    A bad ``FRIDAY_TIMEZONE`` must not take the assistant down — it degrades to
    UTC, which is wrong by a fixed offset rather than broken.
    """
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError, KeyError):
        return ZoneInfo("UTC")


def parse_when(
    text: str, *, tz_name: str, now: datetime | None = None
) -> tuple[str, str] | None:
    """Resolve a spoken time to ``(utc_iso, spoken_description)``, or ``None``.

    ``None`` means no time was spoken — a valid undated reminder the user will
    ask about later, not an error.
    """
    tz = zone(tz_name)
    local_now = (now or datetime.now(UTC)).astimezone(tz)
    low = text.lower()

    target = _relative(low, local_now) or _absolute(low, local_now, tz)
    if target is None:
        return None
    if target <= local_now:
        target += timedelta(days=1)  # a moment that has passed is never meant
    return target.astimezone(UTC).isoformat(), _describe(target, local_now)


def strip_time_words(text: str) -> str:
    """Remove the time phrasing so the reminder reads as the task alone.

    "call mum at 6 tomorrow" -> "call mum". Without this the stored text repeats
    a time that is already in ``due_at``, and it gets read back twice.
    """
    cleaned = _STRIP.sub(" ", text)
    return re.sub(r"\s+", " ", cleaned).strip(" ,.")


def _relative(low: str, now: datetime) -> datetime | None:
    """``in 10 minutes`` / ``in an hour`` / ``in 2 days``."""
    match = _IN_DURATION.search(low)
    if match is None:
        return None
    raw = match.group("count").lower()
    count = int(raw) if raw.isdigit() else _NUMBER_WORDS.get(raw, 0)
    if count <= 0:
        return None
    unit = match.group("unit").lower()
    if unit.startswith("min"):
        return now + timedelta(minutes=count)
    if unit.startswith(("hour", "hr")):
        return now + timedelta(hours=count)
    if unit.startswith("day"):
        return now + timedelta(days=count)
    return now + timedelta(weeks=count)


def _absolute(low: str, now: datetime, tz: ZoneInfo) -> datetime | None:
    """A named day and/or a clock time — ``tomorrow at 6``, ``friday morning``."""
    day = _day_offset(low, now)
    clock = _clock(low)
    if clock is None and day is None:
        return None
    if clock is None:
        clock = (9, 0)  # a day with no time at all: assume the morning
    base = day or now
    return base.replace(hour=clock[0], minute=clock[1], second=0, microsecond=0, tzinfo=tz)


def _day_offset(low: str, now: datetime) -> datetime | None:
    """Resolve ``tomorrow`` / ``tonight`` / ``on friday`` to a local date."""
    if re.search(r"\btomorrow\b", low):
        return now + timedelta(days=1)
    if re.search(r"\b(?:today|tonight)\b", low):
        return now
    match = _WEEKDAY_RE.search(low)
    if match is None:
        return None
    ahead = (_WEEKDAYS[match.group("day").lower()] - now.weekday()) % 7
    if ahead == 0:
        ahead = 7  # "on friday" said on a Friday means the next one
    return now + timedelta(days=ahead)


def _clock(low: str) -> tuple[int, int] | None:
    """Extract ``(hour, minute)`` in 24-hour form from a spoken clock time."""
    match = _AT_TIME.search(low) or _BARE_TIME.search(low)
    if match is None:
        for word, hour in _PARTS_OF_DAY.items():
            if re.search(rf"\b{word}\b", low):
                return hour, 0
        return None

    raw_hour = match.group("hour").lower()
    hour = int(raw_hour) if raw_hour.isdigit() else _NUMBER_WORDS.get(raw_hour, -1)
    if not 0 <= hour <= 23:
        return None
    minute = int(match.group("minute") or 0)
    if not 0 <= minute <= 59:
        return None

    meridiem = (match.group("meridiem") or "").replace(".", "").lower()
    if meridiem.startswith("p") and hour < 12:
        hour += 12
    elif meridiem.startswith("a") and hour == 12:
        hour = 0
    elif not meridiem.startswith(("a", "p")):
        # Unqualified. Follow the part of day if one was said ("6 in the
        # evening"), else lean waking hours: 1-6 is afternoon, 7-11 morning.
        if re.search(r"\b(?:evening|night|tonight|afternoon)\b", low) and hour < 12:
            hour += 12
        elif re.search(r"\bmorning\b", low):
            pass
        elif 1 <= hour <= 6:
            hour += 12
    return hour, minute


def _describe(target: datetime, now: datetime) -> str:
    """A short spoken confirmation — "today at 6 PM", "Saturday at 9 AM"."""
    clock = target.strftime("%-I:%M %p") if target.minute else target.strftime("%-I %p")
    days = (target.date() - now.date()).days
    if days == 0:
        return f"today at {clock}"
    if days == 1:
        return f"tomorrow at {clock}"
    if 0 < days < 7:
        return f"{target.strftime('%A')} at {clock}"
    return f"{target.strftime('%-d %B')} at {clock}"
