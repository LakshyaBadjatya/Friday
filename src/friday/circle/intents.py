"""Voice intents for the circle: parse a spoken line, then act on it.

``parse_intent`` recognises only the circle's own phrases (asking what someone is
doing, or setting your own status/mood/place/safe-arrival); anything else returns
``None`` so the caller can fall through to the general assistant. ``handle_intent``
resolves a spoken name within the caller's circle and returns a spoken reply, or
``None`` when the name isn't someone in the circle (again, fall through) — so a
question like "how's the weather" still reaches the assistant.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime

from friday.circle.service import CircleService
from friday.circle.status import StatusService
from friday.errors import PermissionError


@dataclass(frozen=True)
class StatusQuery:
    """Ask what a named person is doing."""

    name: str


@dataclass(frozen=True)
class SetStatus:
    """Set the caller's own status (only the given fields)."""

    text: str | None = None
    mood: str | None = None
    place: str | None = None
    arrived_safe: bool | None = None


Intent = StatusQuery | SetStatus

_SET_STATUS = re.compile(r"^(?:hey friday,?\s+)?set my status to\s+(.+)$")
_SET_MOOD = re.compile(r"^(?:set my mood to|i(?:'m| am) feeling)\s+(.+)$")
# Anchored to a leading "i" so a question like "is the home safe" isn't mistaken
# for the caller reporting their own safe arrival.
_HOME_SAFE = re.compile(r"^(?:tell friday(?: that)?\s+)?i\b.*\bhome\b.*\bsafe")
_AT_PLACE = re.compile(r"^(?:tell friday(?: that)?\s+)?i(?:'m| am)\s+at\s+(.+)$")

_QUERY_PATTERNS = (
    re.compile(r"\bwhat(?:'s| is|s| are)?\s+(.+?)\s+(?:doing|up to)\b"),
    re.compile(r"\bhow(?:'s| is| are)?\s+(.+?)\s+doing\b"),
    re.compile(r"\bwhere(?:'s| is| are)?\s+(.+?)(?:\s+(?:right now|now|at))?$"),
    re.compile(r"\bis\s+(.+?)\s+(?:awake|asleep|free|around|there|up|busy|ok|okay)\b"),
    # "good time to call/reach X" and "can i call/text X" → ask about X's availability.
    re.compile(r"\bgood time to (?:call|reach|text|message|ring)\s+(.+?)$"),
    re.compile(r"\bcan i (?:call|reach|text|message)\s+(.+?)(?:\s+now)?$"),
)

_ME = {"me", "my", "myself", "i", "my status", "my self"}


def _clean(value: str) -> str:
    return value.strip().strip("\"'").strip()


def parse_intent(text: str) -> Intent | None:
    """Return the circle intent in ``text``, or ``None`` if it isn't one."""
    normalised = text.strip().lower().rstrip(".!?")

    m = _SET_STATUS.match(normalised)
    if m:
        return SetStatus(text=_clean(m.group(1)))
    m = _SET_MOOD.match(normalised)
    if m:
        return SetStatus(mood=_clean(m.group(1)))
    if _HOME_SAFE.search(normalised):
        return SetStatus(place="home", arrived_safe=True)
    m = _AT_PLACE.match(normalised)
    if m:
        return SetStatus(place=_clean(m.group(1)))

    for pattern in _QUERY_PATTERNS:
        m = pattern.search(normalised)
        if m:
            name = _clean(m.group(1))
            if name:
                return StatusQuery(name=name)
    return None


def _ack(intent: SetStatus) -> str:
    if intent.arrived_safe:
        return "Glad you made it home safe — I've let your circle know."
    if intent.mood:
        return f"Got it — feeling {intent.mood}."
    if intent.place:
        return f"Got it — you're at {intent.place}."
    return f"Got it — status set to {intent.text}."


def _resolve_member(
    circle: CircleService, caller_uid: str, name: str
) -> str | None:
    """Resolve a spoken ``name`` to a member uid in the caller's circle."""
    key = name.strip().lower()
    if key in _ME:
        member = circle.find_member(caller_uid)
        return member.uid if member else None
    for member in circle.members_visible_to(caller_uid):
        display = member.display_name.strip().lower()
        if display == key or key in display.split():
            return member.uid
    return None


def handle_intent(
    circle: CircleService,
    status: StatusService,
    caller_uid: str,
    intent: Intent,
    *,
    now: datetime,
) -> str | None:
    """Act on ``intent`` for ``caller_uid``; ``None`` means 'fall through'."""
    if isinstance(intent, SetStatus):
        status.set_status(
            caller_uid,
            text=intent.text,
            mood=intent.mood,
            place=intent.place,
            arrived_safe=intent.arrived_safe,
            now=now,
        )
        return _ack(intent)

    target_uid = _resolve_member(circle, caller_uid, intent.name)
    if target_uid is None:
        return None
    try:
        return status.describe(caller_uid, target_uid, now=now)
    except PermissionError:
        return None
