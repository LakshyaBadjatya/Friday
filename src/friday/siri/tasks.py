"""Reminders by voice — the intent seam over the existing reminder store.

The store (:mod:`friday.reminders.store`), the ``/reminders`` REST surface, the
scheduler loop and the registry tools in :mod:`friday.tools.reminders` all predate
this module. What was missing was any way to *speak* to them: the registry tools
take already-structured arguments and are reachable only through the orchestrator,
which the Siri fast path deliberately bypasses — so "remind me to call mum at six"
got a friendly chat reply while nothing was ever stored.

Three intents, matched by regex before any model is consulted, because a reminder
must be recorded exactly as dictated and a paraphrase is a bug:

* **create** — "remind me to X at Y", "set a reminder to X"
* **list** — "what are my reminders", "what's on my list"
* **complete** — "mark X done", "I've finished X"

:func:`handle` returns the spoken reply, or ``None`` when the words are not about
reminders, so the caller falls through untouched. Every reply names what it
understood ("Got it — call mum, tomorrow at 6 PM"): a reminder stored against a
misheard time is worse than one that was never stored, and the read-back is the
only moment the user can catch it.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from friday.siri.when import parse_when, strip_time_words, zone

#: "remind me to X" / "set a reminder to X" / "don't let me forget to X".
_CREATE = re.compile(
    r"^(?:hey\s+friday[,\s]*)?(?:please\s+)?"
    r"(?:remind\s+me\s+(?:to|about|that)?|set\s+(?:a\s+)?reminder\s+(?:to|for|about)?"
    r"|add\s+(?:a\s+)?reminder\s+(?:to|for|about)?|don'?t\s+let\s+me\s+forget\s+to)"
    r"\s+(?P<task>.+)$",
    re.IGNORECASE,
)
#: Asking what is outstanding.
_LIST = re.compile(
    r"\b(?:what(?:'s|\s+are|\s+is)?\s+(?:my\s+)?(?:reminders?|on\s+my\s+list|due)"
    r"|list\s+(?:my\s+)?reminders?|any\s+reminders?|read\s+(?:me\s+)?my\s+reminders?"
    r"|my\s+reminders?|show\s+(?:me\s+)?(?:my\s+)?reminders?"
    r"|reminders?\s*\?|upcoming\s+reminders?|do\s+i\s+have\s+any\s+reminders?"
    r"|what\s+do\s+i\s+have\s+(?:to\s+do|coming\s+up))\b",
    re.IGNORECASE,
)

#: Anything that mentions reminders at all. A model asked "my reminders?" will
#: happily invent "a meeting at 2 PM, a birthday gift for your sister" — it has no
#: way to know it is guessing. So once the word appears and a store is wired, the
#: store answers, always. Falling through to the model here does not risk a vague
#: reply, it risks a confident fabrication about the owner's own life.
_ABOUT_REMINDERS = re.compile(r"\breminders?\b|\bmy\s+list\b", re.IGNORECASE)
#: Marking something finished.
_COMPLETE = re.compile(
    r"\b(?:mark|tick)\s+(?:off\s+)?(?P<what>.+?)\s+(?:as\s+)?(?:done|completed?|off)\b"
    r"|\bi(?:'ve)?\s+(?:did|done|finished|completed)\s+(?P<what2>.+)$",
    re.IGNORECASE,
)

#: Spoken recurrence -> ``(stored value, the noun to say back)``. The pair exists
#: because the stored value and the spoken one differ: "daily" is what the column
#: holds, "every day" is what a person hears without wincing.
_RECURRENCE = (
    (re.compile(r"\bevery\s+day\b|\bdaily\b", re.IGNORECASE), "daily", "day"),
    (re.compile(r"\bevery\s+week\b|\bweekly\b", re.IGNORECASE), "weekly", "week"),
    (re.compile(r"\bevery\s+month\b|\bmonthly\b", re.IGNORECASE), "monthly", "month"),
)
#: The recurrence phrasing itself, removed from the task text — it lives in the
#: ``recurrence`` column, so leaving it in makes the read-back say it twice.
_STRIP_RECURRENCE = re.compile(
    r"\b(?:every\s+(?:day|week|month)|daily|weekly|monthly)\b", re.IGNORECASE
)
#: How many outstanding reminders to read before summarising the remainder.
_SPEAK_LIMIT = 5


def handle(store: Any, query: str, now: datetime, *, tz_name: str = "UTC") -> str | None:
    """Answer a reminder intent, or ``None`` when this is not one."""
    if store is None:
        return None
    text = (query or "").strip()
    if not text:
        return None

    if _LIST.search(text):
        return _speak_list(store, now, tz_name)
    done = _COMPLETE.search(text)
    if done is not None:
        return _speak_complete(store, done.group("what") or done.group("what2") or "")
    created = _CREATE.match(text)
    if created is not None:
        return _speak_create(store, created.group("task"), now, tz_name)

    # The word came up but no intent matched ("anything on my reminders?",
    # "reminders for tomorrow"). Read the list rather than hand the turn to a
    # model that would invent one. A slightly-off answer from the store beats a
    # fluent fabrication about the owner's own commitments.
    if _ABOUT_REMINDERS.search(text):
        return _speak_list(store, now, tz_name)
    return None


def _speak_create(store: Any, task: str, now: datetime, tz_name: str) -> str:
    """Store one reminder and confirm both halves of it out loud."""
    when = parse_when(task, tz_name=tz_name, now=now)
    repeat = next(((stored, said) for rx, stored, said in _RECURRENCE if rx.search(task)), None)
    body = strip_time_words(_STRIP_RECURRENCE.sub(" ", task)).strip(" ,.") or task.strip()

    try:
        store.add(body, when[0] if when else None, repeat[0] if repeat else None)
    except Exception:  # noqa: BLE001 - never lose the turn over a storage error
        return "I couldn't save that reminder, Boss. Try again in a moment."

    if when is None:
        return f"Saved, Boss — {body}. No time on it, so ask me when you need it."
    every = f", every {repeat[1]}" if repeat else ""
    return f"Got it, Boss — {body}, {when[1]}{every}."


def _speak_list(store: Any, now: datetime, tz_name: str) -> str:
    """Read the outstanding reminders, soonest first."""
    items = _open_reminders(store)
    if items is None:
        return "I couldn't reach your reminders, Boss."
    if not items:
        return "Nothing on your list, Boss."

    items.sort(key=lambda r: (getattr(r, "due_at", None) is None, getattr(r, "due_at", "") or ""))
    spoken = [_one_line(r, now, tz_name) for r in items[:_SPEAK_LIMIT]]
    head = "one reminder" if len(items) == 1 else f"{len(items)} reminders"
    extra = len(items) - _SPEAK_LIMIT
    more = f" And {extra} more." if extra > 0 else ""
    return f"You have {head}, Boss. " + " ".join(spoken) + more


def _speak_complete(store: Any, phrase: str) -> str:
    """Close the reminder whose text best matches what was said."""
    needle = strip_time_words(phrase).lower().strip(" .,")
    if not needle:
        return "Which one should I mark done, Boss?"
    items = _open_reminders(store)
    if items is None:
        return "I couldn't reach your reminders, Boss."

    match = next(
        (r for r in items if needle in (getattr(r, "text", "") or "").lower()), None
    )
    if match is None:
        return f"I don't have an open reminder about {needle}, Boss."
    try:
        store.complete(match.id)
    except Exception:  # noqa: BLE001
        return "I couldn't update that one, Boss."
    return f"Done — {match.text} is off your list, Boss."


def _open_reminders(store: Any) -> list[Any] | None:
    """Every open reminder, or ``None`` when the store cannot be read."""
    try:
        items = list(store.list_reminders())
    except Exception:  # noqa: BLE001
        return None
    return [r for r in items if getattr(r, "status", "open") == "open"]


def _one_line(reminder: Any, now: datetime, tz_name: str) -> str:
    """One reminder as a short spoken clause."""
    text = (getattr(reminder, "text", "") or "").strip()
    due = getattr(reminder, "due_at", None)
    if not due:
        return f"{text}."
    return f"{text}, {describe_due(due, now, tz_name)}."


def describe_due(due_at: str, now: datetime, tz_name: str) -> str:
    """Render a stored UTC timestamp as the local phrasing a person would say."""
    try:
        moment = datetime.fromisoformat(due_at)
    except (TypeError, ValueError):
        return "at some point"
    tz = zone(tz_name)
    local = moment.astimezone(tz)
    local_now = now.astimezone(tz)
    clock = local.strftime("%-I:%M %p") if local.minute else local.strftime("%-I %p")
    days = (local.date() - local_now.date()).days
    if days == 0:
        return f"today at {clock}"
    if days == 1:
        return f"tomorrow at {clock}"
    if days == -1:
        return f"yesterday at {clock}, overdue"
    if days < 0:
        return f"{local.strftime('%-d %B')}, overdue"
    if days < 7:
        return f"{local.strftime('%A')} at {clock}"
    return f"{local.strftime('%-d %B')} at {clock}"
