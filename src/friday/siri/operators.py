"""Calling a specialist by name, and meaning it.

FRIDAY has eight named operators — EDITH for security, VISION for research, and
so on — each a real :class:`~friday.roster.definitions.Persona` with its own
charter in ``roster/definitions.py``. Asked to "call EDITH", FRIDAY would happily
answer *"I'm calling in EDITH, your security specialist — she'll handle security
requests from now on"* and then change nothing at all. The next turn was plain
FRIDAY again. The claim was theatre, which is the same failure as a reminder that
is confirmed but never stored: a confident sentence with nothing behind it.

This module makes the sentence true. A "call X" turn pins X as the session's
active operator; every turn after it is answered in that operator's voice, from
that operator's charter, until the owner switches back. The pin lives per session,
so a Telegram thread and a Siri session can be talking to different specialists at
once without either leaking into the other.

Kept apart from the roster registry on purpose: that package is pure data and
lookup, with no notion of a conversation. This is the conversational layer over it.
"""

from __future__ import annotations

import re
from typing import Any

#: "call edith", "get me vision", "switch to oracle", "put me through to karen".
_CALL = re.compile(
    r"\b(?:call|get|bring)\s+(?:me\s+|in\s+|up\s+)?(?P<name>[a-z]+)\b"
    r"|\b(?:switch|transfer|put\s+me\s+through)\s+(?:me\s+)?to\s+(?P<name2>[a-z]+)\b"
    r"|\b(?:talk|speak)\s+to\s+(?P<name3>[a-z]+)\b"
    r"|\bask\s+(?P<name4>[a-z]+)\s+(?:about|to|for)\b",
    re.IGNORECASE,
)
#: Returning to FRIDAY herself ends the session's specialist pin.
_RELEASE = re.compile(
    r"\b(?:back\s+to\s+friday|friday\s+again|that'?s\s+all\s+(?:for\s+)?(?:now|thanks)"
    r"|thanks?\s+\w+\s*,?\s*that'?s\s+all|dismiss(?:ed)?|stand\s+down"
    r"|switch\s+back|never\s?mind)\b",
    re.IGNORECASE,
)
#: "who's on the roster" / "what operators do you have".
_LIST = re.compile(
    r"\b(?:who(?:'s|\s+is)?\s+(?:on\s+)?(?:the\s+)?roster|what\s+operators?"
    r"|list\s+(?:the\s+)?(?:operators?|specialists?)|who\s+can\s+you\s+call)\b",
    re.IGNORECASE,
)
#: Words that follow "call"/"ask" but are never an operator name — without this,
#: "call mum at 6" would try to put mum on the line.
_NOT_A_NAME = frozenset(
    {
        "me", "you", "him", "her", "them", "us", "it", "back", "again", "out",
        "up", "in", "on", "off", "mum", "mom", "dad", "home", "someone",
        "anyone", "everyone", "later", "now", "the", "a", "an", "my",
    }
)


def handle(state: Any, session_id: str, query: str) -> str | None:
    """Answer a roster intent (call / release / list), or ``None`` to fall through."""
    text = (query or "").strip()
    if not text:
        return None
    registry = getattr(state, "roster", None)
    if registry is None:
        return None

    if _LIST.search(text):
        return _speak_roster(registry)

    if _RELEASE.search(text) and active(state, session_id) is not None:
        name = active(state, session_id)
        release(state, session_id)
        return f"{name} standing down. Back with you, Boss."

    name = _spoken_name(text)
    if name is None:
        return None
    persona = _lookup(registry, name)
    if persona is None:
        return None  # not an operator — let the normal paths have the turn

    _pins(state)[session_id] = persona.name
    return (
        f"{persona.name} here, Boss — {persona.title}. "
        f"I'll stay on until you say back to Friday."
    )


def active(state: Any, session_id: str) -> str | None:
    """The operator currently pinned to this session, if any."""
    pinned = _pins(state).get(session_id)
    return pinned if isinstance(pinned, str) else None


def release(state: Any, session_id: str) -> None:
    """Hand the session back to FRIDAY herself."""
    _pins(state).pop(session_id, None)


def system_prompt(state: Any, session_id: str) -> str | None:
    """The pinned operator's charter, for use as the turn's system prompt.

    ``None`` when no operator is pinned or the roster cannot resolve it, so the
    caller falls back to the ordinary FRIDAY persona.
    """
    name = active(state, session_id)
    if name is None:
        return None
    persona = _lookup(getattr(state, "roster", None), name)
    if persona is None:
        return None
    return getattr(persona, "system_prompt", None) or None


def _spoken_name(text: str) -> str | None:
    """Pull the operator name out of a 'call X' phrasing, if there is one."""
    match = _CALL.search(text)
    if match is None:
        return None
    name = next(
        (
            group
            for group in (
                match.group("name"),
                match.group("name2"),
                match.group("name3"),
                match.group("name4"),
            )
            if group
        ),
        None,
    )
    if name is None:
        return None
    name = name.strip().lower()
    if name in _NOT_A_NAME or len(name) < 3:
        return None
    return name


def _lookup(registry: Any, name: str) -> Any:
    """Resolve a persona by name, or ``None`` when the roster has no such operator."""
    if registry is None:
        return None
    getter = getattr(registry, "get", None)
    if not callable(getter):
        return None
    try:
        return getter(name)
    except (KeyError, LookupError):
        return None


def _speak_roster(registry: Any) -> str:
    """Name the operators and what each is for."""
    try:
        people = list(registry.personas())
    except Exception:  # noqa: BLE001
        return "I can't reach the roster right now, Boss."
    if not people:
        return "There are no operators on the roster, Boss."
    parts = [f"{p.name} for {p.title.lower()}" for p in people]
    return "I can call " + ", ".join(parts[:-1]) + f", or {parts[-1]}, Boss."


def _pins(state: Any) -> dict[str, str]:
    """The per-session active-operator map, created on first use."""
    existing = getattr(state, "_operator_pins", None)
    if not isinstance(existing, dict):
        existing = {}
        state._operator_pins = existing  # noqa: SLF001 - app state is a namespace
    return existing
