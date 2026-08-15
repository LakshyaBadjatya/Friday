"""The rest of the desk, by voice: briefing, journal, protocols and facts.

Four subsystems that were fully built and completely unreachable from a phone.
Each has a store and a REST route; none had any way to be *spoken* to, so asking
"brief me" or "what did I do yesterday" fell through to the general chat path and
got an answer invented from nothing.

They share one module because they share one shape — match a phrase, call a store,
speak the result — and splitting four thirty-line handlers across four files would
be filing, not architecture. Anything genuinely different lives elsewhere:
reminders need time parsing (:mod:`friday.siri.tasks`), drilling needs a running
session (:mod:`friday.siri.drill`).

Every handler returns ``None`` when the words are not its business, so
:func:`handle` can try each in turn and fall through to the model untouched.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Any

#: "brief me" / "what's my briefing" / "catch me up".
_BRIEF = re.compile(
    r"\b(?:brief\s+me|my\s+brief(?:ing)?|daily\s+brief(?:ing)?|what'?s\s+my\s+brief"
    r"|morning\s+brief(?:ing)?|catch\s+me\s+up)\b",
    re.IGNORECASE,
)
#: "log this: X" / "journal that X" / "note that X" — writing the day down.
_JOURNAL_WRITE = re.compile(
    r"^(?:hey\s+friday[,\s]*)?(?:please\s+)?"
    r"(?:log|journal)\s+(?:this|that|it)?[:,\s]+(?P<entry>.+)$",
    re.IGNORECASE,
)
#: "what did I do today/yesterday" — reading the day back.
_JOURNAL_READ = re.compile(
    r"\bwhat\s+did\s+i\s+(?:do|get\s+done|work\s+on)\s+(?P<when>today|yesterday)\b",
    re.IGNORECASE,
)
#: "run my morning protocol" / "start the shutdown routine".
_PROTOCOL_RUN = re.compile(
    r"\b(?:run|start|execute|kick\s+off)\s+(?:my\s+|the\s+)?(?P<name>[\w\s]+?)"
    r"\s*(?:protocol|routine)\b",
    re.IGNORECASE,
)
#: "what protocols do I have" / "list my routines".
_PROTOCOL_LIST = re.compile(
    r"\b(?:what|list|which)\s+(?:are\s+)?(?:my\s+)?(?:protocols?|routines?)\b",
    re.IGNORECASE,
)
#: Asking about the owner. "Who is Lakshya" is not a general-knowledge question —
#: there are other people with that name and a model will happily describe one of
#: them. In this assistant it always means *the owner*, so it is answered from his
#: own stored facts rather than from whatever the model has read.
_OWNER = re.compile(
    r"\bwho\s+(?:is|was)\s+(?:lakshya(?:\s+badjatya)?|my\s+(?:master|owner|boss))\b"
    r"|\btell\s+me\s+about\s+(?:lakshya(?:\s+badjatya)?|myself|me)\b"
    r"|\bwho\s+am\s+i\b"
    r"|\bwhat\s+do\s+you\s+know\s+about\s+(?:me|lakshya(?:\s+badjatya)?)\b",
    re.IGNORECASE,
)
#: The part of the answer that never changes, whatever memory holds.
#: Short on purpose. The old version recited every surface she runs on, which
#: nobody asked about and which reads like a product description.
_OWNER_BASE = "that's my Boss — Lakshya. he built me."

#: "remember that X" / "keep in mind that X" — a durable fact about the user.
_FACT_WRITE = re.compile(
    r"^(?:hey\s+friday[,\s]*)?(?:please\s+)?"
    r"(?:remember|keep\s+in\s+mind|note)\s+(?:that\s+|this[:,\s]+)(?P<fact>.+)$",
    re.IGNORECASE,
)
#: "what do you know about X" / "what do you remember about X".
_FACT_READ = re.compile(
    r"\bwhat\s+do\s+you\s+(?:know|remember)\s+about\s+(?P<topic>.+?)\s*[?.]?$",
    re.IGNORECASE,
)

#: Facts and journal lines read aloud before the rest is summarised.
_SPEAK_LIMIT = 4


async def handle(state: Any, query: str, now: datetime) -> str | None:
    """Try each desk intent in turn; ``None`` when none of them apply.

    ``state`` is the app state, read with ``getattr`` throughout so a surface
    whose store was never wired falls through rather than raising.
    """
    text = (query or "").strip()
    if not text:
        return None
    return (
        _owner(state, text)
        or _facts(state, text)
        or _protocols_list(state, text)
        or await _protocol_run(state, text)
        or _journal(state, text, now)
        or await _brief(state, text, now)
    )


async def _brief(state: Any, text: str, now: datetime) -> str | None:
    """Speak the daily briefing on demand.

    ``/siri/digest`` already builds this, but only a cron could ask for it and it
    only ever landed in Telegram — there was no way to simply request it aloud.
    """
    if not _BRIEF.search(text):
        return None
    service = getattr(state, "briefing", None)
    if service is None:
        return "My briefing isn't wired up yet, Boss."
    try:
        briefing = await service.build(now)
    except Exception:  # noqa: BLE001 - a failed briefing is spoken, never raised
        return "I couldn't put your briefing together, Boss."

    parts = [getattr(briefing, "greeting", "") or "Here's your briefing, Boss."]
    for section in getattr(briefing, "sections", None) or []:
        items = ". ".join(getattr(section, "items", None) or [])
        if items:
            parts.append(f"{getattr(section, 'title', '')}: {items}.")
    return " ".join(part for part in parts if part.strip())


def _journal(state: Any, text: str, now: datetime) -> str | None:
    """Write a line into today's journal, or read a day back."""
    written = _JOURNAL_WRITE.match(text)
    if written is not None:
        store = getattr(state, "journal_store", None)
        entry = written.group("entry").strip()
        if store is None:
            return "My journal isn't wired up yet, Boss."
        try:
            _append_journal_line(store, now, entry)
        except Exception:  # noqa: BLE001
            return "I couldn't write that down, Boss."
        return f"Logged it, Boss — {entry}"

    asked = _JOURNAL_READ.search(text)
    if asked is None:
        return None
    store = getattr(state, "journal_store", None)
    if store is None:
        return "My journal isn't wired up yet, Boss."
    when = asked.group("when").lower()
    day = now.date() if when == "today" else (now - timedelta(days=1)).date()
    try:
        found = store.get(day.isoformat())
    except Exception:  # noqa: BLE001
        return "I couldn't reach your journal, Boss."
    if found is None:
        return f"Nothing in the journal for {when}, Boss."
    return _speak_entry(found)


async def _protocol_run(state: Any, text: str) -> str | None:
    """Run a named protocol — "run my morning routine"."""
    match = _PROTOCOL_RUN.search(text)
    if match is None:
        return None
    store = getattr(state, "protocol_store", None)
    runner = getattr(state, "protocol_runner", None)
    if store is None or runner is None:
        return "Protocols aren't wired up yet, Boss."

    name = match.group("name").strip()
    protocol = _find_protocol(store, name)
    if protocol is None:
        return f"I don't have a protocol called {name}, Boss."
    try:
        await runner.run(protocol)
    except Exception:  # noqa: BLE001 - a failed run is reported, never raised
        return f"I started {protocol.name} but it didn't finish cleanly, Boss."
    return f"Ran your {protocol.name} protocol, Boss."


def _protocols_list(state: Any, text: str) -> str | None:
    """Name the protocols that exist."""
    if not _PROTOCOL_LIST.search(text):
        return None
    store = getattr(state, "protocol_store", None)
    if store is None:
        return "Protocols aren't wired up yet, Boss."
    try:
        items = [p for p in store.list_protocols() if getattr(p, "enabled", True)]
    except Exception:  # noqa: BLE001
        return "I couldn't reach your protocols, Boss."
    if not items:
        return "You haven't set up any protocols yet, Boss."
    names = ", ".join(p.name for p in items[:_SPEAK_LIMIT])
    return f"You have {names}, Boss."


def _owner(state: Any, text: str) -> str | None:
    """Answer "who is Lakshya" from the owner's own stored facts.

    Handled here rather than left to the model for a specific reason: there are
    other people with that name, and asked cold a model describes one of them, or
    invents a plausible biography. In this assistant the name always means the
    owner. The fixed half of the answer states who he is; the rest is whatever he
    has actually told FRIDAY to remember, so it grows as he adds to it and stays
    true because none of it was guessed.
    """
    if not _OWNER.search(text):
        return None
    store = getattr(state, "long_term", None)
    remembered: list[str] = []
    if store is not None:
        try:
            # His facts are stored as plain sentences, so there is no single
            # keyword to match on — take the recent ones and let the base answer
            # carry the rest.
            remembered = [
                (getattr(f, "text", "") or "").strip().rstrip(".")
                for f in store.all_facts(limit=_SPEAK_LIMIT)
            ]
        except Exception:  # noqa: BLE001 - memory is a bonus, not a dependency
            remembered = []
    remembered = [line for line in remembered if line]
    if not remembered:
        return _OWNER_BASE
    # One thing, not the file. Reciting everything reads like a database being
    # queried rather than someone who knows him, and "here's what you've had me
    # remember" narrates the mechanism — nobody says that about a friend.
    return f"{_OWNER_BASE} {_one_thing(remembered)}"


#: How she drops a known fact in: as knowledge, never as a lookup.
_ASIDES = (
    "{fact}, if that's what you're after.",
    "also {lower}.",
    "{fact}.",
    "he's the one {lower}, if that helps.",
)


def _one_thing(facts: list[str]) -> str:
    """Mention a single thing she knows, phrased like knowing rather than recall."""
    import secrets  # noqa: PLC0415

    fact = secrets.choice(facts).rstrip(".")
    # First clause only — the stored facts are dense paragraphs, and reading a
    # whole one out is a briefing rather than an answer.
    clipped = fact.split(". ")[0]
    if len(clipped) > 140:
        clipped = clipped[:140].rsplit(" ", 1)[0] + "…"
    lower = (clipped[0].lower() + clipped[1:]) if clipped else clipped
    return secrets.choice(_ASIDES).format(fact=clipped, lower=lower)


def _facts(state: Any, text: str) -> str | None:
    """Remember a durable fact, or answer what is known about a topic."""
    written = _FACT_WRITE.match(text)
    if written is not None:
        store = getattr(state, "long_term", None)
        fact = written.group("fact").strip().rstrip(".")
        if store is None:
            return "My long-term memory isn't wired up yet, Boss."
        try:
            store.add_fact(fact, "siri")
        except Exception:  # noqa: BLE001
            return "I couldn't hold on to that, Boss."
        return f"Noted, Boss — I'll remember that {fact}."

    asked = _FACT_READ.search(text)
    if asked is None:
        return None
    store = getattr(state, "long_term", None)
    if store is None:
        return None  # fall through to the model rather than claim amnesia
    topic = asked.group("topic").strip()
    try:
        found = store.query_facts(topic, limit=_SPEAK_LIMIT)
    except Exception:  # noqa: BLE001
        return None
    if not found:
        # Nothing *stored* is not the same as nothing knowable — let the model try.
        return None
    lines = ". ".join((getattr(f, "text", "") or "").rstrip(".") for f in found)
    return f"Here's what I have on {topic}, Boss. {lines}."


def _find_protocol(store: Any, name: str) -> Any:
    """Resolve a spoken protocol name, exactly first then loosely."""
    getter = getattr(store, "get_by_name", None)
    if callable(getter):
        try:
            found = getter(name)
        except Exception:  # noqa: BLE001
            found = None
        if found is not None:
            return found
    try:
        candidates = store.list_protocols()
    except Exception:  # noqa: BLE001
        return None
    low = name.lower()
    return next(
        (p for p in candidates if low in (getattr(p, "name", "") or "").lower()), None
    )


def _append_journal_line(store: Any, now: datetime, entry: str) -> None:
    """Add one spoken line to today's entry, creating the day if needed."""
    from friday.journal import JournalEntry  # noqa: PLC0415

    day = now.date().isoformat()
    existing = store.get(day)
    body = (getattr(existing, "summary", "") or "") if existing is not None else ""
    merged = f"{body}\n{entry}".strip() if body else entry
    store.save(JournalEntry(date=day, summary=merged))


def _speak_entry(entry: Any) -> str:
    """Read a journal entry back as plain spoken prose."""
    summary = (getattr(entry, "summary", "") or "").strip()
    if not summary:
        return "That day's entry is empty, Boss."
    lines = [line.strip() for line in summary.splitlines() if line.strip()]
    head = ". ".join(lines[:_SPEAK_LIMIT])
    extra = len(lines) - _SPEAK_LIMIT
    return f"{head}." + (f" And {extra} more." if extra > 0 else "")
