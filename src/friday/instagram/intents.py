"""Voice intents for Instagram DMs: classify a spoken line, then act on it.

``parse_intent`` recognises only Instagram phrases (counting unread DMs, reading
them aloud, or replying to someone) and returns ``None`` for everything else so
the Siri pipeline falls through to the circle / orchestrator unchanged. Every
pattern requires an explicit Instagram token, so "tell mom I'll be late" never
collides with the circle's relay intents. ``handle_intent`` dispatches a parsed
intent to :class:`~friday.instagram.service.InstagramService`, which shapes the
spoken reply and swallows all client errors into soft messages.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from friday.instagram.service import InstagramService


@dataclass(frozen=True)
class CountDMs:
    """Ask how many unread Instagram DMs there are."""


@dataclass(frozen=True)
class ReadAloud:
    """Read the unread Instagram DMs aloud."""


@dataclass(frozen=True)
class ReplyDM:
    """Reply to ``name`` on Instagram with ``text``."""

    name: str
    text: str


Intent = CountDMs | ReadAloud | ReplyDM

_IG = r"\b(?:instagram|insta|ig)\b"
_MSG = r"\b(?:dm|dms|message|messages|inbox|notifications?)\b"
_REPLY = re.compile(
    r"\b(?:reply to|message|dm|tell|send)\s+(?P<name>.+?)\s+on\s+(?:instagram|insta|ig)\b"
    r"\s*(?:saying\s+|that\s+|:\s*)?(?P<text>.+)$"
)
_READ = re.compile(rf"\bread\b.*{_IG}.*\b(?:dm|dms|message|messages)\b")
# Either order: "any instagram dms" OR "any dms on instagram"; plus "check my ig".
_COUNT = re.compile(
    rf"(?:{_IG}.*{_MSG}|{_MSG}.*{_IG}"
    rf"|\bcheck\s+(?:my\s+)?(?:instagram|insta|ig)\b)"
)
#: A bare "read them aloud" with no Instagram token — handled by the Siri layer
#: only inside the just-asked-about-Instagram window (see ``siri_instagram``).
_READ_BARE = re.compile(r"\bread\s+(?:them|those|it|these)\b")


def _clean(value: str) -> str:
    return value.strip().strip("\"'").strip()


def parse_intent(text: str) -> Intent | None:
    """Return the Instagram intent in ``text``, or ``None`` if it isn't one.

    Order matters: reply is tried first (it's the most specific), then read (which
    must win over count, since "read my instagram messages" matches both), then
    count. No network: this is pure regex over the lowercased text.
    """
    low = text.strip().lower().rstrip(".!?")

    m = _REPLY.search(low)
    if m:
        name = _clean(m.group("name"))
        body = _clean(m.group("text"))
        if name and body:
            return ReplyDM(name=name, text=body)

    if _READ.search(low):
        return ReadAloud()
    if _COUNT.search(low):
        return CountDMs()
    return None


def is_bare_read(text: str) -> bool:
    """Whether ``text`` is a bare "read them/those/it/these" (no Instagram token)."""
    return bool(_READ_BARE.search(text.strip().lower().rstrip(".!?")))


def handle_intent(service: InstagramService, intent: Intent) -> str | None:
    """Act on ``intent`` via ``service``; returns a spoken string (never raises)."""
    if isinstance(intent, ReplyDM):
        return service.reply(intent.name, intent.text)
    if isinstance(intent, ReadAloud):
        return service.read_unread_aloud()
    if isinstance(intent, CountDMs):
        return service.unread_summary()
    return None
