"""Turn a spoken transcript into a :class:`TVAction` (rule-based, no I/O).

The rules are intentionally small and deterministic so the route works without a
model. ``parse_tv_command`` returns ``None`` for anything that is not a TV command
(the caller then falls back to a normal spoken answer). ``strip_tv_suffix`` detects
the phone-routing marker ("… on the TV") so a command spoken to the phone can be
forwarded to the paired TV.
"""

from __future__ import annotations

import re

from friday.tv.models import TVAction, TVActionType

#: Exact spoken phrases that map straight to a transport key. Checked before the verb
#: rules so a bare "play"/"forward" never looks like "play <x>".
_MEDIA_KEYS: dict[str, tuple[str, str]] = {
    "pause": ("play_pause", "Paused."),
    "resume": ("play_pause", "Resuming."),
    "unpause": ("play_pause", "Resuming."),
    "continue": ("play_pause", "Resuming."),
    "stop": ("stop", "Stopped."),
    "next": ("next", "Next."),
    "skip": ("next", "Next."),
    "next video": ("next", "Next."),
    "previous": ("previous", "Previous."),
    "prev": ("previous", "Previous."),
    "rewind": ("rewind", "Rewinding."),
    "fast forward": ("fast_forward", "Fast forwarding."),
    "forward": ("fast_forward", "Fast forwarding."),
}
#: Exact spoken phrases that map to a navigation key.
_NAV_KEYS: dict[str, tuple[str, str]] = {
    "home": ("home", "Going home."),
    "go home": ("home", "Going home."),
    "home screen": ("home", "Going home."),
    "back": ("back", "Going back."),
    "go back": ("back", "Going back."),
}

#: "open|launch|start|go to <app>" — the trailing " app" and a leading "the" are noise.
_OPEN_RE = re.compile(r"^(?:open|launch|start|go to)\s+(?:the\s+)?(.+?)(?:\s+app)?$")
#: "play <query> [on <app>]" — default app is YouTube.
_PLAY_RE = re.compile(r"^play\s+(.+?)(?:\s+on\s+(.+?))?$")
#: "search [for] <query> [on <app>]".
_SEARCH_RE = re.compile(r"^search\s+(?:for\s+)?(.+?)(?:\s+on\s+(.+?))?$")
#: Trailing "on (the) tv" / "on (the) television" — the phone routing marker.
_TV_SUFFIX_RE = re.compile(r"\s+on\s+(?:the\s+)?(?:tv|television)\s*[.!?]?$", re.I)


def _norm(text: str) -> str:
    """Lowercase, collapse whitespace, strip surrounding punctuation."""
    return re.sub(r"\s{2,}", " ", text.strip().lower()).strip(" .!?,")


def _clean_app(name: str) -> str:
    """Normalise a spoken app name: drop a leading 'the' and a trailing 'app'."""
    name = re.sub(r"^the\s+", "", name.strip())
    name = re.sub(r"\s+app$", "", name)
    return name.strip()


def parse_tv_command(text: str) -> TVAction | None:
    """Parse ``text`` into a :class:`TVAction`, or ``None`` if it is not a command."""
    norm = _norm(text)
    if not norm:
        return None

    if norm in _MEDIA_KEYS:
        key, speak = _MEDIA_KEYS[norm]
        return TVAction(type=TVActionType.MEDIA, key=key, speak=speak)
    if norm in _NAV_KEYS:
        key, speak = _NAV_KEYS[norm]
        return TVAction(type=TVActionType.NAVIGATE, key=key, speak=speak)

    open_match = _OPEN_RE.match(norm)
    if open_match:
        app = _clean_app(open_match.group(1))
        return TVAction(type=TVActionType.OPEN_APP, app=app, speak=f"Opening {app}.")

    play_match = _PLAY_RE.match(norm)
    if play_match:
        query = play_match.group(1).strip()
        app = _clean_app(play_match.group(2)) if play_match.group(2) else "youtube"
        where = "YouTube" if app == "youtube" else app
        return TVAction(
            type=TVActionType.PLAY, app=app, query=query, speak=f"Playing {query} on {where}."
        )

    search_match = _SEARCH_RE.match(norm)
    if search_match:
        query = search_match.group(1).strip()
        app = _clean_app(search_match.group(2)) if search_match.group(2) else "youtube"
        where = "YouTube" if app == "youtube" else app
        return TVAction(
            type=TVActionType.SEARCH, app=app, query=query, speak=f"Searching {query} on {where}."
        )

    return None


def strip_tv_suffix(text: str) -> str | None:
    """Return ``text`` minus a trailing 'on the TV', or ``None`` if absent.

    Used by the phone path: "play X on the TV" routes to the paired TV; a sentence
    that merely mentions "on TV" mid-phrase is left alone (returns ``None``).
    """
    stripped = _TV_SUFFIX_RE.sub("", text)
    if stripped == text:
        return None
    return stripped.strip() or None
