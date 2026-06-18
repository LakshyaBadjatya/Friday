"""Turn an orchestrator reply into a short, plain string Siri can speak.

The core loop returns chat-style text — markdown, bullet lists, links, code
spans, and any length. Siri's "Speak Text" action wants a clean, bounded string.
:func:`for_speech` strips the markdown to its spoken words, collapses whitespace,
and truncates an over-long reply on a sentence boundary (falling back to a word
boundary) with a trailing ellipsis. It is pure and side-effect free so the route
stays thin and this shaping is unit-tested in isolation.
"""

from __future__ import annotations

import re

#: ``[label](url)`` -> ``label`` (keep what a listener would hear, drop the URL).
_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
#: A leading ATX heading marker (``#`` .. ``######``) on a line.
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s*")
#: A leading list marker: ``-``/``*``/``+`` bullets or ``1.``/``1)`` numbering.
_BULLET = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
#: Inline markdown punctuation that should not be spoken (emphasis, code, quote).
_MARKUP = re.compile(r"[*_~`>#]")
#: Any run of whitespace (incl. newlines/tabs), collapsed to a single space.
_WHITESPACE = re.compile(r"\s+")
#: Sentence-ending punctuation used to pick a clean truncation point.
_SENTENCE_END = re.compile(r"[.!?]")


def for_speech(text: str, max_chars: int = 600) -> str:
    """Return ``text`` reduced to a clean, bounded line suitable for TTS.

    Markdown emphasis/headings/bullets/links/code spans are removed, whitespace
    is collapsed, and the result is truncated to at most ``max_chars`` characters
    (plus a one-character ellipsis) at the last sentence boundary within the
    window, or the last word boundary if no sentence ends there. Empty or
    whitespace-only input yields ``""`` so callers can substitute a fallback.
    """
    if not text:
        return ""

    # Strip per-line markers first (headings, bullets) so list/heading text reads
    # as plain sentences once the newlines become spaces.
    stripped_lines = []
    for line in text.splitlines():
        line = _HEADING.sub("", line)
        line = _BULLET.sub("", line)
        stripped_lines.append(line)

    cleaned = " ".join(stripped_lines)
    cleaned = _LINK.sub(r"\1", cleaned)
    cleaned = _MARKUP.sub("", cleaned)
    cleaned = _WHITESPACE.sub(" ", cleaned).strip()

    if len(cleaned) <= max_chars:
        return cleaned

    window = cleaned[:max_chars]
    cut = None
    for match in _SENTENCE_END.finditer(window):
        cut = match.end()
    if cut is None:
        space = window.rfind(" ")
        cut = space if space > 0 else max_chars

    return cleaned[:cut].rstrip() + "…"
