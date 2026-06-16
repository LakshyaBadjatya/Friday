# © Lakshya Badjatya — Author
"""PII redaction: scrub personal data from text before it leaves the machine.

When FRIDAY is pointed at a real provider (or any external sink), the text it
sends should not carry the owner's personal data unnecessarily. :class:`PIIRedactor`
replaces the common, high-confidence PII shapes — email addresses, 16-digit
payment-card numbers, IPv4 addresses, and 10-digit phone numbers — with a typed
placeholder (``[EMAIL]``, ``[CARD]``, ``[IP]``, ``[PHONE]``), and reports how
many of each it found.

The patterns are deliberately conservative and applied in a fixed order
(card → email → IP → phone) so a longer structure is redacted before a shorter
one can nibble at its digits, keeping the transform deterministic. This is a
pure, offline module: regexes only, no LLM, no configuration, no network — so it
can be unit-tested exhaustively and safely run on the hot path before egress.
"""

from __future__ import annotations

import re

from pydantic import BaseModel

# Ordered (kind, placeholder, pattern). Order matters: a 16-digit card is matched
# before the phone rule could claim a sub-run of its digits, and emails before the
# IP rule could catch a dotted run inside a domain.
_PATTERNS: tuple[tuple[str, str, re.Pattern[str]], ...] = (
    (
        "card",
        "[CARD]",
        re.compile(r"\b(?:\d[ -]?){15}\d\b"),
    ),
    (
        "email",
        "[EMAIL]",
        re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+\.[A-Za-z0-9.-]+\b"),
    ),
    (
        "ip",
        "[IP]",
        re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    ),
    (
        "phone",
        "[PHONE]",
        re.compile(r"(?:\+?\d{1,3}[ -]?)?\(?\d{3}\)?[ -]?\d{3}[ -]?\d{4}\b"),
    ),
)


class RedactionResult(BaseModel):
    """The result of redacting one piece of text.

    Attributes:
        text: The redacted text, with each PII match replaced by its typed
            placeholder.
        counts: How many matches were redacted, keyed by kind (``email`` /
            ``card`` / ``ip`` / ``phone``); kinds with zero matches are omitted.
    """

    text: str
    counts: dict[str, int] = {}


class PIIRedactor:
    """Replaces common PII shapes in text with typed placeholders.

    Stateless and deterministic; one instance is safe to share. Construction
    takes no arguments — the patterns are fixed and high-confidence by design.
    """

    def redact(self, text: str) -> RedactionResult:
        """Return ``text`` with PII replaced by placeholders, plus per-kind counts."""
        counts: dict[str, int] = {}
        redacted = text
        for kind, placeholder, pattern in _PATTERNS:
            redacted, n = pattern.subn(placeholder, redacted)
            if n:
                counts[kind] = n
        return RedactionResult(text=redacted, counts=counts)

    def scrub(self, text: str) -> str:
        """Shorthand returning only the redacted text."""
        return self.redact(text).text

    def contains_pii(self, text: str) -> bool:
        """Whether ``text`` contains any recognized PII."""
        return any(pattern.search(text) for _, _, pattern in _PATTERNS)
