# © Lakshya Badjatya — Author
"""A tiny, offline, deterministic sentiment analyzer — no model, no network.

A lexicon scorer with negation handling: it tokenizes text, matches a curated
positive/negative word list, flips polarity inside a short window after a negator
("not great" → negative), and returns a normalized score in ``[-1, 1]`` with a
``positive``/``negative``/``neutral`` label. It is pure and deterministic — the
same text always scores the same — so FRIDAY can read the owner's mood (e.g. to
soften a reply, or flag a frustrated turn) with zero dependencies and no latency.

This is intentionally simple: a transparent baseline that needs no ML backend.
Swapping in a learned classifier later is a drop-in behind the same
:meth:`SentimentAnalyzer.analyze` contract.
"""

from __future__ import annotations

import re

from pydantic import BaseModel

#: Words read as positive sentiment (lemma-ish; matched on lowercased tokens).
_POSITIVE: frozenset[str] = frozenset({
    "good", "great", "excellent", "awesome", "love", "loved", "loving", "happy",
    "glad", "wonderful", "fantastic", "nice", "perfect", "thanks", "thank",
    "appreciate", "appreciated", "brilliant", "amazing", "superb", "pleased",
    "delighted", "win", "wins", "success", "successful", "works", "working",
    "resolved", "fixed", "smooth", "fast", "best", "better", "enjoy", "enjoyed",
    "yay", "excited", "grateful", "solid", "clean", "clear", "helpful",
})

#: Words read as negative sentiment.
_NEGATIVE: frozenset[str] = frozenset({
    "bad", "terrible", "awful", "hate", "hated", "sad", "angry", "frustrated",
    "frustrating", "annoyed", "annoying", "horrible", "worst", "worse", "broken",
    "break", "fail", "failed", "failing", "failure", "error", "errors", "bug",
    "bugs", "crash", "crashed", "slow", "stuck", "wrong", "problem", "problems",
    "issue", "issues", "disappointed", "disappointing", "useless", "ugly",
    "confusing", "confused", "pain", "painful", "hard", "difficult", "messy",
})

#: Tokens that flip the polarity of sentiment words appearing shortly after.
_NEGATORS: frozenset[str] = frozenset({
    "not", "no", "never", "cannot", "without", "hardly", "barely", "neither",
})

#: How many tokens after a negator stay flipped.
_NEG_WINDOW = 3


class SentimentResult(BaseModel):
    """A sentiment verdict for one text.

    ``label`` is ``positive`` / ``negative`` / ``neutral``; ``score`` is the
    normalized polarity in ``[-1, 1]`` (``(pos - neg) / (pos + neg)``, ``0`` when
    no sentiment word matched). ``positive_hits`` / ``negative_hits`` list the
    matched words *after* negation was applied (so "not good" lands in
    ``negative_hits``).
    """

    label: str
    score: float
    positive_hits: list[str]
    negative_hits: list[str]


class SentimentAnalyzer:
    """A lexicon sentiment scorer with simple negation handling."""

    def __init__(
        self,
        positive: frozenset[str] = _POSITIVE,
        negative: frozenset[str] = _NEGATIVE,
        negators: frozenset[str] = _NEGATORS,
    ) -> None:
        self._pos = positive
        self._neg = negative
        self._negators = negators

    def analyze(self, text: str) -> SentimentResult:
        """Score ``text`` and return its :class:`SentimentResult`."""
        # Normalize contractions so "isn't" / "can't" read as "... not ...".
        normalized = text.lower().replace("n't", " not")
        tokens = re.findall(r"[a-z']+", normalized)

        positive_hits: list[str] = []
        negative_hits: list[str] = []
        negate_countdown = 0
        for token in tokens:
            negate = negate_countdown > 0
            if negate_countdown > 0:
                negate_countdown -= 1
            if token in self._negators:
                negate_countdown = _NEG_WINDOW
                continue
            if token in self._pos:
                (negative_hits if negate else positive_hits).append(token)
            elif token in self._neg:
                (positive_hits if negate else negative_hits).append(token)

        pos, neg = len(positive_hits), len(negative_hits)
        total = pos + neg
        score = round((pos - neg) / total, 4) if total else 0.0
        if score > 0.1:
            label = "positive"
        elif score < -0.1:
            label = "negative"
        else:
            label = "neutral"
        return SentimentResult(
            label=label,
            score=score,
            positive_hits=positive_hits,
            negative_hits=negative_hits,
        )
