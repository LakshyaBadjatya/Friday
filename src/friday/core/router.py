"""Deterministic intent router: ``route(state) -> RouteDecision`` (Task 1.2).

This router is a **deterministic keyword/heuristic classifier** — it does NOT
call an LLM. The rule set is intentionally small, ordered, and documented:

1. **Empty / too-short / gibberish** input -> low confidence -> CLARIFY.
2. **Research / analysis / compare** phrasing -> RESEARCH at high confidence.
3. **General / chit-chat / simple question** signals -> CONVERSATION at high
   confidence.
4. Anything left genuinely ambiguous -> low confidence -> CLARIFY.

The final mode is gated by ``settings.route_min_confidence``: any decision whose
confidence falls below the threshold is downgraded to :class:`Mode.CLARIFY` so
FRIDAY asks rather than guesses. Rules are evaluated top-to-bottom; the first
match wins. ``RouteDecision``/``Mode``/``GraphState`` come from ``core.state`` to
keep the dependency edge one-directional (state never imports router).
"""

from __future__ import annotations

import re

from friday.config import get_settings
from friday.core.state import GraphState, Mode, RouteDecision

# --- Rule vocabulary (ordered, documented) -------------------------------- #

# Strong signals that the user wants multi-source research / analysis /
# comparison. Matched as whole words so "research" hits but "researcher" in a
# casual sentence still reads naturally.
_RESEARCH_KEYWORDS: frozenset[str] = frozenset(
    {
        "research",
        "compare",
        "comparison",
        "analyze",
        "analyse",
        "analysis",
        "investigate",
        "evaluate",
        "benchmark",
        "benchmarks",
        "sources",
        "cite",
        "citation",
        "summarize",
        "summarise",
        "pros",
        "cons",
    }
)

# Multi-word research phrases that single-word matching would miss.
_RESEARCH_PHRASES: tuple[str, ...] = (
    "look up",
    "find out",
    "dig into",
    "deep dive",
    "what is the latest",
)

# Signals of ordinary conversation / chit-chat / a simple question. Greetings,
# politeness, and question-word openers all read as CONVERSATION.
_CONVERSATION_KEYWORDS: frozenset[str] = frozenset(
    {
        "hello",
        "hi",
        "hey",
        "thanks",
        "thank",
        "please",
        "joke",
        "what",
        "what's",
        "whats",
        "who",
        "when",
        "where",
        "why",
        "how",
        "is",
        "are",
        "can",
        "could",
        "would",
        "do",
        "does",
        "tell",
        "time",
    }
)

# A token is "word-like" (and therefore plausibly real language) if it contains
# a vowel or is a short function word. Strings of consonants like "asdfghjkl"
# are treated as gibberish.
_VOWELS: frozenset[str] = frozenset("aeiou")
_SHORT_REAL_WORDS: frozenset[str] = frozenset({"my", "by", "ok", "no"})

# Confidence levels. Kept as named constants so the threshold gate is legible.
_HIGH_CONFIDENCE = 0.9
_LOW_CONFIDENCE = 0.2

_WORD_RE = re.compile(r"[a-z0-9']+")


def _tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def _looks_like_language(tokens: list[str]) -> bool:
    """True if at least one token reads like a real word (has a vowel)."""
    for tok in tokens:
        stripped = tok.strip("'")
        if not stripped:
            continue
        if any(ch in _VOWELS for ch in stripped):
            return True
        if stripped in _SHORT_REAL_WORDS:
            return True
        if stripped.isdigit():
            return True
    return False


def _classify(text: str) -> RouteDecision:
    """Apply the ordered rule set; return a pre-threshold :class:`RouteDecision`."""
    normalized = text.strip()

    # Rule 1: empty / whitespace-only input -> clarify.
    if not normalized:
        return RouteDecision(
            mode=Mode.CLARIFY,
            agent=None,
            rationale="empty input",
            confidence=0.0,
        )

    lowered = normalized.lower()
    tokens = _tokenize(normalized)

    # Rule 1b: no word-like tokens (pure gibberish) -> clarify.
    if not _looks_like_language(tokens):
        return RouteDecision(
            mode=Mode.CLARIFY,
            agent=None,
            rationale="unrecognized / gibberish input",
            confidence=_LOW_CONFIDENCE,
        )

    token_set = set(tokens)

    # Rule 2: research / analysis / compare phrasing -> RESEARCH (high).
    matched_research = sorted(token_set & _RESEARCH_KEYWORDS)
    matched_phrase = next((p for p in _RESEARCH_PHRASES if p in lowered), None)
    if matched_research or matched_phrase:
        signal = matched_phrase if matched_phrase else ", ".join(matched_research)
        return RouteDecision(
            mode=Mode.RESEARCH,
            agent="research",
            rationale=f"research/analysis phrasing: {signal}",
            confidence=_HIGH_CONFIDENCE,
        )

    # Rule 3: general / chit-chat / simple question -> CONVERSATION (high).
    matched_convo = sorted(token_set & _CONVERSATION_KEYWORDS)
    if matched_convo:
        return RouteDecision(
            mode=Mode.CONVERSATION,
            agent=None,
            rationale=f"conversational phrasing: {', '.join(matched_convo)}",
            confidence=_HIGH_CONFIDENCE,
        )

    # Rule 4: word-like but no recognized intent -> ambiguous -> clarify.
    return RouteDecision(
        mode=Mode.CLARIFY,
        agent=None,
        rationale="no clear research or conversational intent",
        confidence=_LOW_CONFIDENCE,
    )


async def route(state: GraphState) -> RouteDecision:
    """Classify ``state.user_input`` into a :class:`RouteDecision`.

    Deterministic: identical input always yields the same decision. The
    classification is gated by ``settings.route_min_confidence`` — any decision
    below the threshold is downgraded to :class:`Mode.CLARIFY` so FRIDAY asks a
    clarifying question instead of guessing.
    """
    decision = _classify(state.user_input)

    threshold = get_settings().route_min_confidence
    if decision.mode is not Mode.CLARIFY and decision.confidence < threshold:
        return RouteDecision(
            mode=Mode.CLARIFY,
            agent=None,
            rationale=(
                f"confidence {decision.confidence:.2f} below threshold "
                f"{threshold:.2f}; clarifying instead of guessing"
            ),
            confidence=decision.confidence,
        )
    return decision
