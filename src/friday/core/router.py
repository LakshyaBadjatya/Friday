"""Deterministic intent router: ``route(state) -> RouteDecision`` (Task 1.2).

This router is a **deterministic keyword/heuristic classifier** — it does NOT
call an LLM. The rule set is intentionally small, ordered, and documented:

1. **Empty / too-short / gibberish** input -> low confidence -> CLARIFY.
2. **Security lockdown** phrasing (``lockdown`` / "barn door" / ``revoke
   tokens`` / ``kill sessions``) -> SECURITY_LOCKDOWN at high confidence. Checked
   first because it is the defensive emergency path — it must win over every
   other intent.
3. **Device control** phrasing ("turn on/off" / ``lights`` / ``thermostat`` /
   ``device`` / ``lock``) -> DEVICE_CONTROL at high confidence.
4. **Alerting** phrasing (``alert`` / ``notify`` / ``escalate`` / ``warn``) ->
   ALERTING at high confidence.
5. **Automation** phrasing (``task`` / ``schedule`` / ``remind`` / ``automate``)
   -> AUTOMATION at high confidence.
6. **Research / analysis / compare** phrasing -> RESEARCH at high confidence.
7. **General / chit-chat / simple question** signals -> CONVERSATION at high
   confidence.
8. Anything left genuinely ambiguous -> low confidence -> CLARIFY.

The specialist-agent rules (2-5) are evaluated *before* the generic
research/conversation rules so an imperative like "warn me when cpu spikes"
(which contains the conversational opener "when") still routes to ALERTING.

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

# --- Specialist-agent vocabulary (evaluated before research/conversation) --- #

# Security lockdown ("barn door") signals: the defensive emergency path. Matched
# as whole words plus a few multi-word phrases. This rule is highest priority.
_SECURITY_KEYWORDS: frozenset[str] = frozenset(
    {
        "lockdown",
    }
)
_SECURITY_PHRASES: tuple[str, ...] = (
    "barn door",
    "revoke tokens",
    "revoke token",
    "kill sessions",
    "kill session",
    "kill all sessions",
    "security lockdown",
    "lock down",
)

# Device-control signals: "turn on/off" plus the actuatable nouns/verbs.
_DEVICE_KEYWORDS: frozenset[str] = frozenset(
    {
        "lights",
        "light",
        "thermostat",
        "device",
        "lock",
        "unlock",
        "toggle",
        "dimmer",
        "plug",
        "outlet",
        "switch",
    }
)
_DEVICE_PHRASES: tuple[str, ...] = (
    "turn on",
    "turn off",
    "switch on",
    "switch off",
)

# Alerting signals: "alert / notify / escalate / warn".
_ALERTING_KEYWORDS: frozenset[str] = frozenset(
    {
        "alert",
        "alerts",
        "notify",
        "notification",
        "escalate",
        "warn",
        "page",
    }
)

# Automation signals: "task / schedule / remind / automate".
_AUTOMATION_KEYWORDS: frozenset[str] = frozenset(
    {
        "task",
        "tasks",
        "schedule",
        "scheduled",
        "remind",
        "reminder",
        "automate",
        "automation",
        "recurring",
        "cron",
    }
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

    # Rule 2: security lockdown ("barn door") -> SECURITY_LOCKDOWN (high).
    # Checked first: the defensive emergency path must win over every other
    # intent so a "revoke tokens" / "kill sessions" / "lockdown" can never be
    # mis-routed to research or conversation.
    matched_security = sorted(token_set & _SECURITY_KEYWORDS)
    matched_security_phrase = next(
        (p for p in _SECURITY_PHRASES if p in lowered), None
    )
    if matched_security or matched_security_phrase:
        signal = (
            matched_security_phrase
            if matched_security_phrase
            else ", ".join(matched_security)
        )
        return RouteDecision(
            mode=Mode.SECURITY_LOCKDOWN,
            agent=None,
            rationale=f"security lockdown phrasing: {signal}",
            confidence=_HIGH_CONFIDENCE,
        )

    # Rule 3: device control -> DEVICE_CONTROL (high).
    matched_device = sorted(token_set & _DEVICE_KEYWORDS)
    matched_device_phrase = next((p for p in _DEVICE_PHRASES if p in lowered), None)
    if matched_device or matched_device_phrase:
        signal = (
            matched_device_phrase
            if matched_device_phrase
            else ", ".join(matched_device)
        )
        return RouteDecision(
            mode=Mode.DEVICE_CONTROL,
            agent="device",
            rationale=f"device-control phrasing: {signal}",
            confidence=_HIGH_CONFIDENCE,
        )

    # Rule 4: alerting -> ALERTING (high).
    matched_alerting = sorted(token_set & _ALERTING_KEYWORDS)
    if matched_alerting:
        return RouteDecision(
            mode=Mode.ALERTING,
            agent="alerting",
            rationale=f"alerting phrasing: {', '.join(matched_alerting)}",
            confidence=_HIGH_CONFIDENCE,
        )

    # Rule 5: automation -> AUTOMATION (high).
    matched_automation = sorted(token_set & _AUTOMATION_KEYWORDS)
    if matched_automation:
        return RouteDecision(
            mode=Mode.AUTOMATION,
            agent="automation",
            rationale=f"automation phrasing: {', '.join(matched_automation)}",
            confidence=_HIGH_CONFIDENCE,
        )

    # Rule 6: research / analysis / compare phrasing -> RESEARCH (high).
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

    # Rule 7: general / chit-chat / simple question -> CONVERSATION (high).
    matched_convo = sorted(token_set & _CONVERSATION_KEYWORDS)
    if matched_convo:
        return RouteDecision(
            mode=Mode.CONVERSATION,
            agent=None,
            rationale=f"conversational phrasing: {', '.join(matched_convo)}",
            confidence=_HIGH_CONFIDENCE,
        )

    # Rule 8: word-like but no recognized intent -> ambiguous -> clarify.
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
