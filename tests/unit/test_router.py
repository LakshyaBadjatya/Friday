"""Unit tests for ``core/router.py`` — deterministic intent router (Task 1.2).

The router is a deterministic keyword/heuristic classifier (NOT an LLM call).
The core test is table-driven: a list of ``(utterance, expected_mode)`` rows
asserted in a loop. Genuinely ambiguous, empty, or gibberish input MUST route
to ``CLARIFY`` (never a guess).
"""

from __future__ import annotations

import pytest

from friday.core.router import route
from friday.core.state import GraphState, Mode, RouteDecision

# (utterance, expected_mode) — extend this table as the rule set grows.
ROUTER_TABLE: list[tuple[str, Mode]] = [
    # --- Conversation: general / chit-chat / simple Q&A ---
    ("what's 2+2", Mode.CONVERSATION),
    ("hello there", Mode.CONVERSATION),
    ("how are you doing today", Mode.CONVERSATION),
    ("tell me a joke", Mode.CONVERSATION),
    ("what time is it", Mode.CONVERSATION),
    ("thanks, that's helpful", Mode.CONVERSATION),
    # --- Research: research / analysis / compare phrasing ---
    ("research the best vector database and compare options", Mode.RESEARCH),
    ("compare postgres and mysql for our workload", Mode.RESEARCH),
    ("analyze the pros and cons of serverless", Mode.RESEARCH),
    ("find sources on the history of the transistor", Mode.RESEARCH),
    ("investigate why our latency regressed", Mode.RESEARCH),
    ("look up the latest benchmarks for llama models", Mode.RESEARCH),
    # --- Clarify: ambiguous / empty / gibberish ---
    ("", Mode.CLARIFY),
    ("   ", Mode.CLARIFY),
    ("asdfghjkl", Mode.CLARIFY),
    ("xyz qwerty", Mode.CLARIFY),
]


def _state(text: str) -> GraphState:
    return GraphState(session_id="s1", user_input=text)


@pytest.mark.parametrize(("utterance", "expected_mode"), ROUTER_TABLE)
async def test_router_table(utterance: str, expected_mode: Mode) -> None:
    decision = await route(_state(utterance))
    assert decision.mode is expected_mode, (
        f"{utterance!r} routed to {decision.mode} (expected {expected_mode}); "
        f"rationale={decision.rationale!r} confidence={decision.confidence}"
    )


async def test_router_returns_route_decision() -> None:
    decision = await route(_state("what's 2+2"))
    assert isinstance(decision, RouteDecision)
    assert decision.rationale  # always explains itself
    assert 0.0 <= decision.confidence <= 1.0


async def test_research_high_confidence() -> None:
    decision = await route(_state("research the best vector database and compare options"))
    assert decision.mode is Mode.RESEARCH
    assert decision.agent == "research"
    assert decision.confidence >= 0.55


async def test_conversation_high_confidence() -> None:
    decision = await route(_state("what's 2+2"))
    assert decision.mode is Mode.CONVERSATION
    assert decision.confidence >= 0.55


async def test_ambiguous_routes_to_clarify_low_confidence() -> None:
    # Gibberish must clarify, never guess.
    decision = await route(_state("asdfghjkl"))
    assert decision.mode is Mode.CLARIFY
    assert decision.confidence < 0.55


async def test_empty_routes_to_clarify() -> None:
    decision = await route(_state(""))
    assert decision.mode is Mode.CLARIFY


async def test_below_threshold_forces_clarify(monkeypatch: pytest.MonkeyPatch) -> None:
    """When confidence falls below ``route_min_confidence`` the mode is CLARIFY.

    Raising the threshold above a normally-confident utterance's score must
    flip its decision to CLARIFY, proving the threshold gate is wired to
    settings rather than hard-coded.
    """
    import friday.core.router as router_mod
    from friday.config import Settings

    def _high_threshold() -> Settings:
        return Settings(_env_file=None, route_min_confidence=0.99)

    monkeypatch.setattr(router_mod, "get_settings", _high_threshold)
    decision = await route(_state("hello there"))
    assert decision.mode is Mode.CLARIFY
