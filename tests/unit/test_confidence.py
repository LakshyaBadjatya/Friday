# © Lakshya Badjatya — Author
"""Unit tests for :mod:`friday.core.confidence` (calibrated confidence scoring).

Pure and offline — no LLM, no network, no clock. The pinned behaviours mirror
the slice contract:

* the score is clamped to ``[0, 1]`` at BOTH ends, even for out-of-range inputs;
* it is monotonic in every positive signal — raising one input (route, agent,
  grounding, a web hit) never lowers the score;
* the rationale is a non-empty single line that names the mode;
* :func:`signals_from_state` reconstructs the signals from a hand-built
  :class:`~friday.core.state.GraphState` (route + the scratchpad keys the
  orchestrator stamps), tolerating absent / malformed entries.
"""

from __future__ import annotations

from friday.core.confidence import (
    ConfidenceScorer,
    ConfidenceSignals,
    signals_from_state,
)
from friday.core.state import GraphState, Mode, RouteDecision


def _signals(
    *,
    route_confidence: float = 0.7,
    agent_confidence: float | None = None,
    retrieval_grounded: bool = False,
    web_search_hit: bool | None = None,
    mode: str = "CONVERSATION",
) -> ConfidenceSignals:
    """Build :class:`ConfidenceSignals` with sensible defaults for one test knob."""
    return ConfidenceSignals(
        route_confidence=route_confidence,
        agent_confidence=agent_confidence,
        retrieval_grounded=retrieval_grounded,
        web_search_hit=web_search_hit,
        mode=mode,
    )


# --------------------------------------------------------------------------- #
# value object
# --------------------------------------------------------------------------- #
def test_signals_are_frozen() -> None:
    signals = _signals()
    # A frozen pydantic model rejects mutation.
    try:
        signals.route_confidence = 0.1  # type: ignore[misc]
    except (ValueError, TypeError):
        return
    raise AssertionError("ConfidenceSignals must be frozen")


# --------------------------------------------------------------------------- #
# clamping (both ends)
# --------------------------------------------------------------------------- #
def test_clamps_above_one() -> None:
    scorer = ConfidenceScorer()
    # Every signal pushed past its range; the result must still land in [0, 1].
    score = scorer.score(
        _signals(
            route_confidence=5.0,
            agent_confidence=9.0,
            retrieval_grounded=True,
            web_search_hit=True,
        )
    )
    assert score.value == 1.0


def test_clamps_below_zero() -> None:
    scorer = ConfidenceScorer()
    score = scorer.score(
        _signals(
            route_confidence=-3.0,
            agent_confidence=-1.0,
            retrieval_grounded=False,
            web_search_hit=False,
        )
    )
    assert score.value == 0.0


def test_value_always_in_unit_interval() -> None:
    scorer = ConfidenceScorer()
    for rc in (-2.0, 0.0, 0.33, 0.81, 1.0, 4.0):
        for ac in (None, -1.0, 0.5, 2.0):
            for grounded in (False, True):
                for web in (None, False, True):
                    score = scorer.score(
                        _signals(
                            route_confidence=rc,
                            agent_confidence=ac,
                            retrieval_grounded=grounded,
                            web_search_hit=web,
                        )
                    )
                    assert 0.0 <= score.value <= 1.0


# --------------------------------------------------------------------------- #
# determinism
# --------------------------------------------------------------------------- #
def test_deterministic() -> None:
    scorer = ConfidenceScorer()
    signals = _signals(
        route_confidence=0.6, agent_confidence=0.8, retrieval_grounded=True
    )
    first = scorer.score(signals)
    second = scorer.score(signals)
    assert first == second


# --------------------------------------------------------------------------- #
# monotonicity: raising one signal never lowers the score
# --------------------------------------------------------------------------- #
def test_monotonic_in_route_confidence() -> None:
    scorer = ConfidenceScorer()
    low = scorer.score(_signals(route_confidence=0.2)).value
    high = scorer.score(_signals(route_confidence=0.9)).value
    assert high >= low
    assert high > low  # route weight is strictly positive


def test_monotonic_in_agent_confidence() -> None:
    scorer = ConfidenceScorer()
    low = scorer.score(_signals(agent_confidence=0.1)).value
    high = scorer.score(_signals(agent_confidence=0.95)).value
    assert high >= low
    assert high > low


def test_grounding_never_lowers_score() -> None:
    scorer = ConfidenceScorer()
    # Holding every other signal fixed, turning grounding ON must not lower it.
    for rc in (0.0, 0.4, 0.75, 1.0):
        ungrounded = scorer.score(
            _signals(route_confidence=rc, retrieval_grounded=False)
        ).value
        grounded = scorer.score(
            _signals(route_confidence=rc, retrieval_grounded=True)
        ).value
        assert grounded >= ungrounded


def test_web_hit_never_lowers_score() -> None:
    scorer = ConfidenceScorer()
    for rc in (0.0, 0.4, 0.75, 1.0):
        miss = scorer.score(
            _signals(route_confidence=rc, web_search_hit=False)
        ).value
        hit = scorer.score(_signals(route_confidence=rc, web_search_hit=True)).value
        assert hit >= miss


def test_absent_web_signal_drops_out_of_average() -> None:
    # A web search that was never attempted (None) is neither a help nor a hurt:
    # its weight enters NEITHER the numerator nor the denominator, so the score
    # is identical whether or not the (absent) web signal is considered. We pin
    # this by comparing an all-else-equal pair where only web toggles None vs a
    # value: the None case must equal a hand-computed route+grounding average.
    scorer = ConfidenceScorer()
    # route 0.62 (weight .45) + grounded (weight .15) over (.45 + .15) present.
    expected = (0.45 * 0.62 + 0.15 * 1.0) / (0.45 + 0.15)
    score = scorer.score(
        _signals(route_confidence=0.62, retrieval_grounded=True, web_search_hit=None)
    )
    assert abs(score.value - expected) < 1e-9


def test_route_only_collapses_to_route_when_grounded() -> None:
    # With no agent and no web search, a fully grounded answer's only present
    # signals are the route (value rc) and grounding (value 1.0). When rc == 1.0
    # both are at the ceiling, so the weighted average is exactly 1.0; and in
    # general the score sits between rc and 1.0 — never below rc.
    scorer = ConfidenceScorer()
    score = scorer.score(_signals(route_confidence=1.0, retrieval_grounded=True))
    assert abs(score.value - 1.0) < 1e-9


# --------------------------------------------------------------------------- #
# rationale
# --------------------------------------------------------------------------- #
def test_rationale_non_empty_and_names_mode() -> None:
    scorer = ConfidenceScorer()
    score = scorer.score(_signals(mode="RESEARCH", web_search_hit=True))
    assert score.rationale.strip()
    assert "RESEARCH" in score.rationale


def test_rationale_reflects_present_signals() -> None:
    scorer = ConfidenceScorer()
    score = scorer.score(
        _signals(agent_confidence=0.8, retrieval_grounded=True, web_search_hit=True)
    )
    lowered = score.rationale.lower()
    assert "agent" in lowered
    assert "grounded" in lowered
    assert "web hit" in lowered


# --------------------------------------------------------------------------- #
# extraction from a hand-built GraphState
# --------------------------------------------------------------------------- #
def test_extraction_full_signals() -> None:
    state = GraphState(session_id="s1", user_input="q")
    state.route = RouteDecision(
        mode=Mode.RESEARCH, rationale="r", confidence=0.66
    )
    state.mode = Mode.RESEARCH
    state.scratchpad["agent_confidence"] = 0.8
    state.scratchpad["retrieval_grounded"] = True
    state.scratchpad["web_search_invoked"] = True
    state.scratchpad["web_search_results"] = [{"title": "t", "url": "u"}]

    signals = signals_from_state(state)

    assert signals.route_confidence == 0.66
    assert signals.agent_confidence == 0.8
    assert signals.retrieval_grounded is True
    assert signals.web_search_hit is True
    assert signals.mode == "RESEARCH"


def test_extraction_defaults_when_bare() -> None:
    # A bare state (no route, empty scratchpad): route confidence floors to 0.0,
    # agent confidence is absent, ungrounded, and no web search was attempted.
    state = GraphState(session_id="s2", user_input="q")
    signals = signals_from_state(state)
    assert signals.route_confidence == 0.0
    assert signals.agent_confidence is None
    assert signals.retrieval_grounded is False
    assert signals.web_search_hit is None
    assert signals.mode == str(Mode.IDLE)


def test_extraction_web_search_invoked_but_empty_is_miss() -> None:
    # A search was invoked but returned no rows -> an explicit miss (False), not
    # an absent signal (None).
    state = GraphState(session_id="s3", user_input="q")
    state.scratchpad["web_search_invoked"] = True
    state.scratchpad["web_search_results"] = []
    signals = signals_from_state(state)
    assert signals.web_search_hit is False


def test_extraction_grounded_inferred_from_citations() -> None:
    state = GraphState(session_id="s4", user_input="q")
    state.scratchpad["citations"] = ["fact:1", "fact:2"]
    signals = signals_from_state(state)
    assert signals.retrieval_grounded is True


def test_extraction_tolerates_malformed_scratchpad() -> None:
    # Garbage of the wrong type on the scratchpad must be treated as absent, not
    # crash the extractor or fabricate a confidence.
    state = GraphState(session_id="s5", user_input="q")
    state.scratchpad["agent_confidence"] = "not a number"
    state.scratchpad["web_search_invoked"] = True
    state.scratchpad["web_search_results"] = "not a list"
    signals = signals_from_state(state)
    assert signals.agent_confidence is None
    assert signals.web_search_hit is False


def test_extracted_signals_feed_scorer() -> None:
    # End-to-end: extraction -> scoring yields a clamped, in-range score.
    state = GraphState(session_id="s6", user_input="q")
    state.route = RouteDecision(
        mode=Mode.CONVERSATION, rationale="r", confidence=0.5
    )
    state.mode = Mode.CONVERSATION
    score = ConfidenceScorer().score(signals_from_state(state))
    assert 0.0 <= score.value <= 1.0
    assert score.rationale.strip()
