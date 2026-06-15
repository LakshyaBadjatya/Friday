"""Unit tests for ``core/state.py`` — Mode, RouteDecision, GraphState (Task 1.1).

These tests pin the contract: the ``Mode`` enum members, ``RouteDecision``
construction, ``GraphState`` defaults, and — critically — that ``GraphState``
round-trips through ``model_dump_json`` / ``model_validate_json`` unchanged.
``RouteDecision`` is defined here (not in ``router.py``) to avoid a
router <-> state import cycle.
"""

from __future__ import annotations

from friday.core.state import GraphState, Mode, RouteDecision
from friday.providers.llm import Message


def test_mode_members() -> None:
    # Exactly these six modes are defined this phase (voice/security deferred).
    assert {m.name for m in Mode} == {
        "IDLE",
        "LISTENING",
        "ROUTING",
        "CONVERSATION",
        "RESEARCH",
        "CLARIFY",
    }


def test_route_decision_constructs() -> None:
    rd = RouteDecision(
        mode=Mode.RESEARCH,
        agent="research",
        rationale="matched research/compare phrasing",
        confidence=0.9,
    )
    assert rd.mode is Mode.RESEARCH
    assert rd.agent == "research"
    assert rd.rationale
    assert rd.confidence == 0.9


def test_route_decision_agent_optional() -> None:
    rd = RouteDecision(
        mode=Mode.CLARIFY,
        agent=None,
        rationale="ambiguous",
        confidence=0.1,
    )
    assert rd.agent is None


def test_graph_state_defaults() -> None:
    state = GraphState(session_id="s1", user_input="hello")
    assert state.session_id == "s1"
    assert state.mode is Mode.IDLE
    assert state.messages == []
    assert state.user_input == "hello"
    assert state.route is None
    assert state.scratchpad == {}
    assert state.response is None


def test_graph_state_round_trip_minimal() -> None:
    state = GraphState(session_id="s1", user_input="hello")
    restored = GraphState.model_validate_json(state.model_dump_json())
    assert restored == state


def test_graph_state_round_trip_full() -> None:
    state = GraphState(
        session_id="abc-123",
        mode=Mode.RESEARCH,
        messages=[
            Message(role="user", content="research vector dbs"),
            Message(role="assistant", content="on it"),
        ],
        user_input="research vector dbs",
        route=RouteDecision(
            mode=Mode.RESEARCH,
            agent="research",
            rationale="research/compare phrasing",
            confidence=0.92,
        ),
        scratchpad={"results": [1, 2, 3], "note": "x"},
        response="here is what I found",
    )
    restored = GraphState.model_validate_json(state.model_dump_json())
    assert restored == state
    assert restored.mode is Mode.RESEARCH
    assert restored.route is not None
    assert restored.route.mode is Mode.RESEARCH
    assert restored.messages[0].content == "research vector dbs"
