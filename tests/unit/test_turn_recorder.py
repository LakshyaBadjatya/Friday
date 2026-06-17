# © Lakshya Badjatya — Author
"""Unit tests for the turn-replay recorder."""

from __future__ import annotations

from friday.observability.replay import TurnRecorder


def test_record_assigns_monotonic_ids_and_returns_record() -> None:
    rec = TurnRecorder()
    first = rec.record(session_id="s1", user_input="hi", response="hello", mode="CONVERSATION")
    second = rec.record(session_id="s1", user_input="bye", response="ok", mode="CONVERSATION")
    assert first.id == 1
    assert second.id == 2
    assert first.user_input == "hi"
    assert first.response == "hello"


def test_recent_is_oldest_first_and_limited() -> None:
    rec = TurnRecorder()
    for i in range(5):
        rec.record(session_id="s", user_input=f"q{i}", response=f"a{i}", mode=None)
    recent = rec.recent(3)
    assert [r.user_input for r in recent] == ["q2", "q3", "q4"]


def test_get_returns_record_or_none() -> None:
    rec = TurnRecorder()
    rec.record(session_id="s", user_input="q", response="a", mode="CONVERSATION")
    assert rec.get(1) is not None
    assert rec.get(999) is None


def test_ring_buffer_evicts_oldest_but_ids_keep_climbing() -> None:
    rec = TurnRecorder(capacity=2)
    rec.record(session_id="s", user_input="q1", response="a1", mode=None)
    rec.record(session_id="s", user_input="q2", response="a2", mode=None)
    third = rec.record(session_id="s", user_input="q3", response="a3", mode=None)
    assert third.id == 3
    assert rec.get(1) is None  # evicted
    assert rec.get(2) is not None
    assert [r.id for r in rec.recent()] == [2, 3]


def test_mode_and_response_may_be_none() -> None:
    rec = TurnRecorder()
    record = rec.record(session_id="s", user_input="q", response=None, mode=None)
    assert record.mode is None
    assert record.response is None
