"""Unit tests for ``memory/short_term.py`` — the per-session conversation buffer (Task 1.3).

These tests pin the :class:`ShortTermMemory` contract:

* round-trip ``append`` / ``history`` for a single session;
* isolation: messages in one session never bleed into another;
* ``clear`` empties a session;
* the per-session bound: appending past ``max_messages`` keeps only the most
  recent ``max_messages`` (oldest dropped, order preserved).
"""

from __future__ import annotations

from friday.memory.short_term import ShortTermMemory
from friday.providers.llm import Message


def _msg(content: str, role: str = "user") -> Message:
    return Message(role=role, content=content)


def test_empty_history_is_empty_list() -> None:
    mem = ShortTermMemory()
    assert mem.history("never-seen") == []


def test_append_then_history_round_trip() -> None:
    mem = ShortTermMemory()
    a = _msg("hello", role="user")
    b = _msg("hi Boss", role="assistant")
    mem.append("s1", a)
    mem.append("s1", b)
    assert mem.history("s1") == [a, b]


def test_history_returns_a_copy_not_internal_list() -> None:
    # Mutating the returned list must not corrupt stored state.
    mem = ShortTermMemory()
    mem.append("s1", _msg("one"))
    snapshot = mem.history("s1")
    snapshot.append(_msg("injected"))
    assert mem.history("s1") == [_msg("one")]


def test_sessions_are_isolated() -> None:
    mem = ShortTermMemory()
    mem.append("s1", _msg("from-one"))
    mem.append("s2", _msg("from-two"))
    assert mem.history("s1") == [_msg("from-one")]
    assert mem.history("s2") == [_msg("from-two")]


def test_clear_empties_only_target_session() -> None:
    mem = ShortTermMemory()
    mem.append("s1", _msg("a"))
    mem.append("s2", _msg("b"))
    mem.clear("s1")
    assert mem.history("s1") == []
    assert mem.history("s2") == [_msg("b")]


def test_clear_unknown_session_is_noop() -> None:
    mem = ShortTermMemory()
    mem.clear("never-seen")  # must not raise
    assert mem.history("never-seen") == []


def test_default_max_messages_is_fifty() -> None:
    mem = ShortTermMemory()
    assert mem.max_messages == 50


def test_bound_keeps_only_most_recent() -> None:
    mem = ShortTermMemory(max_messages=3)
    for i in range(5):
        mem.append("s1", _msg(f"m{i}"))
    history = mem.history("s1")
    assert len(history) == 3
    # Oldest (m0, m1) dropped; the three most recent remain, in order.
    assert [m.content for m in history] == ["m2", "m3", "m4"]


def test_bound_is_per_session() -> None:
    mem = ShortTermMemory(max_messages=2)
    for i in range(4):
        mem.append("s1", _msg(f"a{i}"))
    for i in range(4):
        mem.append("s2", _msg(f"b{i}"))
    assert [m.content for m in mem.history("s1")] == ["a2", "a3"]
    assert [m.content for m in mem.history("s2")] == ["b2", "b3"]
