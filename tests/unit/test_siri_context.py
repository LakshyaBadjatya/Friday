"""Unit tests for :mod:`friday.siri.context` — the spoken conversation's window.

The module's whole job is that a follow-up turn can see the turns before it, so
these tests pin the three behaviours the voice path depends on: what comes back
(chronological, bounded), what goes in (both halves of a turn), and what happens
when the memory is missing or broken (degrade, never raise — a lost context
window must not cost the user their answer).
"""

from __future__ import annotations

from friday.memory.short_term import ShortTermMemory
from friday.providers.llm import Message
from friday.siri.context import (
    NO_HISTORY_REPLY,
    is_recall_question,
    recall,
    remember,
)


def test_remember_then_recall_round_trips_the_turn() -> None:
    memory = ShortTermMemory()
    remember(memory, "siri", "what's the capital of france", "Paris, Boss.")

    window = recall(memory, "siri")

    assert [(m.role, m.content) for m in window] == [
        ("user", "what's the capital of france"),
        ("assistant", "Paris, Boss."),
    ]


def test_recall_is_chronological_across_turns() -> None:
    memory = ShortTermMemory()
    remember(memory, "siri", "tell me about mars", "It's the fourth planet.")
    remember(memory, "siri", "how far is it", "About 225 million kilometres.")

    contents = [m.content for m in recall(memory, "siri")]

    assert contents == [
        "tell me about mars",
        "It's the fourth planet.",
        "how far is it",
        "About 225 million kilometres.",
    ]


def test_recall_keeps_only_the_most_recent_messages() -> None:
    memory = ShortTermMemory()
    for i in range(10):
        remember(memory, "siri", f"question {i}", f"answer {i}")

    window = recall(memory, "siri", max_messages=4)

    assert [m.content for m in window] == [
        "question 8",
        "answer 8",
        "question 9",
        "answer 9",
    ]


def test_recall_drops_oldest_messages_to_fit_the_character_budget() -> None:
    """One pasted wall of text must not crowd the rest of the window out."""
    memory = ShortTermMemory()
    memory.append("siri", Message(role="user", content="x" * 500))
    memory.append("siri", Message(role="assistant", content="short reply"))

    window = recall(memory, "siri", max_chars=200)

    assert [m.content for m in window] == ["short reply"]


def test_recall_truncates_a_single_over_long_message() -> None:
    memory = ShortTermMemory()
    memory.append("siri", Message(role="user", content="y" * 5000))

    (only,) = recall(memory, "siri", max_chars=10_000)

    assert len(only.content) < 5000
    assert only.content.endswith("…")


def test_sessions_are_isolated() -> None:
    memory = ShortTermMemory()
    remember(memory, "siri", "my question", "my answer")

    assert recall(memory, "someone-else") == []


def test_recall_without_memory_returns_empty() -> None:
    assert recall(None, "siri") == []
    assert recall(object(), "siri") == []


def test_recall_survives_a_broken_memory() -> None:
    """Context is an enhancement; a failure to read it must not fail the turn."""

    class _Exploding:
        def history(self, session_id: str) -> list[Message]:
            raise RuntimeError("storage is down")

    assert recall(_Exploding(), "siri") == []


def test_remember_survives_a_broken_memory() -> None:
    class _Exploding:
        def append(self, session_id: str, msg: Message) -> None:
            raise RuntimeError("storage is down")

    remember(_Exploding(), "siri", "q", "a")  # must not raise


def test_remember_ignores_a_wholly_empty_turn() -> None:
    memory = ShortTermMemory()
    remember(memory, "siri", "", "")

    assert recall(memory, "siri") == []


def test_remember_without_memory_is_a_no_op() -> None:
    remember(None, "siri", "q", "a")  # must not raise


def test_is_recall_question_matches_natural_phrasings() -> None:
    for query in (
        "what was the last topic",
        "What were we talking about?",
        "remind me what I asked",
        "give me a recap",
        "what did you just say",
    ):
        assert is_recall_question(query), query


def test_is_recall_question_ignores_ordinary_queries() -> None:
    for query in ("what's the weather", "who made you", "12 * 37"):
        assert not is_recall_question(query), query


def test_no_history_reply_is_honest_and_non_empty() -> None:
    """An empty window is admitted, not papered over with an invented topic."""
    assert NO_HISTORY_REPLY.strip()
    assert "don't have" in NO_HISTORY_REPLY
