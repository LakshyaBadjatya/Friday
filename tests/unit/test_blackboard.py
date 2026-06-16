# © Lakshya Badjatya — Author
"""Unit tests for the debate blackboard (pure, in-memory draft store)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from friday.core.blackboard import Blackboard, Draft


def test_post_preserves_insertion_order() -> None:
    board = Blackboard()
    board.post("VISION", "first")
    board.post("GECKO", "second")
    board.post("VISION", "third")
    assert [d.content for d in board.drafts()] == ["first", "second", "third"]
    assert len(board) == 3


def test_drafts_returns_a_copy() -> None:
    board = Blackboard()
    board.post("VISION", "a")
    snapshot = board.drafts()
    snapshot.clear()
    # Mutating the returned list must not touch the board's own state.
    assert len(board) == 1


def test_by_operator_is_case_insensitive() -> None:
    board = Blackboard()
    board.post("VISION", "v1")
    board.post("gecko", "g1")
    board.post("Vision", "v2")
    got = [d.content for d in board.by_operator("vision")]
    assert got == ["v1", "v2"]
    assert board.by_operator("unknown") == []


def test_by_round_and_latest_round() -> None:
    board = Blackboard()
    assert board.latest_round() == -1  # empty board
    board.post("VISION", "r0")
    board.post("GECKO", "r1", round=1)
    board.post("VISION", "r1b", round=1)
    assert [d.content for d in board.by_round(0)] == ["r0"]
    assert [d.content for d in board.by_round(1)] == ["r1", "r1b"]
    assert board.latest_round() == 1


def test_draft_is_frozen() -> None:
    draft = Draft(operator="VISION", content="x")
    with pytest.raises(ValidationError):
        draft.content = "mutated"  # type: ignore[misc]
