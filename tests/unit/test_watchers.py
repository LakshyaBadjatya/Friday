# © Lakshya Badjatya — Author
"""Unit tests for the proactive watchers (calendar conflicts + price breach)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from friday.proactive.watchers import TimeEvent, find_conflicts, price_breach


def _ev(title: str, start: float, end: float) -> TimeEvent:
    return TimeEvent(title=title, start=start, end=end)


def test_detects_overlapping_events() -> None:
    events = [_ev("a", 0, 10), _ev("b", 5, 15), _ev("c", 20, 30)]
    conflicts = find_conflicts(events)
    assert [(c.a, c.b) for c in conflicts] == [("a", "b")]


def test_touching_endpoints_do_not_conflict() -> None:
    # a ends exactly when b starts -> no overlap (half-open intervals).
    assert find_conflicts([_ev("a", 0, 10), _ev("b", 10, 20)]) == []


def test_multiple_overlaps() -> None:
    events = [_ev("a", 0, 30), _ev("b", 5, 10), _ev("c", 20, 25)]
    pairs = {(c.a, c.b) for c in find_conflicts(events)}
    assert pairs == {("a", "b"), ("a", "c")}


def test_unsorted_input_is_handled() -> None:
    # Conflicts found regardless of input order (engine sorts by start).
    assert len(find_conflicts([_ev("late", 5, 15), _ev("early", 0, 10)])) == 1


def test_bad_interval_rejected() -> None:
    with pytest.raises(ValidationError):
        TimeEvent(title="x", start=10, end=5)


def test_price_breach() -> None:
    assert price_breach(105, above=100) is True
    assert price_breach(95, above=100) is False
    assert price_breach(40, below=50) is True
    assert price_breach(50, below=50) is False  # boundary is calm
    assert price_breach(100, above=200, below=50) is False
    assert price_breach(100) is False  # no bounds -> never breaches
