"""Unit tests for the pure SM-2 spaced-repetition core (Tier 2 study module).

:func:`friday.study.srs.sm2` is a pure function: given a card's current
:class:`~friday.study.srs.ReviewState` and a recall ``grade`` (0..5) it returns
the *next* state — never mutating the input, never reading a clock (the interval
is a day count; the caller turns it into a ``due_at`` against an injected now).

Pinned behaviours (the standard SM-2 algorithm):

* A passing grade (``>= 3``) advances the interval ``1 -> 6 -> round(interval *
  ease)`` and increments ``reps``; the ease is nudged by the standard SM-2
  formula.
* A lapse (``grade < 3``) resets ``reps`` and ``interval_days`` to 1 and lowers
  the ease.
* The ease never drops below the 1.3 floor, no matter how many lapses.
"""

from __future__ import annotations

import pytest

from friday.study.srs import ReviewState, sm2

# The standard SM-2 starting ease for a fresh card.
_START = ReviewState(ease=2.5, interval_days=0, reps=0)


def _ease_after(grade: int, ease: float) -> float:
    """The standard SM-2 ease update (mirrors the implementation under test)."""
    delta = 0.1 - (5 - grade) * (0.08 + (5 - grade) * 0.02)
    return max(1.3, ease + delta)


# --------------------------------------------------------------------------- #
# purity
# --------------------------------------------------------------------------- #
def test_sm2_is_pure_does_not_mutate_input() -> None:
    state = ReviewState(ease=2.5, interval_days=6, reps=2)
    before = state.model_copy(deep=True)
    sm2(state, 4)
    assert state == before


# --------------------------------------------------------------------------- #
# passing progression: 1 -> 6 -> round(interval * ease)
# --------------------------------------------------------------------------- #
def test_first_pass_sets_interval_to_one_day() -> None:
    out = sm2(_START, 4)
    assert out.reps == 1
    assert out.interval_days == 1


def test_second_pass_sets_interval_to_six_days() -> None:
    after_first = sm2(_START, 4)
    out = sm2(after_first, 4)
    assert out.reps == 2
    assert out.interval_days == 6


def test_third_pass_multiplies_interval_by_ease_and_rounds() -> None:
    after_first = sm2(_START, 4)
    after_second = sm2(after_first, 4)
    out = sm2(after_second, 4)
    assert out.reps == 3
    # interval was 6; new interval = round(6 * ease).
    assert out.interval_days == round(6 * after_second.ease)


def test_progression_reaches_about_fifteen_on_the_third_review() -> None:
    """A grade-4 progression yields 1 -> 6 -> ~15 (round(6 * ease))."""
    intervals: list[int] = []
    state = _START
    for _ in range(3):
        state = sm2(state, 4)
        intervals.append(state.interval_days)
    assert intervals[0] == 1
    assert intervals[1] == 6
    assert 14 <= intervals[2] <= 16  # round(6 * ~2.5) ≈ 15


def test_pass_grade_adjusts_ease_per_standard_formula() -> None:
    out = sm2(_START, 4)
    assert out.ease == pytest.approx(_ease_after(4, 2.5))


def test_perfect_grade_raises_ease() -> None:
    out = sm2(_START, 5)
    assert out.ease > 2.5
    assert out.ease == pytest.approx(_ease_after(5, 2.5))


def test_grade_three_is_a_pass_that_lowers_ease_but_advances() -> None:
    # Grade 3 is the lowest passing grade: it advances reps/interval but the
    # standard formula still lowers the ease.
    out = sm2(_START, 3)
    assert out.reps == 1
    assert out.interval_days == 1
    assert out.ease < 2.5
    assert out.ease == pytest.approx(_ease_after(3, 2.5))


# --------------------------------------------------------------------------- #
# lapse: reset reps + interval, lower ease
# --------------------------------------------------------------------------- #
def test_lapse_resets_reps_and_interval() -> None:
    mature = ReviewState(ease=2.6, interval_days=30, reps=5)
    out = sm2(mature, 1)
    assert out.reps == 1
    assert out.interval_days == 1
    assert out.ease < 2.6


def test_lapse_lowers_ease_per_standard_formula() -> None:
    mature = ReviewState(ease=2.5, interval_days=30, reps=5)
    out = sm2(mature, 0)
    assert out.ease == pytest.approx(_ease_after(0, 2.5))


# --------------------------------------------------------------------------- #
# ease floor at 1.3
# --------------------------------------------------------------------------- #
def test_ease_never_drops_below_floor() -> None:
    state = ReviewState(ease=1.3, interval_days=1, reps=1)
    # Repeated lapses must never push the ease below 1.3.
    for _ in range(20):
        state = sm2(state, 0)
        assert state.ease >= 1.3
    assert state.ease == pytest.approx(1.3)


def test_ease_floor_holds_from_a_low_start() -> None:
    state = ReviewState(ease=1.35, interval_days=10, reps=3)
    out = sm2(state, 0)
    # The raw update would dip under 1.3; the floor clamps it.
    assert out.ease == pytest.approx(1.3)


# --------------------------------------------------------------------------- #
# grade validation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad", [-1, 6, 10])
def test_out_of_range_grade_raises(bad: int) -> None:
    with pytest.raises(ValueError):
        sm2(_START, bad)
