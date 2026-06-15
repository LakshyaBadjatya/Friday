"""Pure SM-2 spaced-repetition scheduling (Tier 2 study module).

A dependency-free, side-effect-free implementation of the classic SuperMemo-2
algorithm: given a flashcard's current :class:`ReviewState` and the recall
``grade`` (0..5) the learner just gave, :func:`sm2` returns the *next*
:class:`ReviewState` — the new ease factor, the next inter-review interval in
**days**, and the running repetition count.

Design rules (binding):

* **Pure.** :func:`sm2` never mutates its input and never reads a clock. The
  interval it returns is a day *count*; turning it into a concrete ``due_at`` is
  the store's job (against an injected ``now``), so the scheduling math stays
  deterministic and unit-testable without time.
* **Standard SM-2.** A passing grade (``>= 3``) advances the interval
  ``1 -> 6 -> round(interval * ease)`` and increments ``reps``; a lapse
  (``grade < 3``) resets ``reps``/``interval_days`` to 1. The ease is always
  nudged by the canonical SM-2 quality formula and clamped to the 1.3 floor.
"""

from __future__ import annotations

from pydantic import BaseModel

# The SM-2 ease factor floor: the algorithm never lets a card's ease drop below
# this, so even a chronically-failed card keeps a workable (short) interval.
_EASE_FLOOR = 1.3

# The lowest recall grade still counted as a successful recall. Below this the
# card has lapsed and its schedule resets.
_PASS_THRESHOLD = 3

# Inclusive valid range for a recall grade.
_GRADE_MIN = 0
_GRADE_MAX = 5


class ReviewState(BaseModel):
    """A flashcard's spaced-repetition scheduling state.

    ``ease`` is the SM-2 ease factor (>= 1.3); ``interval_days`` is the number of
    days until the next review; ``reps`` is the count of consecutive successful
    recalls (reset to 1 on a lapse). A brand-new card starts at
    ``ReviewState(ease=2.5, interval_days=0, reps=0)``.
    """

    ease: float = 2.5
    interval_days: int = 0
    reps: int = 0


def _next_ease(ease: float, grade: int) -> float:
    """Apply the canonical SM-2 ease update, clamped to the 1.3 floor.

    ``ease' = ease + (0.1 - (5 - grade) * (0.08 + (5 - grade) * 0.02))`` — a
    perfect grade (5) nudges the ease up, lower grades pull it down, and the
    result never falls below :data:`_EASE_FLOOR`.
    """
    delta = 0.1 - (_GRADE_MAX - grade) * (0.08 + (_GRADE_MAX - grade) * 0.02)
    return max(_EASE_FLOOR, ease + delta)


def sm2(state: ReviewState, grade: int) -> ReviewState:
    """Compute the next :class:`ReviewState` for ``grade`` (0..5). Pure.

    A passing grade (``>= 3``) advances the interval (``1 -> 6 -> round(interval *
    ease)``) and increments ``reps``; a lapse (``grade < 3``) resets ``reps`` and
    ``interval_days`` to 1. In both cases the ease is updated by the standard SM-2
    formula and floored at 1.3. The input ``state`` is never mutated.

    Raises:
        ValueError: if ``grade`` is outside the inclusive ``0..5`` range.
    """
    if not _GRADE_MIN <= grade <= _GRADE_MAX:
        raise ValueError(f"grade must be in 0..5, got {grade}")

    ease = _next_ease(state.ease, grade)

    if grade < _PASS_THRESHOLD:
        # Lapse: the learner failed to recall — restart the schedule.
        return ReviewState(ease=ease, interval_days=1, reps=1)

    reps = state.reps + 1
    if reps == 1:
        interval_days = 1
    elif reps == 2:
        interval_days = 6
    else:
        interval_days = round(state.interval_days * ease)
    return ReviewState(ease=ease, interval_days=interval_days, reps=reps)
