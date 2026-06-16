# © Lakshya Badjatya — Author
"""Watchers: pure detectors that turn raw signals into something worth acting on.

These are the "what changed that I should care about?" half of the proactive
spine — calendar-conflict detection and numeric threshold breaches — kept as pure
functions so they are exhaustively testable and side-effect free. Acting on a
finding (notify, escalate) is the caller's job, typically via the rules engine
and the broker.

No LLM SDK, no configuration, no I/O — inputs (events, prices, thresholds) are all
passed in.
"""

from __future__ import annotations

from collections.abc import Iterable

from pydantic import BaseModel, model_validator


class TimeEvent(BaseModel):
    """A scheduled event over a half-open time interval ``[start, end)``.

    ``start``/``end`` are comparable timestamps (e.g. epoch seconds); ``end`` must
    not precede ``start``.
    """

    title: str
    start: float
    end: float

    @model_validator(mode="after")
    def _check_interval(self) -> TimeEvent:
        if self.end < self.start:
            raise ValueError("event end must not precede start")
        return self


class Conflict(BaseModel):
    """Two events whose intervals overlap."""

    a: str
    b: str


def find_conflicts(events: Iterable[TimeEvent]) -> list[Conflict]:
    """Return every pair of overlapping events (touching endpoints do not overlap).

    Events are processed in start order; because of that ordering, once a later
    event starts at or after the current event's end, no still-later event can
    overlap the current one either — so the inner scan stops early.
    """
    ordered = sorted(events, key=lambda e: e.start)
    conflicts: list[Conflict] = []
    for i, first in enumerate(ordered):
        for second in ordered[i + 1 :]:
            if second.start >= first.end:
                break  # sorted by start -> nothing further overlaps `first`
            conflicts.append(Conflict(a=first.title, b=second.title))
    return conflicts


def price_breach(
    price: float, *, above: float | None = None, below: float | None = None
) -> bool:
    """Whether ``price`` breached an ``above`` ceiling or a ``below`` floor.

    A breach is ``price > above`` or ``price < below`` (boundary-equal is calm).
    With neither bound set, nothing ever breaches.
    """
    if above is not None and price > above:
        return True
    return below is not None and price < below
