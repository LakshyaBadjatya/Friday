"""Unit tests for the deterministic :class:`Foresight` suggester (proactive slice).

Fully dependency-injected and offline: every case passes an explicit ``events``
list and a fixed ``now`` (a :class:`datetime`). No real clock, no network. The
optional ``llm`` phraser is a tiny in-process stub or a deliberately-raising one
to prove the LLM path is non-fatal.

Foresight applies three deterministic rules over the events:

* ``metric`` events with a rising ``value`` over time -> an "trending up"
  suggestion that names the metric.
* ``reminder`` events with a ``due`` timestamp within the look-ahead window ->
  a "due soon" suggestion.
* a label that recurs on a regular cadence -> a "recurring pattern" suggestion.

Covered:
* A planted upward metric trend yields exactly one trend suggestion naming it.
* An empty event list yields no suggestions.
* A reminder due within the window is surfaced; one far out is not.
* A recurring weekly label is surfaced.
* An LLM phraser rewrites the text; a raising LLM is swallowed (rules still fire
  with their default phrasing).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from friday.proactive import Foresight
from friday.proactive.foresight import Suggestion

NOW = datetime(2026, 6, 15, 9, 0, tzinfo=UTC)


def _ts(minutes: int = 0, hours: int = 0, days: int = 0) -> str:
    """An ISO timestamp offset from :data:`NOW` (positive = future)."""
    return (NOW + timedelta(minutes=minutes, hours=hours, days=days)).isoformat()


def test_empty_events_yield_no_suggestions() -> None:
    """No events means no suggestions (and never raises)."""
    assert Foresight().suggest([], now=NOW) == []


def test_mixed_naive_and_aware_timestamps_do_not_raise() -> None:
    # datetime.fromisoformat accepts both naive and aware ISO strings; comparing a
    # mix raises TypeError. suggest() promises to be total, so it must normalize.
    events = [
        {"type": "metric", "name": "cpu", "at": "2026-06-15T09:00", "value": 1.0},
        {"type": "metric", "name": "cpu", "at": "2026-06-15T10:00+00:00", "value": 2.0},
    ]
    suggestions = Foresight().suggest(events, now=NOW)
    assert isinstance(suggestions, list)

    # A tz-naive reminder due against the aware NOW must also not raise.
    reminder_events = [{"type": "reminder", "title": "renew", "due": "2026-06-15T09:30"}]
    assert isinstance(Foresight().suggest(reminder_events, now=NOW), list)


def test_rising_metric_trend_is_surfaced() -> None:
    """A metric whose value climbs over time produces one trend suggestion."""
    events = [
        {"type": "metric", "name": "cpu", "value": 10.0, "at": _ts(minutes=-30)},
        {"type": "metric", "name": "cpu", "value": 20.0, "at": _ts(minutes=-20)},
        {"type": "metric", "name": "cpu", "value": 35.0, "at": _ts(minutes=-10)},
        {"type": "metric", "name": "cpu", "value": 55.0, "at": _ts(minutes=-1)},
    ]

    suggestions = Foresight().suggest(events, now=NOW)

    trends = [s for s in suggestions if "trend" in s.reason]
    assert len(trends) == 1
    assert isinstance(trends[0], Suggestion)
    assert "cpu" in trends[0].text


def test_flat_metric_does_not_trend() -> None:
    """A metric holding steady is not reported as a trend."""
    events = [
        {"type": "metric", "name": "cpu", "value": 10.0, "at": _ts(minutes=-30)},
        {"type": "metric", "name": "cpu", "value": 10.0, "at": _ts(minutes=-20)},
        {"type": "metric", "name": "cpu", "value": 10.0, "at": _ts(minutes=-10)},
    ]

    assert [s for s in Foresight().suggest(events, now=NOW) if "trend" in s.reason] == []


def test_reminder_due_soon_is_surfaced_but_not_far_future() -> None:
    """A reminder inside the look-ahead window fires; a distant one does not."""
    events = [
        {"type": "reminder", "title": "standup", "due": _ts(minutes=15)},
        {"type": "reminder", "title": "vacation", "due": _ts(days=30)},
    ]

    suggestions = Foresight().suggest(events, now=NOW)

    due = [s for s in suggestions if "due" in s.reason]
    assert len(due) == 1
    assert "standup" in due[0].text


def test_recurring_pattern_is_surfaced() -> None:
    """A label recurring on a regular cadence yields a pattern suggestion."""
    events = [
        {"type": "activity", "label": "weekly_report", "at": _ts(days=-21)},
        {"type": "activity", "label": "weekly_report", "at": _ts(days=-14)},
        {"type": "activity", "label": "weekly_report", "at": _ts(days=-7)},
    ]

    suggestions = Foresight().suggest(events, now=NOW)

    patterns = [s for s in suggestions if "recurring" in s.reason]
    assert len(patterns) == 1
    assert "weekly_report" in patterns[0].text


def test_llm_phraser_rewrites_text() -> None:
    """An injected phraser may rewrite suggestion text; reason is preserved."""

    def phraser(text: str) -> str:
        return f"[polished] {text}"

    events = [
        {"type": "reminder", "title": "standup", "due": _ts(minutes=15)},
    ]

    suggestions = Foresight(llm=phraser).suggest(events, now=NOW)

    assert len(suggestions) == 1
    assert suggestions[0].text.startswith("[polished] ")


def test_raising_llm_is_non_fatal() -> None:
    """A phraser that raises is swallowed; the rule still fires with default text."""

    def boom(text: str) -> str:
        raise RuntimeError("llm down")

    events = [
        {"type": "reminder", "title": "standup", "due": _ts(minutes=15)},
    ]

    suggestions = Foresight(llm=boom).suggest(events, now=NOW)

    assert len(suggestions) == 1
    assert "standup" in suggestions[0].text
    assert not suggestions[0].text.startswith("[")


def test_suggest_is_deterministic() -> None:
    """The same inputs yield identical suggestions across calls."""
    events = [
        {"type": "metric", "name": "mem", "value": 1.0, "at": _ts(minutes=-30)},
        {"type": "metric", "name": "mem", "value": 2.0, "at": _ts(minutes=-20)},
        {"type": "metric", "name": "mem", "value": 4.0, "at": _ts(minutes=-10)},
        {"type": "reminder", "title": "standup", "due": _ts(minutes=15)},
    ]
    foresight = Foresight()

    first = foresight.suggest(events, now=NOW)
    second = foresight.suggest(events, now=NOW)

    assert [(s.text, s.reason) for s in first] == [(s.text, s.reason) for s in second]
