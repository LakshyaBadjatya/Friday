"""Deterministic, rule-based foresight suggestions (proactive slice).

:class:`Foresight` turns a list of recent event dicts into a short list of
:class:`Suggestion` objects ("you might want to ...") using simple, explainable
rules — **no LLM is required**. Every input is injected (the events and the
reference ``now``); there is no clock, no network and no FRIDAY-config / app
dependency in this module.

Rules applied (each independent and order-stable):

* **Rising metric** — for events ``{"type": "metric", "name", "value", "at"}``
  grouped by ``name``, if the values are (weakly) increasing over time and the
  last value exceeds the first, emit a "trending up" suggestion.
* **Reminder due soon** — for events ``{"type": "reminder", "title", "due"}``
  whose ``due`` timestamp falls within the look-ahead window after ``now``, emit
  a "due soon" suggestion.
* **Recurring pattern** — for events sharing a ``label`` (``{"type": ...,
  "label", "at"}``) seen on a regular cadence (>= 3 occurrences at roughly even
  spacing), emit a "recurring pattern" suggestion anticipating the next one.

An optional ``llm`` phraser may rewrite a suggestion's user-facing ``text``; it
is **best-effort and non-fatal**: if it raises, the default phrasing is kept and
no exception escapes :meth:`Foresight.suggest`. The phraser is a plain callable
(``str -> str``) so this module never imports an LLM SDK.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel

#: Default look-ahead horizon for "due soon" reminders.
_DEFAULT_LOOKAHEAD = timedelta(hours=24)

#: Minimum occurrences before a recurring cadence is considered established.
_MIN_RECURRENCES = 3

#: Allowed jitter (as a fraction of the median gap) for a cadence to count as
#: "regular". 0.25 -> gaps within +/-25 % of the median are treated as even.
_CADENCE_TOLERANCE = 0.25


@runtime_checkable
class Phraser(Protocol):
    """Structural contract for the optional LLM text phraser.

    Any callable taking the default text and returning a polished string
    satisfies this; the phraser must not be relied upon (it may raise, in which
    case the default text is used).
    """

    def __call__(self, text: str) -> str: ...


class Suggestion(BaseModel):
    """A single proactive suggestion surfaced to the user.

    ``text`` is the user-facing phrasing; ``reason`` is a short, stable machine
    tag explaining which rule produced it (``"trend"`` / ``"due"`` /
    ``"recurring"`` substrings are part of the contract callers may match on).
    """

    text: str
    reason: str


class Foresight:
    """Generate deterministic suggestions from injected events.

    Args:
        lookahead: How far ahead of ``now`` a reminder must fall to be surfaced.
        llm: Optional best-effort phraser (``str -> str``) used to polish
            suggestion text. Never required; failures are swallowed.
    """

    def __init__(
        self,
        *,
        lookahead: timedelta = _DEFAULT_LOOKAHEAD,
        llm: Callable[[str], str] | None = None,
    ) -> None:
        self.lookahead = lookahead
        self._llm = llm

    def suggest(self, events: list[dict[str, Any]], now: datetime) -> list[Suggestion]:
        """Return suggestions derived from ``events`` relative to ``now``.

        Deterministic and total: an empty (or wholly irrelevant) event list
        yields ``[]`` and the method never raises on malformed individual events
        — entries missing required keys for a rule are simply skipped.
        """
        raw: list[Suggestion] = []
        raw.extend(self._metric_trends(events))
        raw.extend(self._due_reminders(events, now))
        raw.extend(self._recurring_patterns(events))
        return [self._phrase(suggestion) for suggestion in raw]

    # ------------------------------------------------------------------ #
    # Rules
    # ------------------------------------------------------------------ #
    def _metric_trends(self, events: list[dict[str, Any]]) -> list[Suggestion]:
        """Emit one suggestion per metric whose value trends upward over time."""
        by_name: dict[str, list[tuple[datetime, float]]] = defaultdict(list)
        for event in events:
            if event.get("type") != "metric":
                continue
            name = event.get("name")
            at = _parse_dt(event.get("at"))
            value = _as_float(event.get("value"))
            if name is None or at is None or value is None:
                continue
            by_name[str(name)].append((at, value))

        suggestions: list[Suggestion] = []
        for name in sorted(by_name):
            points = sorted(by_name[name], key=lambda pair: pair[0])
            if not _is_rising(points):
                continue
            first, last = points[0][1], points[-1][1]
            suggestions.append(
                Suggestion(
                    text=f"'{name}' is trending up ({first:g} -> {last:g}).",
                    reason="trend",
                )
            )
        return suggestions

    def _due_reminders(
        self, events: list[dict[str, Any]], now: datetime
    ) -> list[Suggestion]:
        """Emit a suggestion per reminder due within the look-ahead window."""
        now = _aware(now)  # parsed dues are aware; keep the window bounds aware too
        horizon = now + self.lookahead
        candidates: list[tuple[datetime, str]] = []
        for event in events:
            if event.get("type") != "reminder":
                continue
            due = _parse_dt(event.get("due"))
            title = event.get("title")
            if due is None or title is None:
                continue
            if now <= due <= horizon:
                candidates.append((due, str(title)))

        return [
            Suggestion(text=f"'{title}' is due soon.", reason="due")
            for _due, title in sorted(candidates, key=lambda pair: pair[0])
        ]

    def _recurring_patterns(self, events: list[dict[str, Any]]) -> list[Suggestion]:
        """Emit a suggestion per label recurring on a regular cadence."""
        by_label: dict[str, list[datetime]] = defaultdict(list)
        for event in events:
            label = event.get("label")
            at = _parse_dt(event.get("at"))
            if label is None or at is None:
                continue
            by_label[str(label)].append(at)

        suggestions: list[Suggestion] = []
        for label in sorted(by_label):
            times = sorted(by_label[label])
            if _is_regular_cadence(times):
                suggestions.append(
                    Suggestion(
                        text=f"'{label}' recurs regularly; another is likely due.",
                        reason="recurring",
                    )
                )
        return suggestions

    # ------------------------------------------------------------------ #
    # Optional LLM phrasing (best-effort, non-fatal)
    # ------------------------------------------------------------------ #
    def _phrase(self, suggestion: Suggestion) -> Suggestion:
        """Return ``suggestion`` with optionally LLM-polished text.

        If no phraser is configured, or it raises, or it returns an empty/blank
        string, the original text is kept untouched.
        """
        if self._llm is None:
            return suggestion
        try:
            polished = self._llm(suggestion.text)
        except Exception:  # noqa: BLE001 - phraser is best-effort; never fatal
            return suggestion
        if not isinstance(polished, str) or not polished.strip():
            return suggestion
        return Suggestion(text=polished, reason=suggestion.reason)


# --------------------------------------------------------------------------- #
# Helpers (pure)
# --------------------------------------------------------------------------- #
def _aware(dt: datetime) -> datetime:
    """Coerce a naive datetime to UTC so naive/aware values never mix in compares.

    ``datetime.fromisoformat`` accepts both ``"...T09:00"`` (naive) and
    ``"...T09:00+00:00"`` (aware); sorting or comparing a mix raises ``TypeError``.
    Normalizing every parsed/inbound datetime to aware (treating naive as UTC)
    keeps :meth:`Foresight.suggest` total — it never raises on mixed timestamps.
    """
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)


def _parse_dt(raw: Any) -> datetime | None:
    """Coerce ``raw`` to a tz-aware :class:`datetime`, returning ``None`` if impossible."""
    if isinstance(raw, datetime):
        return _aware(raw)
    if isinstance(raw, str):
        try:
            return _aware(datetime.fromisoformat(raw))
        except ValueError:
            return None
    return None


def _as_float(raw: Any) -> float | None:
    """Coerce ``raw`` to ``float``, returning ``None`` on failure."""
    if isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    return None


def _is_rising(points: list[tuple[datetime, float]]) -> bool:
    """True if the time-ordered ``points`` are weakly increasing and net up."""
    if len(points) < 2:
        return False
    values = [value for _at, value in points]
    weakly_increasing = all(b >= a for a, b in zip(values, values[1:], strict=False))
    return weakly_increasing and values[-1] > values[0]


def _is_regular_cadence(times: list[datetime]) -> bool:
    """True if the sorted ``times`` recur at roughly even spacing.

    Requires at least :data:`_MIN_RECURRENCES` points; every consecutive gap
    must fall within :data:`_CADENCE_TOLERANCE` of the median gap (and gaps must
    be strictly positive).
    """
    if len(times) < _MIN_RECURRENCES:
        return False
    gaps = [
        (later - earlier).total_seconds()
        for earlier, later in zip(times, times[1:], strict=False)
    ]
    if any(gap <= 0.0 for gap in gaps):
        return False
    median = sorted(gaps)[len(gaps) // 2]
    if median <= 0.0:
        return False
    return all(abs(gap - median) <= _CADENCE_TOLERANCE * median for gap in gaps)
