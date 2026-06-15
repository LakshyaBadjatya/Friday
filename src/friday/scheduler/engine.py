"""Dependency-free next-run math + the :class:`Scheduler` (Tier 1).

Two pieces, both clock-injectable and offline-testable:

* :func:`compute_next_run` — pure next-run math over stdlib ``datetime`` /
  ``timedelta`` only (no cron library). Given a ``kind`` + ``spec`` + an
  ``after`` instant it returns the next firing ``datetime`` (or ``None`` when the
  spec is spent/invalid). ``interval`` adds seconds; ``once`` returns the spec
  datetime when strictly in the future; ``daily``/``weekly`` roll forward to the
  next matching ``HH:MM`` (today/tomorrow, this week/next week).
* :class:`Scheduler` — an action registry plus ``tick(now)``, the *tested* unit:
  it runs each due trigger's registered async action, advances
  ``last_run``/``next_run``, disables a fired ``once`` (no future run), persists,
  and returns the fired triggers. An unknown action name (or a handler that
  raises) is logged and skipped — a single bad trigger never crashes the tick.

The thin :meth:`Scheduler.run_loop` wraps ``tick(utcnow)`` + ``asyncio.sleep``;
it is deliberately *not* unit-tested for timing (``tick(now)`` is the unit).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta

from friday.scheduler.store import SQLiteTriggerStore, Trigger, TriggerKind

logger = logging.getLogger("friday.scheduler")

#: A registered action: an async callable handed the firing trigger.
ActionHandler = Callable[[Trigger], Awaitable[None]]

# Day-of-week tokens (``weekly`` spec) -> Python ``weekday()`` index (MON == 0).
_DOW_INDEX: dict[str, int] = {
    "MON": 0,
    "TUE": 1,
    "WED": 2,
    "THU": 3,
    "FRI": 4,
    "SAT": 5,
    "SUN": 6,
}


# --------------------------------------------------------------------------- #
# Pure next-run math
# --------------------------------------------------------------------------- #
def compute_next_run(
    kind: TriggerKind, spec: str, after: datetime
) -> datetime | None:
    """Compute the next firing instant strictly after ``after``, or ``None``.

    Pure and dependency-free (stdlib ``datetime``/``timedelta`` only):

    * ``interval`` — ``after + int(spec)`` seconds (``spec`` must be a positive
      integer).
    * ``once`` — the ``spec`` ISO datetime when it is strictly after ``after``,
      else ``None`` (spent).
    * ``daily`` — the next ``HH:MM`` at or after ``after`` (today if still ahead,
      else tomorrow; a boundary equal to ``after`` rolls to the next day).
    * ``weekly`` — ``"DOW HH:MM"``; the next matching weekday-at-time strictly
      after ``after`` (this week or next week).

    Any malformed spec (bad number, bad datetime, out-of-range ``HH:MM``, unknown
    ``DOW``) returns ``None`` rather than raising, so a misconfigured trigger is
    simply never advanced.
    """
    if kind == "interval":
        return _next_interval(spec, after)
    if kind == "once":
        return _next_once(spec, after)
    if kind == "daily":
        return _next_daily(spec, after)
    if kind == "weekly":
        return _next_weekly(spec, after)
    return None


def _next_interval(spec: str, after: datetime) -> datetime | None:
    try:
        seconds = int(spec)
    except (TypeError, ValueError):
        return None
    if seconds <= 0:
        return None
    return after + timedelta(seconds=seconds)


def _next_once(spec: str, after: datetime) -> datetime | None:
    moment = _parse_iso(spec)
    if moment is None:
        return None
    moment = _align_tz(moment, after)
    return moment if moment > after else None


def _next_daily(spec: str, after: datetime) -> datetime | None:
    hhmm = _parse_hhmm(spec)
    if hhmm is None:
        return None
    hour, minute = hhmm
    candidate = after.replace(
        hour=hour, minute=minute, second=0, microsecond=0
    )
    if candidate <= after:
        candidate += timedelta(days=1)
    return candidate


def _next_weekly(spec: str, after: datetime) -> datetime | None:
    parts = spec.split()
    if len(parts) != 2:
        return None
    dow_token, hhmm_token = parts
    dow = _DOW_INDEX.get(dow_token.strip().upper())
    if dow is None:
        return None
    hhmm = _parse_hhmm(hhmm_token)
    if hhmm is None:
        return None
    hour, minute = hhmm

    candidate = after.replace(hour=hour, minute=minute, second=0, microsecond=0)
    days_ahead = (dow - after.weekday()) % 7
    candidate += timedelta(days=days_ahead)
    if candidate <= after:
        candidate += timedelta(days=7)
    return candidate


def _parse_iso(spec: str) -> datetime | None:
    try:
        return datetime.fromisoformat(spec)
    except (TypeError, ValueError):
        return None


def _parse_hhmm(token: str) -> tuple[int, int] | None:
    pieces = token.split(":")
    if len(pieces) != 2:
        return None
    try:
        hour = int(pieces[0])
        minute = int(pieces[1])
    except ValueError:
        return None
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None
    return hour, minute


def _align_tz(moment: datetime, after: datetime) -> datetime:
    """Make ``moment`` comparable to ``after`` (both naive or both tz-aware).

    Mixed naive/aware ``datetime`` comparisons raise; align the parsed ``once``
    moment to ``after``'s awareness so the comparison is well-defined. A naive
    moment is assumed UTC when ``after`` is aware; an aware moment is dropped to
    naive UTC when ``after`` is naive.
    """
    after_aware = after.tzinfo is not None
    moment_aware = moment.tzinfo is not None
    if after_aware and not moment_aware:
        return moment.replace(tzinfo=UTC)
    if not after_aware and moment_aware:
        return moment.astimezone(UTC).replace(tzinfo=None)
    return moment


# --------------------------------------------------------------------------- #
# Scheduler
# --------------------------------------------------------------------------- #
class Scheduler:
    """An action registry over a :class:`SQLiteTriggerStore` with ``tick(now)``.

    Register named async actions, then call ``tick(now)`` (the tested unit) to
    fire every due trigger. The background :meth:`run_loop` is a thin wrapper over
    ``tick(utcnow)`` and is not unit-tested for timing.
    """

    def __init__(self, store: SQLiteTriggerStore) -> None:
        self._store = store
        self._actions: dict[str, ActionHandler] = {}

    def register_action(self, name: str, handler: ActionHandler) -> None:
        """Register an async ``handler`` under ``name``; reject duplicates."""
        if name in self._actions:
            raise ValueError(f"action {name!r} is already registered")
        self._actions[name] = handler

    async def run_action(self, trigger: Trigger) -> bool:
        """Fire ``trigger``'s registered action once *without* advancing it.

        The fire-now path (the ``/schedules/{id}/run`` route): runs the handler
        regardless of the trigger's ``next_run``/``enabled`` state and does not
        touch ``last_run``/``next_run``. Returns ``True`` when the action ran to
        completion, ``False`` when the action name is unknown or the handler
        raised (both logged) — never propagating the error.
        """
        handler = self._actions.get(trigger.action)
        if handler is None:
            logger.warning(
                "scheduler: run-now unknown action %r for trigger %s (id=%s)",
                trigger.action,
                trigger.name,
                trigger.id,
            )
            return False
        try:
            await handler(trigger)
        except Exception:  # noqa: BLE001 - report failure, don't propagate
            logger.exception(
                "scheduler: run-now action %r failed for trigger %s (id=%s)",
                trigger.action,
                trigger.name,
                trigger.id,
            )
            return False
        return True

    async def tick(self, now: datetime) -> list[Trigger]:
        """Fire every due trigger as of ``now``; return those that fired.

        For each due trigger, runs its registered action, then sets
        ``last_run = now`` and ``next_run = compute_next_run(...)`` (a ``None``
        next run for a fired ``once`` disables it), persists the change, and
        includes it in the returned list. A trigger whose action name is unknown
        — or whose handler raises — is logged and skipped: it is *not* advanced
        and does *not* appear in the returned list, and the tick continues.
        """
        fired: list[Trigger] = []
        for trigger in self._store.due(now):
            handler = self._actions.get(trigger.action)
            if handler is None:
                logger.warning(
                    "scheduler: unknown action %r for trigger %s (id=%s); skipping",
                    trigger.action,
                    trigger.name,
                    trigger.id,
                )
                continue
            try:
                await handler(trigger)
            except Exception:  # noqa: BLE001 - one bad trigger must not crash tick
                logger.exception(
                    "scheduler: action %r failed for trigger %s (id=%s); skipping",
                    trigger.action,
                    trigger.name,
                    trigger.id,
                )
                continue

            next_run = compute_next_run(trigger.kind, trigger.spec, now)
            trigger.last_run = now.isoformat()
            trigger.next_run = None if next_run is None else next_run.isoformat()
            if trigger.kind == "once" or next_run is None:
                # A fired once (or any spec with no future run) is spent.
                trigger.enabled = False
            self._store.update(trigger)
            fired.append(trigger)
        return fired

    async def run_loop(self, tick_seconds: float) -> None:  # pragma: no cover
        """Tick forever every ``tick_seconds`` using the wall clock (utcnow).

        Deliberately *not* unit-tested for timing — it is a thin wrapper over the
        tested :meth:`tick`. Runs until cancelled (the app cancels it on
        shutdown); a failing tick is logged and the loop continues.
        """
        logger.info("scheduler run_loop started", extra={"tick_seconds": tick_seconds})
        try:
            while True:
                try:
                    await self.tick(datetime.now(UTC))
                except Exception:  # noqa: BLE001 - never let the loop die on one tick
                    logger.exception("scheduler tick failed")
                await asyncio.sleep(tick_seconds)
        except asyncio.CancelledError:
            logger.info("scheduler run_loop cancelled")
            raise
