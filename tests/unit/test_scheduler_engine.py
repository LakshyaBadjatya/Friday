"""Unit tests for the scheduler engine (Tier 1).

Two units under test, both offline and clock-injected:

* :func:`friday.scheduler.engine.compute_next_run` — pure, dependency-free
  next-run math over stdlib ``datetime``/``timedelta``. Covers interval / once /
  daily / weekly including roll-forward to tomorrow / next week and a ``once`` in
  the past returning ``None``.
* :class:`friday.scheduler.engine.Scheduler` — ``tick(now)`` runs each due
  trigger's registered async action, advances ``last_run``/``next_run``, disables
  a fired ``once``, persists, and returns the fired triggers; an unknown action
  name is logged and skipped (never crashes the tick).

The thin ``run_loop`` (a wrapper over ``tick(utcnow)`` + ``asyncio.sleep``) is
deliberately *not* unit-tested for timing — ``tick(now)`` is the tested unit.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from friday.scheduler.engine import Scheduler, compute_next_run
from friday.scheduler.store import SQLiteTriggerStore, Trigger


def _dt(iso: str) -> datetime:
    return datetime.fromisoformat(iso)


# --------------------------------------------------------------------------- #
# compute_next_run — interval
# --------------------------------------------------------------------------- #
def test_interval_adds_seconds() -> None:
    after = _dt("2026-06-15T12:00:00+00:00")
    nxt = compute_next_run("interval", "90", after)
    assert nxt == _dt("2026-06-15T12:01:30+00:00")


def test_interval_invalid_spec_is_none() -> None:
    after = _dt("2026-06-15T12:00:00+00:00")
    assert compute_next_run("interval", "not-a-number", after) is None
    assert compute_next_run("interval", "0", after) is None
    assert compute_next_run("interval", "-5", after) is None


# --------------------------------------------------------------------------- #
# compute_next_run — once
# --------------------------------------------------------------------------- #
def test_once_future_returns_spec_datetime() -> None:
    after = _dt("2026-06-15T12:00:00+00:00")
    nxt = compute_next_run("once", "2026-06-15T18:00:00+00:00", after)
    assert nxt == _dt("2026-06-15T18:00:00+00:00")


def test_once_in_the_past_is_none() -> None:
    after = _dt("2026-06-15T12:00:00+00:00")
    assert compute_next_run("once", "2026-06-15T08:00:00+00:00", after) is None


def test_once_exactly_now_is_none() -> None:
    """A ``once`` whose moment equals ``after`` has no *future* run -> None."""
    after = _dt("2026-06-15T12:00:00+00:00")
    assert compute_next_run("once", "2026-06-15T12:00:00+00:00", after) is None


def test_once_invalid_spec_is_none() -> None:
    after = _dt("2026-06-15T12:00:00+00:00")
    assert compute_next_run("once", "not-a-datetime", after) is None


# --------------------------------------------------------------------------- #
# compute_next_run — daily
# --------------------------------------------------------------------------- #
def test_daily_later_today() -> None:
    after = _dt("2026-06-15T08:00:00+00:00")
    nxt = compute_next_run("daily", "09:30", after)
    assert nxt == _dt("2026-06-15T09:30:00+00:00")


def test_daily_rolls_to_tomorrow_when_past() -> None:
    after = _dt("2026-06-15T10:00:00+00:00")
    nxt = compute_next_run("daily", "09:30", after)
    assert nxt == _dt("2026-06-16T09:30:00+00:00")


def test_daily_exactly_now_rolls_forward() -> None:
    """At the boundary the next run is the *next* day's occurrence."""
    after = _dt("2026-06-15T09:30:00+00:00")
    nxt = compute_next_run("daily", "09:30", after)
    assert nxt == _dt("2026-06-16T09:30:00+00:00")


def test_daily_invalid_spec_is_none() -> None:
    after = _dt("2026-06-15T08:00:00+00:00")
    assert compute_next_run("daily", "25:00", after) is None
    assert compute_next_run("daily", "9-30", after) is None
    assert compute_next_run("daily", "", after) is None


# --------------------------------------------------------------------------- #
# compute_next_run — weekly
# --------------------------------------------------------------------------- #
def test_weekly_later_this_week() -> None:
    # 2026-06-15 is a Monday. Next WED 09:00 is 2026-06-17.
    after = _dt("2026-06-15T08:00:00+00:00")
    nxt = compute_next_run("weekly", "WED 09:00", after)
    assert nxt == _dt("2026-06-17T09:00:00+00:00")


def test_weekly_same_day_later_time() -> None:
    after = _dt("2026-06-15T08:00:00+00:00")  # Monday
    nxt = compute_next_run("weekly", "MON 09:00", after)
    assert nxt == _dt("2026-06-15T09:00:00+00:00")


def test_weekly_rolls_to_next_week_when_past() -> None:
    after = _dt("2026-06-15T10:00:00+00:00")  # Monday, after 09:00
    nxt = compute_next_run("weekly", "MON 09:00", after)
    assert nxt == _dt("2026-06-22T09:00:00+00:00")


def test_weekly_earlier_day_rolls_to_next_week() -> None:
    after = _dt("2026-06-17T12:00:00+00:00")  # Wednesday
    nxt = compute_next_run("weekly", "TUE 08:00", after)
    assert nxt == _dt("2026-06-23T08:00:00+00:00")


def test_weekly_is_case_insensitive() -> None:
    after = _dt("2026-06-15T08:00:00+00:00")  # Monday
    nxt = compute_next_run("weekly", "wed 09:00", after)
    assert nxt == _dt("2026-06-17T09:00:00+00:00")


def test_weekly_invalid_spec_is_none() -> None:
    after = _dt("2026-06-15T08:00:00+00:00")
    assert compute_next_run("weekly", "FUNDAY 09:00", after) is None
    assert compute_next_run("weekly", "MON", after) is None
    assert compute_next_run("weekly", "MON 25:00", after) is None


def test_unknown_kind_is_none() -> None:
    after = _dt("2026-06-15T08:00:00+00:00")
    assert compute_next_run("nonsense", "x", after) is None  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Scheduler.tick
# --------------------------------------------------------------------------- #
def _scheduler(tmp_path: object) -> tuple[Scheduler, SQLiteTriggerStore]:
    store = SQLiteTriggerStore(str(tmp_path) + "/triggers.db")  # type: ignore[operator]
    return Scheduler(store), store


async def test_tick_runs_registered_action_and_advances(tmp_path: object) -> None:
    scheduler, store = _scheduler(tmp_path)
    calls: list[Trigger] = []

    async def handler(trigger: Trigger) -> None:
        calls.append(trigger)

    scheduler.register_action("counter", handler)
    t = store.add(
        name="every-minute",
        kind="interval",
        spec="60",
        action="counter",
        next_run="2026-06-15T11:59:00+00:00",
    )

    now = _dt("2026-06-15T12:00:00+00:00")
    fired = await scheduler.tick(now)

    assert [f.id for f in fired] == [t.id]
    assert len(calls) == 1 and calls[0].id == t.id

    reloaded = store.get(t.id)
    assert reloaded is not None
    assert reloaded.last_run == now.isoformat()
    # interval -> next run is now + 60s.
    assert reloaded.next_run == _dt("2026-06-15T12:01:00+00:00").isoformat()
    assert reloaded.enabled is True


async def test_tick_disables_fired_once(tmp_path: object) -> None:
    scheduler, store = _scheduler(tmp_path)

    async def handler(trigger: Trigger) -> None:
        return None

    scheduler.register_action("noop", handler)
    t = store.add(
        name="one-shot",
        kind="once",
        spec="2026-06-15T12:00:00+00:00",
        action="noop",
        next_run="2026-06-15T12:00:00+00:00",
    )

    now = _dt("2026-06-15T12:00:00+00:00")
    fired = await scheduler.tick(now)

    assert [f.id for f in fired] == [t.id]
    reloaded = store.get(t.id)
    assert reloaded is not None
    assert reloaded.enabled is False  # a fired once is disabled
    assert reloaded.next_run is None
    assert reloaded.last_run == now.isoformat()
    # It is no longer due on a later tick.
    assert await scheduler.tick(_dt("2026-06-15T13:00:00+00:00")) == []


async def test_tick_skips_unknown_action_without_crashing(
    tmp_path: object,
) -> None:
    scheduler, store = _scheduler(tmp_path)
    t = store.add(
        name="orphan",
        kind="interval",
        spec="60",
        action="does_not_exist",
        next_run="2026-06-15T11:00:00+00:00",
    )

    now = _dt("2026-06-15T12:00:00+00:00")
    fired = await scheduler.tick(now)  # must not raise

    assert fired == []
    # The trigger was not advanced (it never ran) and is left as-is for the
    # operator to fix.
    reloaded = store.get(t.id)
    assert reloaded is not None
    assert reloaded.last_run is None
    assert reloaded.next_run == "2026-06-15T11:00:00+00:00"


async def test_tick_only_runs_due_triggers(tmp_path: object) -> None:
    scheduler, store = _scheduler(tmp_path)
    ran: list[str] = []

    async def handler(trigger: Trigger) -> None:
        ran.append(trigger.name)

    scheduler.register_action("track", handler)
    store.add(
        name="due",
        kind="interval",
        spec="60",
        action="track",
        next_run="2026-06-15T11:00:00+00:00",
    )
    store.add(
        name="not-due",
        kind="interval",
        spec="60",
        action="track",
        next_run="2026-06-15T18:00:00+00:00",
    )

    await scheduler.tick(_dt("2026-06-15T12:00:00+00:00"))
    assert ran == ["due"]


async def test_tick_action_error_does_not_crash_tick(tmp_path: object) -> None:
    """A handler raising must not crash the tick; the trigger is left unadvanced."""
    scheduler, store = _scheduler(tmp_path)

    async def boom(trigger: Trigger) -> None:
        raise RuntimeError("handler exploded")

    scheduler.register_action("boom", boom)
    t = store.add(
        name="explodes",
        kind="interval",
        spec="60",
        action="boom",
        next_run="2026-06-15T11:00:00+00:00",
    )

    fired = await scheduler.tick(_dt("2026-06-15T12:00:00+00:00"))
    assert fired == []
    reloaded = store.get(t.id)
    assert reloaded is not None
    assert reloaded.last_run is None


def test_register_action_rejects_duplicate(tmp_path: object) -> None:
    scheduler, _ = _scheduler(tmp_path)

    async def handler(trigger: Trigger) -> None:
        return None

    scheduler.register_action("dup", handler)
    with pytest.raises(ValueError):
        scheduler.register_action("dup", handler)


async def test_tick_with_utc_aware_now(tmp_path: object) -> None:
    """A tz-aware UTC now (the run_loop path) ticks the same as a parsed ISO."""
    scheduler, store = _scheduler(tmp_path)
    seen: list[str] = []

    async def handler(trigger: Trigger) -> None:
        seen.append(trigger.name)

    scheduler.register_action("track", handler)
    store.add(
        name="due",
        kind="interval",
        spec="30",
        action="track",
        next_run="2026-06-15T11:00:00+00:00",
    )
    await scheduler.tick(datetime(2026, 6, 15, 12, 0, 0, tzinfo=UTC))
    assert seen == ["due"]
