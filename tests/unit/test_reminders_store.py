"""Unit tests for :class:`friday.reminders.store.SQLiteReminderStore` (Tier 1).

The reminder store is the local-first, SQLite-backed durable layer for the
reminders feature. Every test here is offline, uses a *tmp-file* database (so the
connection-per-call path is exercised and the store is thread-safe by
construction), and injects a deterministic ``clock`` so ``due()`` never reads the
wall clock.

Pinned behaviours:

* ``add`` returns a typed :class:`Reminder` with an assigned id and ``open``
  status; the schema init is idempotent (constructing twice over one file is
  safe).
* ``list_reminders`` returns open reminders soonest-due first, then by creation
  order; ``status="all"`` includes completed ones.
* ``due(now_iso)`` returns only open reminders whose ``due_at <= now`` — it is
  driven entirely by the passed timestamp, never the wall clock.
* ``complete`` flips a one-shot reminder ``open -> done``; a *recurring* reminder
  stays ``open`` and rolls its ``due_at`` forward by the recurrence period.
* ``delete`` removes a row and returns the number removed.
"""

from __future__ import annotations

from pathlib import Path

from friday.reminders.store import Reminder, SQLiteReminderStore


def _store(tmp_path: Path, *, now: float = 1_000_000.0) -> SQLiteReminderStore:
    """A tmp-file store with a fixed injected clock (no wall-clock reads)."""
    return SQLiteReminderStore(str(tmp_path / "reminders.db"), clock=lambda: now)


# --------------------------------------------------------------------------- #
# add + schema
# --------------------------------------------------------------------------- #
def test_add_returns_typed_open_reminder(tmp_path: Path) -> None:
    store = _store(tmp_path)
    r = store.add("call the dentist", due_at="2026-06-16T09:00:00+00:00")

    assert isinstance(r, Reminder)
    assert r.id >= 1
    assert r.text == "call the dentist"
    assert r.due_at == "2026-06-16T09:00:00+00:00"
    assert r.recurrence is None
    assert r.status == "open"
    assert r.created_at  # a non-empty ISO timestamp


def test_add_without_due_at_is_allowed(tmp_path: Path) -> None:
    store = _store(tmp_path)
    r = store.add("buy milk")
    assert r.due_at is None
    assert r.status == "open"


def test_schema_init_is_idempotent(tmp_path: Path) -> None:
    db = str(tmp_path / "reminders.db")
    first = SQLiteReminderStore(db, clock=lambda: 0.0)
    first.add("persisted")
    # Re-opening the same file must not clobber existing rows.
    second = SQLiteReminderStore(db, clock=lambda: 0.0)
    assert [r.text for r in second.list_reminders(status="all")] == ["persisted"]


# --------------------------------------------------------------------------- #
# list ordering
# --------------------------------------------------------------------------- #
def test_list_reminders_orders_soonest_due_first(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.add("later", due_at="2026-06-20T00:00:00+00:00")
    store.add("sooner", due_at="2026-06-16T00:00:00+00:00")
    store.add("middle", due_at="2026-06-18T00:00:00+00:00")

    texts = [r.text for r in store.list_reminders()]
    assert texts == ["sooner", "middle", "later"]


def test_list_reminders_no_due_sorts_after_dated_then_by_creation(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    store.add("no-due-first")
    store.add("dated", due_at="2026-06-16T00:00:00+00:00")
    store.add("no-due-second")

    texts = [r.text for r in store.list_reminders()]
    # Dated reminders sort ahead of undated ones; undated keep creation order.
    assert texts == ["dated", "no-due-first", "no-due-second"]


def test_list_status_open_excludes_done(tmp_path: Path) -> None:
    store = _store(tmp_path)
    keep = store.add("keep", due_at="2026-06-16T00:00:00+00:00")
    done = store.add("finish me", due_at="2026-06-17T00:00:00+00:00")
    store.complete(done.id)

    open_texts = [r.text for r in store.list_reminders(status="open")]
    assert open_texts == ["keep"]
    all_texts = {r.text for r in store.list_reminders(status="all")}
    assert all_texts == {"keep", "finish me"}
    assert keep.id  # silence unused


# --------------------------------------------------------------------------- #
# due() respects the injected clock / passed timestamp
# --------------------------------------------------------------------------- #
def test_due_filters_by_passed_timestamp(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.add("past", due_at="2026-06-15T08:00:00+00:00")
    store.add("future", due_at="2026-06-20T08:00:00+00:00")
    store.add("exactly-now", due_at="2026-06-16T12:00:00+00:00")

    due = store.due("2026-06-16T12:00:00+00:00")
    texts = {r.text for r in due}
    # ``past`` and the exact-now boundary are due; ``future`` is not.
    assert texts == {"past", "exactly-now"}


def test_due_excludes_done_and_undated(tmp_path: Path) -> None:
    store = _store(tmp_path)
    done = store.add("done", due_at="2026-06-15T00:00:00+00:00")
    store.complete(done.id)
    store.add("undated")  # no due_at -> never due

    due = store.due("2026-06-30T00:00:00+00:00")
    assert due == []


# --------------------------------------------------------------------------- #
# complete: one-shot vs recurring
# --------------------------------------------------------------------------- #
def test_complete_one_shot_flips_open_to_done(tmp_path: Path) -> None:
    store = _store(tmp_path)
    r = store.add("one-shot", due_at="2026-06-16T00:00:00+00:00")

    assert store.complete(r.id) is True
    rows = store.list_reminders(status="all")
    assert len(rows) == 1
    assert rows[0].status == "done"
    # No longer in the open list.
    assert store.list_reminders(status="open") == []


def test_complete_unknown_id_returns_false(tmp_path: Path) -> None:
    store = _store(tmp_path)
    assert store.complete(9999) is False


def test_complete_daily_recurrence_rolls_forward_one_day(tmp_path: Path) -> None:
    store = _store(tmp_path)
    r = store.add(
        "take pills",
        due_at="2026-06-16T09:00:00+00:00",
        recurrence="daily",
    )

    assert store.complete(r.id) is True
    rows = store.list_reminders(status="all")
    assert len(rows) == 1
    rolled = rows[0]
    # A recurring reminder stays OPEN and its due_at advances by one period.
    assert rolled.status == "open"
    assert rolled.due_at == "2026-06-17T09:00:00+00:00"
    assert rolled.recurrence == "daily"


def test_complete_weekly_recurrence_rolls_forward_seven_days(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    r = store.add(
        "weekly review",
        due_at="2026-06-16T09:00:00+00:00",
        recurrence="weekly",
    )

    assert store.complete(r.id) is True
    rolled = store.list_reminders(status="all")[0]
    assert rolled.status == "open"
    assert rolled.due_at == "2026-06-23T09:00:00+00:00"


def test_complete_recurring_without_due_at_falls_back_to_done(
    tmp_path: Path,
) -> None:
    # A recurrence with no anchor date cannot be rolled forward, so it completes
    # as a one-shot (documented fallback) rather than silently staying open.
    store = _store(tmp_path)
    r = store.add("recurring but undated", recurrence="daily")

    assert store.complete(r.id) is True
    rolled = store.list_reminders(status="all")[0]
    assert rolled.status == "done"


# --------------------------------------------------------------------------- #
# delete
# --------------------------------------------------------------------------- #
def test_delete_removes_row_and_returns_count(tmp_path: Path) -> None:
    store = _store(tmp_path)
    r = store.add("temporary")

    assert store.delete(r.id) == 1
    assert store.list_reminders(status="all") == []
    # Deleting again is a no-op (0 rows removed).
    assert store.delete(r.id) == 0
