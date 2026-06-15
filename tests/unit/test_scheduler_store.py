"""Unit tests for :class:`friday.scheduler.store.SQLiteTriggerStore` (Tier 1).

The trigger store is the local-first, SQLite-backed durable layer for the
scheduler feature. Every test here is offline, uses a *tmp-file* database (so the
connection-per-call path is exercised and the store is thread-safe by
construction), and drives ``due()`` entirely by an injected ``now`` datetime so
the store never reads the wall clock.

Pinned behaviours:

* ``add`` returns a typed :class:`Trigger` with an assigned id; the schema init
  is idempotent (constructing twice over one file is safe).
* ``get`` returns a stored trigger or ``None`` when absent.
* ``list_triggers`` returns triggers in insertion (id) order.
* ``update`` persists ``next_run``/``last_run``/``enabled`` changes.
* ``set_enabled`` toggles the flag; ``delete`` removes a row.
* ``due(now)`` returns only enabled triggers whose ``next_run <= now`` — driven
  entirely by the passed datetime, never the wall clock.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from friday.scheduler.store import SQLiteTriggerStore, Trigger


def _store(tmp_path: Path) -> SQLiteTriggerStore:
    """A tmp-file store exercising the connection-per-call (thread-safe) path."""
    return SQLiteTriggerStore(str(tmp_path / "triggers.db"))


def _dt(iso: str) -> datetime:
    return datetime.fromisoformat(iso)


# --------------------------------------------------------------------------- #
# add + schema
# --------------------------------------------------------------------------- #
def test_add_returns_typed_trigger(tmp_path: Path) -> None:
    store = _store(tmp_path)
    t = store.add(
        name="nightly",
        kind="daily",
        spec="09:00",
        action="due_reminders",
        next_run="2026-06-16T09:00:00+00:00",
    )
    assert isinstance(t, Trigger)
    assert t.id >= 1
    assert t.name == "nightly"
    assert t.kind == "daily"
    assert t.spec == "09:00"
    assert t.action == "due_reminders"
    assert t.enabled is True
    assert t.next_run == "2026-06-16T09:00:00+00:00"
    assert t.last_run is None


def test_add_can_create_disabled(tmp_path: Path) -> None:
    store = _store(tmp_path)
    t = store.add(
        name="off", kind="interval", spec="60", action="noop", enabled=False
    )
    assert t.enabled is False


def test_schema_init_is_idempotent(tmp_path: Path) -> None:
    db = str(tmp_path / "triggers.db")
    first = SQLiteTriggerStore(db)
    first.add(name="persisted", kind="interval", spec="60", action="noop")
    # Re-opening the same file must not clobber existing rows.
    second = SQLiteTriggerStore(db)
    assert [t.name for t in second.list_triggers()] == ["persisted"]


# --------------------------------------------------------------------------- #
# get / list
# --------------------------------------------------------------------------- #
def test_get_returns_trigger_or_none(tmp_path: Path) -> None:
    store = _store(tmp_path)
    created = store.add(name="a", kind="interval", spec="30", action="noop")
    fetched = store.get(created.id)
    assert fetched is not None
    assert fetched.id == created.id
    assert fetched.name == "a"
    assert store.get(99999) is None


def test_list_triggers_in_insertion_order(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.add(name="first", kind="interval", spec="10", action="noop")
    store.add(name="second", kind="interval", spec="20", action="noop")
    assert [t.name for t in store.list_triggers()] == ["first", "second"]


# --------------------------------------------------------------------------- #
# update / set_enabled / delete
# --------------------------------------------------------------------------- #
def test_update_persists_run_fields(tmp_path: Path) -> None:
    store = _store(tmp_path)
    t = store.add(name="a", kind="interval", spec="60", action="noop")
    t.last_run = "2026-06-15T00:00:00+00:00"
    t.next_run = "2026-06-15T00:01:00+00:00"
    t.enabled = False
    store.update(t)

    reloaded = store.get(t.id)
    assert reloaded is not None
    assert reloaded.last_run == "2026-06-15T00:00:00+00:00"
    assert reloaded.next_run == "2026-06-15T00:01:00+00:00"
    assert reloaded.enabled is False


def test_set_enabled_toggles_flag(tmp_path: Path) -> None:
    store = _store(tmp_path)
    t = store.add(name="a", kind="interval", spec="60", action="noop")
    assert store.set_enabled(t.id, False) is True
    assert store.get(t.id).enabled is False  # type: ignore[union-attr]
    assert store.set_enabled(t.id, True) is True
    assert store.get(t.id).enabled is True  # type: ignore[union-attr]


def test_set_enabled_unknown_id_is_false(tmp_path: Path) -> None:
    store = _store(tmp_path)
    assert store.set_enabled(404, True) is False


def test_delete_removes_row(tmp_path: Path) -> None:
    store = _store(tmp_path)
    t = store.add(name="a", kind="interval", spec="60", action="noop")
    assert store.delete(t.id) == 1
    assert store.get(t.id) is None
    assert store.delete(t.id) == 0


# --------------------------------------------------------------------------- #
# due() — filters by injected now
# --------------------------------------------------------------------------- #
def test_due_filters_by_injected_now(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.add(
        name="past",
        kind="once",
        spec="2026-06-15T08:00:00+00:00",
        action="noop",
        next_run="2026-06-15T08:00:00+00:00",
    )
    store.add(
        name="exactly_now",
        kind="once",
        spec="2026-06-15T09:00:00+00:00",
        action="noop",
        next_run="2026-06-15T09:00:00+00:00",
    )
    store.add(
        name="future",
        kind="once",
        spec="2026-06-15T10:00:00+00:00",
        action="noop",
        next_run="2026-06-15T10:00:00+00:00",
    )

    now = _dt("2026-06-15T09:00:00+00:00")
    due_names = {t.name for t in store.due(now)}
    assert due_names == {"past", "exactly_now"}


def test_due_excludes_disabled_and_null_next_run(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.add(
        name="disabled",
        kind="once",
        spec="2026-06-15T08:00:00+00:00",
        action="noop",
        enabled=False,
        next_run="2026-06-15T08:00:00+00:00",
    )
    store.add(name="no_next_run", kind="once", spec="x", action="noop")

    now = _dt("2026-06-15T12:00:00+00:00")
    assert store.due(now) == []


def test_due_naive_now_against_utc_next_run(tmp_path: Path) -> None:
    """A naive ``now`` (e.g. utcnow) compares against stored UTC-offset rows."""
    store = _store(tmp_path)
    store.add(
        name="past",
        kind="once",
        spec="2026-06-15T08:00:00+00:00",
        action="noop",
        next_run="2026-06-15T08:00:00+00:00",
    )
    naive_now = datetime(2026, 6, 15, 9, 0, 0)
    assert {t.name for t in store.due(naive_now)} == {"past"}


def test_persists_across_reopen_file_path(tmp_path: Path) -> None:
    db = str(tmp_path / "triggers.db")
    SQLiteTriggerStore(db).add(
        name="kept",
        kind="daily",
        spec="07:30",
        action="noop",
        next_run=datetime(2026, 6, 16, 7, 30, tzinfo=UTC).isoformat(),
    )
    reopened = SQLiteTriggerStore(db)
    kept = reopened.list_triggers()
    assert len(kept) == 1
    assert kept[0].next_run == "2026-06-16T07:30:00+00:00"
