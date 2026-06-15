"""Unit tests for :class:`friday.protocols.store.SQLiteProtocolStore` (Tier 1).

The protocol store is the local-first, SQLite-backed durable layer for named
voice protocols. Every test here is offline, uses a *tmp-file* database (so the
connection-per-call path is exercised and the store is thread-safe by
construction), and steps are persisted as JSON.

Pinned behaviours:

* ``add`` returns a typed :class:`Protocol` with an assigned id; the schema init
  is idempotent (constructing twice over one file is safe).
* ``get`` / ``get_by_name`` round-trip the stored steps (JSON) losslessly;
  ``get_by_name`` is case-insensitive and returns ``None`` when absent.
* ``list_protocols`` returns protocols in insertion order.
* ``update`` rewrites name/trigger/steps/enabled; ``set_enabled`` toggles the flag.
* ``delete`` removes a row and returns the number removed (0/1).
"""

from __future__ import annotations

from pathlib import Path

from friday.protocols.store import Protocol, ProtocolStep, SQLiteProtocolStore


def _store(tmp_path: Path) -> SQLiteProtocolStore:
    """A tmp-file store (exercises the connection-per-call path)."""
    return SQLiteProtocolStore(str(tmp_path / "protocols.db"))


def _steps() -> list[ProtocolStep]:
    return [
        ProtocolStep(tool="list_reminders", args={"status": "open"}),
        ProtocolStep(tool="home", args={"device_id": "light.bedroom", "action": "off"}),
    ]


# --------------------------------------------------------------------------- #
# add + schema
# --------------------------------------------------------------------------- #
def test_add_returns_typed_protocol_with_id(tmp_path: Path) -> None:
    store = _store(tmp_path)
    p = store.add(
        name="Goodnight",
        trigger_phrase="goodnight",
        steps=_steps(),
    )
    assert isinstance(p, Protocol)
    assert p.id >= 1
    assert p.name == "Goodnight"
    assert p.trigger_phrase == "goodnight"
    assert p.enabled is True
    assert [s.tool for s in p.steps] == ["list_reminders", "home"]
    assert p.steps[1].args == {"device_id": "light.bedroom", "action": "off"}


def test_schema_init_is_idempotent(tmp_path: Path) -> None:
    db = str(tmp_path / "protocols.db")
    first = SQLiteProtocolStore(db)
    first.add(name="A", trigger_phrase="a", steps=[])
    # Re-constructing over the same file must not clobber existing data.
    second = SQLiteProtocolStore(db)
    assert [p.name for p in second.list_protocols()] == ["A"]


# --------------------------------------------------------------------------- #
# get / get_by_name — JSON round-trip
# --------------------------------------------------------------------------- #
def test_get_round_trips_steps_as_json(tmp_path: Path) -> None:
    store = _store(tmp_path)
    created = store.add(name="Goodnight", trigger_phrase="goodnight", steps=_steps())

    fetched = store.get(created.id)
    assert fetched is not None
    assert fetched.steps == _steps()


def test_get_missing_returns_none(tmp_path: Path) -> None:
    store = _store(tmp_path)
    assert store.get(999) is None


def test_get_by_name_is_case_insensitive(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.add(name="Goodnight", trigger_phrase="goodnight", steps=_steps())

    fetched = store.get_by_name("goodNIGHT")
    assert fetched is not None
    assert fetched.name == "Goodnight"


def test_get_by_name_missing_returns_none(tmp_path: Path) -> None:
    store = _store(tmp_path)
    assert store.get_by_name("nope") is None


# --------------------------------------------------------------------------- #
# list / update / set_enabled / delete
# --------------------------------------------------------------------------- #
def test_list_protocols_in_insertion_order(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.add(name="First", trigger_phrase="first", steps=[])
    store.add(name="Second", trigger_phrase="second", steps=[])
    assert [p.name for p in store.list_protocols()] == ["First", "Second"]


def test_update_rewrites_fields_and_steps(tmp_path: Path) -> None:
    store = _store(tmp_path)
    created = store.add(name="Goodnight", trigger_phrase="goodnight", steps=_steps())

    updated = created.model_copy(
        update={
            "name": "Bedtime",
            "trigger_phrase": "bedtime",
            "enabled": False,
            "steps": [ProtocolStep(tool="list_reminders", args={})],
        }
    )
    assert store.update(updated) is True

    fetched = store.get(created.id)
    assert fetched is not None
    assert fetched.name == "Bedtime"
    assert fetched.trigger_phrase == "bedtime"
    assert fetched.enabled is False
    assert [s.tool for s in fetched.steps] == ["list_reminders"]


def test_update_unknown_id_returns_false(tmp_path: Path) -> None:
    store = _store(tmp_path)
    ghost = Protocol(id=999, name="x", trigger_phrase="x", steps=[], enabled=True)
    assert store.update(ghost) is False


def test_set_enabled_toggles_flag(tmp_path: Path) -> None:
    store = _store(tmp_path)
    created = store.add(name="Goodnight", trigger_phrase="goodnight", steps=[])

    assert store.set_enabled(created.id, False) is True
    fetched = store.get(created.id)
    assert fetched is not None
    assert fetched.enabled is False

    assert store.set_enabled(999, True) is False


def test_delete_removes_row(tmp_path: Path) -> None:
    store = _store(tmp_path)
    created = store.add(name="Goodnight", trigger_phrase="goodnight", steps=[])

    assert store.delete(created.id) == 1
    assert store.get(created.id) is None
    # Idempotent: deleting again removes nothing.
    assert store.delete(created.id) == 0
