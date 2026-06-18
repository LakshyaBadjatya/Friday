# © Lakshya Badjatya — Author
"""Unit tests for :class:`friday.flows.store.SQLiteFlowStore`.

Mirrors the scheduler store's contract: local-first SQLite, parametrized SQL,
durable across reopen (the checkpoint), and a ``resumable()`` view for crash
recovery (only ``running``/``paused`` flows).
"""

from __future__ import annotations

from pathlib import Path

from friday.flows.models import Flow, FlowStatus, FlowStep
from friday.flows.store import SQLiteFlowStore


def _flow(fid: str = "f1") -> Flow:
    return Flow(id=fid, goal="g", steps=[FlowStep(id="s1", description="x")])


def test_create_get_roundtrip() -> None:
    store = SQLiteFlowStore()
    store.create(_flow())
    got = store.get("f1")
    assert got is not None
    assert got.goal == "g"
    assert got.steps[0].id == "s1"


def test_get_missing_returns_none() -> None:
    assert SQLiteFlowStore().get("nope") is None


def test_update_persists_status_and_checkpoint(tmp_path: Path) -> None:
    path = str(tmp_path / "flows.db")
    store = SQLiteFlowStore(path)
    flow = _flow()
    store.create(flow)
    flow.status = FlowStatus.RUNNING
    flow.cursor = 1
    assert store.update(flow) is True

    reopened = SQLiteFlowStore(path)  # survives a process restart
    got = reopened.get("f1")
    assert got is not None
    assert got.status == FlowStatus.RUNNING
    assert got.cursor == 1


def test_update_missing_returns_false() -> None:
    assert SQLiteFlowStore().update(_flow("ghost")) is False


def test_list_and_resumable() -> None:
    store = SQLiteFlowStore()
    running = _flow("a")
    running.status = FlowStatus.RUNNING
    store.create(running)
    paused = _flow("b")
    paused.status = FlowStatus.PAUSED
    store.create(paused)
    done = _flow("c")
    done.status = FlowStatus.SUCCEEDED
    store.create(done)

    assert {f.id for f in store.list_flows()} == {"a", "b", "c"}
    assert {f.id for f in store.list_flows(status=FlowStatus.SUCCEEDED)} == {"c"}
    assert {f.id for f in store.resumable()} == {"a", "b"}


def test_delete() -> None:
    store = SQLiteFlowStore()
    store.create(_flow())
    assert store.delete("f1") == 1
    assert store.get("f1") is None
    assert store.delete("f1") == 0
