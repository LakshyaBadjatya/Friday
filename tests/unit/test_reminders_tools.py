"""Unit tests for the reminder tools (Tier 1).

The three reminder tools — :class:`CreateReminderTool`, :class:`ListRemindersTool`,
:class:`CompleteReminderTool` — wrap a :class:`~friday.reminders.store.SQLiteReminderStore`
behind the :class:`~friday.tools.base.Tool` contract so the Automation agent (and
the orchestrator) can drive reminders through the registry.

These tools write **local personal data only** (the same category as a memory
write), not an external/irreversible side-effect, so they are
``side_effecting=False`` — that keeps the registry confirm-step (build-spec §12)
from gating reminder creation behind an explicit confirmation. The tests pin both
the attributes and the behaviour, including a full round-trip through a real
:class:`~friday.tools.registry.ToolRegistry`.
"""

from __future__ import annotations

from pathlib import Path

from friday.reminders.store import SQLiteReminderStore
from friday.tools.base import Tool, ToolResult
from friday.tools.registry import ToolRegistry
from friday.tools.reminders import (
    CompleteReminderArgs,
    CompleteReminderTool,
    CreateReminderArgs,
    CreateReminderTool,
    ListRemindersArgs,
    ListRemindersTool,
)

ALLOWED = frozenset({"create_reminder", "list_reminders", "complete_reminder"})


def _store(tmp_path: Path) -> SQLiteReminderStore:
    return SQLiteReminderStore(str(tmp_path / "reminders.db"), clock=lambda: 1_000.0)


# --------------------------------------------------------------------------- #
# Attributes / protocol conformance
# --------------------------------------------------------------------------- #
def test_tools_satisfy_tool_protocol_and_are_not_side_effecting(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    create = CreateReminderTool(store)
    listing = ListRemindersTool(store)
    complete = CompleteReminderTool(store)

    for tool, name, model in (
        (create, "create_reminder", CreateReminderArgs),
        (listing, "list_reminders", ListRemindersArgs),
        (complete, "complete_reminder", CompleteReminderArgs),
    ):
        assert isinstance(tool, Tool)
        assert tool.name == name
        assert tool.args_model is model
        # Local personal data -> NOT side-effecting (no confirm-step friction).
        assert tool.side_effecting is False


# --------------------------------------------------------------------------- #
# CreateReminderTool
# --------------------------------------------------------------------------- #
async def test_create_reminder_persists_and_reports(tmp_path: Path) -> None:
    store = _store(tmp_path)
    tool = CreateReminderTool(store)

    result = await tool(
        CreateReminderArgs(
            text="call mom",
            due_at="2026-06-16T09:00:00+00:00",
            recurrence=None,
        )
    )
    assert isinstance(result, ToolResult)
    assert result.ok is True
    assert result.data["text"] == "call mom"
    assert result.data["due_at"] == "2026-06-16T09:00:00+00:00"
    assert result.data["status"] == "open"
    assert isinstance(result.data["id"], int)

    # The row really landed in the store.
    rows = store.list_reminders(status="all")
    assert [r.text for r in rows] == ["call mom"]


# --------------------------------------------------------------------------- #
# ListRemindersTool
# --------------------------------------------------------------------------- #
async def test_list_reminders_tool_returns_soonest_first(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.add("later", due_at="2026-06-20T00:00:00+00:00")
    store.add("sooner", due_at="2026-06-16T00:00:00+00:00")
    tool = ListRemindersTool(store)

    result = await tool(ListRemindersArgs(status="open"))
    assert result.ok is True
    texts = [item["text"] for item in result.data["reminders"]]
    assert texts == ["sooner", "later"]
    assert result.data["count"] == 2


async def test_list_reminders_tool_status_all_includes_done(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    r = store.add("done one", due_at="2026-06-16T00:00:00+00:00")
    store.complete(r.id)
    tool = ListRemindersTool(store)

    open_result = await tool(ListRemindersArgs(status="open"))
    assert open_result.data["count"] == 0
    all_result = await tool(ListRemindersArgs(status="all"))
    assert all_result.data["count"] == 1


# --------------------------------------------------------------------------- #
# CompleteReminderTool
# --------------------------------------------------------------------------- #
async def test_complete_reminder_tool_completes_existing(tmp_path: Path) -> None:
    store = _store(tmp_path)
    r = store.add("finish me", due_at="2026-06-16T00:00:00+00:00")
    tool = CompleteReminderTool(store)

    result = await tool(CompleteReminderArgs(id=r.id))
    assert result.ok is True
    assert result.data["completed"] is True
    assert store.list_reminders(status="open") == []


async def test_complete_reminder_tool_unknown_id_reports_not_found(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    tool = CompleteReminderTool(store)

    result = await tool(CompleteReminderArgs(id=4242))
    # A missing reminder is a handled failure, not an exception.
    assert result.ok is False
    assert result.error is not None
    assert result.error.code == "not_found"


# --------------------------------------------------------------------------- #
# Full round-trip through the registry
# --------------------------------------------------------------------------- #
async def test_reminder_tools_round_trip_via_registry(tmp_path: Path) -> None:
    store = _store(tmp_path)
    registry = ToolRegistry()
    from typing import cast

    registry.register(cast(Tool, CreateReminderTool(store)))
    registry.register(cast(Tool, ListRemindersTool(store)))
    registry.register(cast(Tool, CompleteReminderTool(store)))

    created = await registry.execute(
        "create_reminder",
        {"text": "via registry", "due_at": "2026-06-16T00:00:00+00:00"},
        allowed_tools=ALLOWED,
    )
    assert created.ok is True
    new_id = created.data["id"]

    listed = await registry.execute(
        "list_reminders", {"status": "open"}, allowed_tools=ALLOWED
    )
    assert listed.ok is True
    assert listed.data["count"] == 1

    completed = await registry.execute(
        "complete_reminder", {"id": new_id}, allowed_tools=ALLOWED
    )
    assert completed.ok is True
    assert completed.data["completed"] is True

    final = await registry.execute(
        "list_reminders", {"status": "open"}, allowed_tools=ALLOWED
    )
    assert final.data["count"] == 0


async def test_create_reminder_not_gated_by_confirm_step(tmp_path: Path) -> None:
    # Because the tool is side_effecting=False, the registry confirm-step does
    # NOT require ``confirmed=True`` — reminder creation succeeds on first call.
    store = _store(tmp_path)
    registry = ToolRegistry()
    from typing import cast

    registry.register(cast(Tool, CreateReminderTool(store)))

    result = await registry.execute(
        "create_reminder",
        {"text": "no confirm needed"},
        allowed_tools=ALLOWED,
        confirmed=False,
    )
    assert result.ok is True
    assert result.error is None
