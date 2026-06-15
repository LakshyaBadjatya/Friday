"""Reminder tools: create / list / complete reminders via the tool registry.

Three tools wrap a :class:`~friday.reminders.store.SQLiteReminderStore` behind
the :class:`~friday.tools.base.Tool` contract so the Automation agent (and the
orchestrator) can manage reminders through the registry:

* :class:`CreateReminderTool` (``create_reminder``) — add a reminder.
* :class:`ListRemindersTool` (``list_reminders``) — list open or all reminders.
* :class:`CompleteReminderTool` (``complete_reminder``) — complete one by id.

**Why ``side_effecting=False``.** These tools write **local personal data only**
— the same category as a memory write — not an external, irreversible action
(unlike ``notify``/``home``, which reach the outside world). The registry's
confirm-step (build-spec §12) gates only side-effecting, non-idempotent tools, so
marking these as non-side-effecting keeps reminder creation friction-free: a user
asking the assistant to "remind me to call mom" should not be forced through a
"are you sure?" confirmation just to record a private note. Reversibility is
covered by :class:`CompleteReminderTool` / a delete, and there is no external blast
radius.

There is no LLM and no network in this path — only the local store — so callers
get deterministic, offline behaviour.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from pydantic import BaseModel, Field

from friday.reminders.store import ReminderStore
from friday.tools.base import ToolError, ToolResult

logger = logging.getLogger("friday.tools.reminders")


# --------------------------------------------------------------------------- #
# Argument models
# --------------------------------------------------------------------------- #
class CreateReminderArgs(BaseModel):
    """Arguments for :class:`CreateReminderTool`.

    ``text`` is the reminder body. ``due_at`` is an optional ISO-8601 timestamp
    (omit for an undated "remember to" note). ``recurrence`` is an optional
    ``"daily"``/``"weekly"`` keyword that rolls the reminder forward on
    completion.
    """

    text: str = Field(min_length=1)
    due_at: str | None = None
    recurrence: str | None = None


class ListRemindersArgs(BaseModel):
    """Arguments for :class:`ListRemindersTool` (``"open"`` by default)."""

    status: Literal["open", "all"] = "open"


class CompleteReminderArgs(BaseModel):
    """Arguments for :class:`CompleteReminderTool` — the reminder id to complete."""

    id: int


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #
class CreateReminderTool:
    """Create a reminder in the local store and report the stored row."""

    name = "create_reminder"
    description = (
        "Create a personal reminder/task with optional due date "
        "(ISO-8601) and recurrence (daily|weekly)."
    )
    args_model = CreateReminderArgs
    required_permission = "reminders"
    idempotent = False
    side_effecting = False

    def __init__(self, store: ReminderStore) -> None:
        self._store = store

    async def __call__(self, args: Any) -> ToolResult:
        if not isinstance(args, CreateReminderArgs):
            args = CreateReminderArgs.model_validate(args)
        reminder = self._store.add(
            args.text, due_at=args.due_at, recurrence=args.recurrence
        )
        logger.info("reminder created id=%s due_at=%s", reminder.id, reminder.due_at)
        return ToolResult(ok=True, data=reminder.model_dump(), error=None)


class ListRemindersTool:
    """List reminders (open or all), soonest-due first."""

    name = "list_reminders"
    description = (
        "List personal reminders, soonest-due first. "
        "status=open (default) or all (includes completed)."
    )
    args_model = ListRemindersArgs
    required_permission = "reminders"
    idempotent = True
    side_effecting = False

    def __init__(self, store: ReminderStore) -> None:
        self._store = store

    async def __call__(self, args: Any) -> ToolResult:
        if not isinstance(args, ListRemindersArgs):
            args = ListRemindersArgs.model_validate(args)
        reminders = self._store.list_reminders(status=args.status)
        data: dict[str, Any] = {
            "reminders": [r.model_dump() for r in reminders],
            "count": len(reminders),
        }
        return ToolResult(ok=True, data=data, error=None)


class CompleteReminderTool:
    """Complete a reminder by id (recurring reminders roll forward instead)."""

    name = "complete_reminder"
    description = (
        "Mark a personal reminder complete by its id. "
        "A recurring reminder rolls its due date forward and stays open."
    )
    args_model = CompleteReminderArgs
    required_permission = "reminders"
    idempotent = False
    side_effecting = False

    def __init__(self, store: ReminderStore) -> None:
        self._store = store

    async def __call__(self, args: Any) -> ToolResult:
        if not isinstance(args, CompleteReminderArgs):
            args = CompleteReminderArgs.model_validate(args)
        completed = self._store.complete(args.id)
        if not completed:
            return ToolResult(
                ok=False,
                data={"id": args.id, "completed": False},
                error=ToolError(
                    code="not_found",
                    message=f"no open reminder with id {args.id}",
                    retriable=False,
                ),
            )
        logger.info("reminder completed id=%s", args.id)
        return ToolResult(
            ok=True, data={"id": args.id, "completed": True}, error=None
        )
