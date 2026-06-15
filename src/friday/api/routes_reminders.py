"""``/reminders`` — the flagged reminders REST API (Tier 1).

Four surfaces, all gated behind ``FRIDAY_ENABLE_REMINDERS`` (read off the startup
settings on ``app.state``); when the flag is off every one is ``404`` so the
feature simply does not exist for callers (mirroring ``/rag`` and ``/studio``):

* ``POST   /reminders`` ``{text, due_at?, recurrence?}`` -> the created reminder.
* ``GET    /reminders?status=open|all`` -> ``{reminders, count}`` (soonest-due
  first).
* ``POST   /reminders/{id}/complete`` -> ``{id, completed}`` (404 if no such open
  reminder).
* ``DELETE /reminders/{id}`` -> ``{id, removed}`` (idempotent; ``removed`` is 0
  when the reminder was already gone).

The route reads the shared :class:`~friday.reminders.store.SQLiteReminderStore`
off ``app.state.reminder_store`` (``app.py`` builds and stashes it when the flag
is on), so an HTTP-created reminder and an Automation-agent-created one land in
the same store.
"""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError

from friday.logging import get_logger
from friday.reminders.store import SQLiteReminderStore

logger = get_logger("friday.api.routes_reminders")

router = APIRouter()


class CreateReminderRequest(BaseModel):
    """JSON body for ``POST /reminders``."""

    text: str = Field(min_length=1, max_length=2000)
    due_at: str | None = None
    recurrence: str | None = None


def _reminders_enabled(request: Request) -> bool:
    """Whether reminders are enabled, read off the startup settings on app state."""
    settings = getattr(request.app.state, "settings", None)
    return bool(getattr(settings, "enable_reminders", False))


def _disabled() -> JSONResponse:
    """The canonical ``reminders disabled`` 404 response."""
    return JSONResponse(status_code=404, content={"detail": "reminders disabled"})


def _get_store(request: Request) -> SQLiteReminderStore:
    """Pull the process-wide reminder store off ``app.state``."""
    store = getattr(request.app.state, "reminder_store", None)
    if not isinstance(store, SQLiteReminderStore):  # pragma: no cover - startup guard
        raise RuntimeError("reminder store is not initialized on app.state")
    return store


@router.post("/reminders", response_model=None)
async def create_reminder(request: Request) -> JSONResponse:
    """Create a reminder from a JSON body; 404 when disabled, 422 on bad body."""
    if not _reminders_enabled(request):
        return _disabled()
    try:
        raw = await request.json()
    except (ValueError, UnicodeDecodeError):
        return JSONResponse(status_code=422, content={"detail": "expected a JSON body"})
    try:
        body = CreateReminderRequest.model_validate(raw)
    except ValidationError as exc:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    store = _get_store(request)
    reminder = store.add(body.text, due_at=body.due_at, recurrence=body.recurrence)
    return JSONResponse(status_code=200, content=reminder.model_dump())


@router.get("/reminders", response_model=None)
async def list_reminders(
    request: Request, status: Literal["open", "all"] = "open"
) -> JSONResponse:
    """List reminders (``open`` by default), soonest-due first; 404 when disabled."""
    if not _reminders_enabled(request):
        return _disabled()
    store = _get_store(request)
    reminders = store.list_reminders(status=status)
    return JSONResponse(
        status_code=200,
        content={
            "reminders": [r.model_dump() for r in reminders],
            "count": len(reminders),
        },
    )


@router.post("/reminders/{reminder_id}/complete", response_model=None)
async def complete_reminder(request: Request, reminder_id: int) -> JSONResponse:
    """Complete a reminder by id; 404 when disabled or no such open reminder."""
    if not _reminders_enabled(request):
        return _disabled()
    store = _get_store(request)
    completed = store.complete(reminder_id)
    if not completed:
        return JSONResponse(
            status_code=404,
            content={"detail": f"no open reminder with id {reminder_id}"},
        )
    return JSONResponse(
        status_code=200, content={"id": reminder_id, "completed": True}
    )


@router.delete("/reminders/{reminder_id}", response_model=None)
async def delete_reminder(request: Request, reminder_id: int) -> JSONResponse:
    """Delete a reminder by id; 404 when disabled. Idempotent (``removed`` 0/1)."""
    if not _reminders_enabled(request):
        return _disabled()
    store = _get_store(request)
    removed = store.delete(reminder_id)
    return JSONResponse(
        status_code=200, content={"id": reminder_id, "removed": removed}
    )
