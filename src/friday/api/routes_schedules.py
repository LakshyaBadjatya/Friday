"""``/schedules`` — the flagged scheduled-triggers REST API (Tier 1).

Six surfaces, all gated behind ``FRIDAY_ENABLE_SCHEDULER`` (read off the startup
settings on ``app.state``); when the flag is off every one is ``404`` so the
feature simply does not exist for callers (mirroring ``/reminders`` / ``/rag`` /
``/studio``):

* ``POST   /schedules`` ``{name, kind, spec, action, enabled?}`` -> the created
  trigger (its initial ``next_run`` is computed from the spec).
* ``GET    /schedules`` -> ``{schedules, count}`` in insertion order.
* ``POST   /schedules/{id}/enable`` + ``/disable`` -> ``{id, enabled}`` (404 when
  no such trigger).
* ``POST   /schedules/{id}/run`` -> ``{id, ran}`` (fire the action now; 404 when
  no such trigger).
* ``DELETE /schedules/{id}`` -> ``{id, removed}`` (idempotent; ``removed`` 0/1).

The route reads the shared :class:`~friday.scheduler.store.SQLiteTriggerStore` and
:class:`~friday.scheduler.engine.Scheduler` off ``app.state`` (``app.py`` builds
and stashes them when the flag is on), so an HTTP-created trigger and the
background tick loop operate on the same store with the same registered actions.
"""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError

from friday.logging import get_logger
from friday.scheduler.engine import Scheduler, compute_next_run
from friday.scheduler.store import SQLiteTriggerStore, TriggerKind

logger = get_logger("friday.api.routes_schedules")

router = APIRouter()


class CreateScheduleRequest(BaseModel):
    """JSON body for ``POST /schedules``."""

    name: str = Field(min_length=1, max_length=200)
    kind: TriggerKind
    spec: str = Field(min_length=1, max_length=200)
    action: str = Field(min_length=1, max_length=200)
    enabled: bool = True


def _scheduler_enabled(request: Request) -> bool:
    """Whether the scheduler is enabled, read off the startup settings."""
    settings = getattr(request.app.state, "settings", None)
    return bool(getattr(settings, "enable_scheduler", False))


def _disabled() -> JSONResponse:
    """The canonical ``scheduler disabled`` 404 response."""
    return JSONResponse(status_code=404, content={"detail": "scheduler disabled"})


def _get_store(request: Request) -> SQLiteTriggerStore:
    """Pull the process-wide trigger store off ``app.state``."""
    store = getattr(request.app.state, "trigger_store", None)
    if not isinstance(store, SQLiteTriggerStore):  # pragma: no cover - startup guard
        raise RuntimeError("trigger store is not initialized on app.state")
    return store


def _get_scheduler(request: Request) -> Scheduler:
    """Pull the process-wide scheduler (action registry) off ``app.state``."""
    scheduler = getattr(request.app.state, "scheduler", None)
    if not isinstance(scheduler, Scheduler):  # pragma: no cover - startup guard
        raise RuntimeError("scheduler is not initialized on app.state")
    return scheduler


@router.post("/schedules", response_model=None)
async def create_schedule(request: Request) -> JSONResponse:
    """Create a trigger, computing its initial ``next_run``; 404 when disabled."""
    if not _scheduler_enabled(request):
        return _disabled()
    try:
        raw = await request.json()
    except (ValueError, UnicodeDecodeError):
        return JSONResponse(status_code=422, content={"detail": "expected a JSON body"})
    try:
        body = CreateScheduleRequest.model_validate(raw)
    except ValidationError as exc:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    store = _get_store(request)
    next_run = compute_next_run(body.kind, body.spec, datetime.now(UTC))
    trigger = store.add(
        name=body.name,
        kind=body.kind,
        spec=body.spec,
        action=body.action,
        enabled=body.enabled,
        next_run=None if next_run is None else next_run.isoformat(),
    )
    return JSONResponse(status_code=200, content=trigger.model_dump())


@router.get("/schedules", response_model=None)
async def list_schedules(request: Request) -> JSONResponse:
    """List triggers in insertion order; 404 when disabled."""
    if not _scheduler_enabled(request):
        return _disabled()
    store = _get_store(request)
    triggers = store.list_triggers()
    return JSONResponse(
        status_code=200,
        content={
            "schedules": [t.model_dump() for t in triggers],
            "count": len(triggers),
        },
    )


@router.post("/schedules/{trigger_id}/enable", response_model=None)
async def enable_schedule(request: Request, trigger_id: int) -> JSONResponse:
    """Enable a trigger by id; 404 when disabled or no such trigger."""
    return await _set_enabled(request, trigger_id, enabled=True)


@router.post("/schedules/{trigger_id}/disable", response_model=None)
async def disable_schedule(request: Request, trigger_id: int) -> JSONResponse:
    """Disable a trigger by id; 404 when disabled or no such trigger."""
    return await _set_enabled(request, trigger_id, enabled=False)


async def _set_enabled(
    request: Request, trigger_id: int, *, enabled: bool
) -> JSONResponse:
    if not _scheduler_enabled(request):
        return _disabled()
    store = _get_store(request)
    if not store.set_enabled(trigger_id, enabled):
        return JSONResponse(
            status_code=404,
            content={"detail": f"no schedule with id {trigger_id}"},
        )
    return JSONResponse(
        status_code=200, content={"id": trigger_id, "enabled": enabled}
    )


@router.post("/schedules/{trigger_id}/run", response_model=None)
async def run_schedule(request: Request, trigger_id: int) -> JSONResponse:
    """Fire a trigger's action now; 404 when disabled or no such trigger.

    Runs the registered action once (ignoring the trigger's ``next_run``/enabled
    state) without advancing the schedule, so an operator can test a trigger
    on demand. An unknown action name is reported as ``ran: False``.
    """
    if not _scheduler_enabled(request):
        return _disabled()
    store = _get_store(request)
    trigger = store.get(trigger_id)
    if trigger is None:
        return JSONResponse(
            status_code=404,
            content={"detail": f"no schedule with id {trigger_id}"},
        )
    scheduler = _get_scheduler(request)
    ran = await scheduler.run_action(trigger)
    return JSONResponse(status_code=200, content={"id": trigger_id, "ran": ran})


@router.delete("/schedules/{trigger_id}", response_model=None)
async def delete_schedule(request: Request, trigger_id: int) -> JSONResponse:
    """Delete a trigger by id; 404 when disabled. Idempotent (``removed`` 0/1)."""
    if not _scheduler_enabled(request):
        return _disabled()
    store = _get_store(request)
    removed = store.delete(trigger_id)
    return JSONResponse(
        status_code=200, content={"id": trigger_id, "removed": removed}
    )
