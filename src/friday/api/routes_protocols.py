"""``/protocols`` — the flagged voice-protocols REST API (Tier 1).

Four surfaces, all gated behind ``FRIDAY_ENABLE_PROTOCOLS`` (read off the startup
settings on ``app.state``); when the flag is off every one is ``404`` so the
feature simply does not exist for callers (mirroring ``/reminders`` / ``/schedules``
/ ``/rag`` / ``/studio``):

* ``POST   /protocols`` ``{name, trigger_phrase, steps}`` -> the created protocol.
* ``GET    /protocols`` -> ``{protocols, count}`` in insertion order.
* ``POST   /protocols/{id}/run`` ``{confirmed?}`` -> the :class:`ProtocolResult`
  (404 when no such protocol). Runs the steps through the shared registry,
  honoring the confirm-step on any side-effecting step.
* ``DELETE /protocols/{id}`` -> ``{id, removed}`` (idempotent; ``removed`` 0/1).

The route reads the shared :class:`~friday.protocols.store.SQLiteProtocolStore` and
:class:`~friday.protocols.runner.ProtocolRunner` off ``app.state`` (``app.py`` builds
and stashes them when the flag is on), so an HTTP-created protocol and the
orchestrator's trigger-phrase hook operate on the same store with the same runner.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError

from friday.logging import get_logger
from friday.observability.audit import AuditLog
from friday.protocols.learn import has_redacted_args, learn_protocol
from friday.protocols.runner import ProtocolRunner
from friday.protocols.store import ProtocolStep, SQLiteProtocolStore

logger = get_logger("friday.api.routes_protocols")

router = APIRouter()


class CreateProtocolRequest(BaseModel):
    """JSON body for ``POST /protocols``."""

    name: str = Field(min_length=1, max_length=200)
    trigger_phrase: str = Field(min_length=1, max_length=400)
    steps: list[ProtocolStep] = Field(default_factory=list)


class LearnProtocolRequest(BaseModel):
    """JSON body for ``POST /protocols/learn`` — learn a macro from recent actions."""

    name: str = Field(min_length=1, max_length=200)
    trigger_phrase: str = Field(min_length=1, max_length=400)
    only_successful: bool = True
    include_tools: list[str] = Field(default_factory=list)


class RunProtocolRequest(BaseModel):
    """JSON body for ``POST /protocols/{id}/run`` (all fields optional)."""

    confirmed: bool = False


def _protocols_enabled(request: Request) -> bool:
    """Whether protocols are enabled, read off the startup settings on app state."""
    settings = getattr(request.app.state, "settings", None)
    return bool(getattr(settings, "enable_protocols", False))


def _disabled() -> JSONResponse:
    """The canonical ``protocols disabled`` 404 response."""
    return JSONResponse(status_code=404, content={"detail": "protocols disabled"})


def _get_store(request: Request) -> SQLiteProtocolStore:
    """Pull the process-wide protocol store off ``app.state``."""
    store = getattr(request.app.state, "protocol_store", None)
    if not isinstance(store, SQLiteProtocolStore):  # pragma: no cover - startup guard
        raise RuntimeError("protocol store is not initialized on app.state")
    return store


def _get_runner(request: Request) -> ProtocolRunner:
    """Pull the process-wide protocol runner off ``app.state``."""
    runner = getattr(request.app.state, "protocol_runner", None)
    if not isinstance(runner, ProtocolRunner):  # pragma: no cover - startup guard
        raise RuntimeError("protocol runner is not initialized on app.state")
    return runner


@router.post("/protocols", response_model=None)
async def create_protocol(request: Request) -> JSONResponse:
    """Create a protocol from a JSON body; 404 when disabled, 422 on bad body."""
    if not _protocols_enabled(request):
        return _disabled()
    try:
        raw = await request.json()
    except (ValueError, UnicodeDecodeError):
        return JSONResponse(status_code=422, content={"detail": "expected a JSON body"})
    try:
        body = CreateProtocolRequest.model_validate(raw)
    except ValidationError as exc:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    store = _get_store(request)
    protocol = store.add(
        name=body.name,
        trigger_phrase=body.trigger_phrase,
        steps=body.steps,
    )
    return JSONResponse(status_code=200, content=protocol.model_dump())


@router.post("/protocols/learn", response_model=None)
async def learn_protocol_route(request: Request) -> JSONResponse:
    """Learn a draft (disabled) protocol from the recent tool-call audit.

    Folds the recent audited tool-calls into a named protocol via
    :func:`~friday.protocols.learn.learn_protocol` and persists it DISABLED, so the
    owner reviews (and any redacted secret arg is filled in) before it can fire.
    404 when protocols are off; 400 when there is nothing to learn from.
    """
    if not _protocols_enabled(request):
        return _disabled()
    try:
        raw = await request.json()
    except (ValueError, UnicodeDecodeError):
        return JSONResponse(status_code=422, content={"detail": "expected a JSON body"})
    try:
        body = LearnProtocolRequest.model_validate(raw)
    except ValidationError as exc:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    audit = getattr(request.app.state, "audit", None)
    calls = audit.recent() if isinstance(audit, AuditLog) else []
    try:
        draft = learn_protocol(
            body.name,
            body.trigger_phrase,
            calls,
            only_successful=body.only_successful,
            include_tools=body.include_tools or None,
        )
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    saved = _get_store(request).add(
        name=draft.name,
        trigger_phrase=draft.trigger_phrase,
        steps=draft.steps,
        enabled=False,
    )
    return JSONResponse(
        status_code=200,
        content={
            "protocol": saved.model_dump(),
            "has_redacted_args": has_redacted_args(saved),
        },
    )


@router.get("/protocols", response_model=None)
async def list_protocols(request: Request) -> JSONResponse:
    """List protocols in insertion order; 404 when disabled."""
    if not _protocols_enabled(request):
        return _disabled()
    store = _get_store(request)
    protocols = store.list_protocols()
    return JSONResponse(
        status_code=200,
        content={
            "protocols": [p.model_dump() for p in protocols],
            "count": len(protocols),
        },
    )


@router.post("/protocols/{protocol_id}/run", response_model=None)
async def run_protocol(request: Request, protocol_id: int) -> JSONResponse:
    """Run a protocol's steps now; 404 when disabled or no such protocol.

    The optional ``{confirmed}`` body is threaded into the runner so a confirming
    re-run executes any side-effecting steps that paused on the confirm-step. A
    bad body is treated as the default (unconfirmed) rather than a 422.
    """
    if not _protocols_enabled(request):
        return _disabled()
    store = _get_store(request)
    protocol = store.get(protocol_id)
    if protocol is None:
        return JSONResponse(
            status_code=404,
            content={"detail": f"no protocol with id {protocol_id}"},
        )

    confirmed = False
    try:
        raw = await request.json()
    except (ValueError, UnicodeDecodeError):
        raw = None
    if isinstance(raw, dict):
        try:
            confirmed = RunProtocolRequest.model_validate(raw).confirmed
        except ValidationError:
            confirmed = False

    runner = _get_runner(request)
    result = await runner.run(protocol, confirmed=confirmed)
    return JSONResponse(status_code=200, content=result.model_dump())


@router.delete("/protocols/{protocol_id}", response_model=None)
async def delete_protocol(request: Request, protocol_id: int) -> JSONResponse:
    """Delete a protocol by id; 404 when disabled. Idempotent (``removed`` 0/1)."""
    if not _protocols_enabled(request):
        return _disabled()
    store = _get_store(request)
    removed = store.delete(protocol_id)
    return JSONResponse(
        status_code=200, content={"id": protocol_id, "removed": removed}
    )
