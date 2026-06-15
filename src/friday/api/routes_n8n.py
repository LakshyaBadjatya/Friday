"""``/n8n`` — the flagged n8n-integration REST API (Tier 2).

Three surfaces, all gated behind ``FRIDAY_ENABLE_N8N`` (read off the startup
settings on ``app.state``); when the flag is off every one is ``404`` so the
feature simply does not exist for callers (mirroring ``/protocols`` / ``/rag`` /
``/studio``):

* ``GET  /n8n/status`` -> ``{"up": bool}`` — a liveness probe (never raises).
* ``POST /n8n/workflow`` ``{description, confirmed?}`` -> the
  :meth:`~friday.n8n.service.N8nService.make_workflow` result (a drafted/imported
  workflow, or a ``needs_confirmation`` to start n8n).
* ``POST /n8n/start`` ``{confirmed}`` -> runs the docker start behind the same
  confirm-gate (``confirmed=False`` -> ``needs_confirmation`` and NO subprocess).

The route reads the shared :class:`~friday.n8n.service.N8nService` off
``app.state`` (``app.py`` builds and stashes it when the flag is on).
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError

from friday.logging import get_logger
from friday.n8n.service import N8nService

logger = get_logger("friday.api.routes_n8n")

router = APIRouter()


class WorkflowRequest(BaseModel):
    """JSON body for ``POST /n8n/workflow``."""

    description: str = Field(min_length=1, max_length=4000)
    confirmed: bool = False


class StartRequest(BaseModel):
    """JSON body for ``POST /n8n/start``."""

    confirmed: bool = False


def _n8n_enabled(request: Request) -> bool:
    """Whether n8n is enabled, read off the startup settings on app state."""
    settings = getattr(request.app.state, "settings", None)
    return bool(getattr(settings, "enable_n8n", False))


def _disabled() -> JSONResponse:
    """The canonical ``n8n disabled`` 404 response."""
    return JSONResponse(status_code=404, content={"detail": "n8n disabled"})


def _get_service(request: Request) -> N8nService:
    """Pull the process-wide n8n service off ``app.state``."""
    service = getattr(request.app.state, "n8n_service", None)
    if not isinstance(service, N8nService):  # pragma: no cover - startup guard
        raise RuntimeError("n8n service is not initialized on app.state")
    return service


@router.get("/n8n/status", response_model=None)
async def n8n_status(request: Request) -> JSONResponse:
    """Report whether the n8n instance is reachable; 404 when disabled."""
    if not _n8n_enabled(request):
        return _disabled()
    service = _get_service(request)
    up = await service.client.is_up()
    return JSONResponse(status_code=200, content={"up": up})


@router.post("/n8n/workflow", response_model=None)
async def n8n_workflow(request: Request) -> JSONResponse:
    """Draft (and optionally import) a workflow; 404 when disabled, 422 on bad body."""
    if not _n8n_enabled(request):
        return _disabled()
    try:
        raw = await request.json()
    except (ValueError, UnicodeDecodeError):
        return JSONResponse(status_code=422, content={"detail": "expected a JSON body"})
    try:
        body = WorkflowRequest.model_validate(raw)
    except ValidationError as exc:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    service = _get_service(request)
    result = await service.make_workflow(
        body.description, confirmed=body.confirmed
    )
    return JSONResponse(status_code=200, content=result)


@router.post("/n8n/start", response_model=None)
async def n8n_start(request: Request) -> JSONResponse:
    """Start n8n via docker behind the confirm-gate; 404 when disabled.

    With ``confirmed=False`` (the default / a bad body) this returns a
    ``needs_confirmation`` payload and runs NO subprocess. With ``confirmed=True``
    it asks the service to start n8n (an empty description draft is skipped — the
    service only starts when n8n is down). The response reports whether a start
    was issued.
    """
    if not _n8n_enabled(request):
        return _disabled()

    confirmed = False
    try:
        raw = await request.json()
    except (ValueError, UnicodeDecodeError):
        raw = None
    if isinstance(raw, dict):
        try:
            confirmed = StartRequest.model_validate(raw).confirmed
        except ValidationError:
            confirmed = False

    service = _get_service(request)
    if not confirmed:
        return JSONResponse(
            status_code=200,
            content={
                "kind": "needs_confirmation",
                "action": "start_n8n",
                "message": "n8n isn't running; start it with docker?",
            },
        )
    started = await service.start(confirmed=True)
    return JSONResponse(
        status_code=200, content={"kind": "start", "started": started}
    )
