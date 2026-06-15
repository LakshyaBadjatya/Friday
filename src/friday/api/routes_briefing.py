"""``/briefing`` — the flagged proactive-briefing API (Tier 1).

A single surface, gated behind ``FRIDAY_ENABLE_BRIEFING`` (read off the startup
settings on ``app.state``); when the flag is off it is ``404`` so the feature
simply does not exist for callers (mirroring ``/reminders`` / ``/rag`` /
``/studio``):

* ``GET /briefing`` -> the current :class:`~friday.briefing.service.Briefing`,
  built for ``utcnow``.

The route reads the shared :class:`~friday.briefing.service.BriefingService` off
``app.state.briefing`` (``app.py`` builds and stashes it when the flag is on),
so the HTTP briefing and the scheduler ``"briefing"`` action assemble from the
same local stores. ``utcnow`` is read here at request time only — the tested
``BriefingService.build(now)`` unit stays clock-injected; this endpoint is the
"give me one right now" seam.
"""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from friday.briefing.service import BriefingService
from friday.logging import get_logger

logger = get_logger("friday.api.routes_briefing")

router = APIRouter()


def _briefing_enabled(request: Request) -> bool:
    """Whether briefing is enabled, read off the startup settings on app state."""
    settings = getattr(request.app.state, "settings", None)
    return bool(getattr(settings, "enable_briefing", False))


def _disabled() -> JSONResponse:
    """The canonical ``briefing disabled`` 404 response."""
    return JSONResponse(status_code=404, content={"detail": "briefing disabled"})


def _get_service(request: Request) -> BriefingService:
    """Pull the process-wide :class:`BriefingService` off ``app.state`` (startup)."""
    service = getattr(request.app.state, "briefing", None)
    if not isinstance(service, BriefingService):  # pragma: no cover - startup guard
        raise RuntimeError("briefing service is not initialized on app.state")
    return service


@router.get("/briefing", response_model=None)
async def get_briefing(request: Request) -> JSONResponse:
    """Build and return the current briefing (utcnow); 404 when disabled."""
    if not _briefing_enabled(request):
        return _disabled()
    service = _get_service(request)
    briefing = await service.build(datetime.now(UTC))
    return JSONResponse(status_code=200, content=briefing.model_dump())
