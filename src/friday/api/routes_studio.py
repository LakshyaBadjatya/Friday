"""``/studio`` — the 3D Studio API + static UI mount (Phase 7, Stage 1).

Three surfaces, all gated behind ``FRIDAY_ENABLE_STUDIO`` (read off the startup
settings on ``app.state``); when the flag is off every one of them is ``404`` so
the feature simply does not exist for callers:

* ``POST /studio/generate`` — ``{description, quality}`` -> a JSON envelope:
  ``{"kind": "scene", "scene": {...}}`` for the free procedural path or
  ``{"kind": "mesh", ...}`` for an available hi-fi provider. The
  :class:`~friday.studio.generator.StudioService` (off ``app.state.studio``)
  owns the procedural/hi-fi/fallback policy, so the route stays thin.
* ``GET  /studio`` — serves the no-build Three.js ``index.html`` (FileResponse).
* ``/studio/static`` — the StaticFiles mount for the frontend assets.

The static directory (``src/friday/studio/static``) is produced by the Stage-2
frontend agent in parallel; this module only references the path. The router is
included unconditionally but self-guards on the flag (mirroring ``/voice``); the
StaticFiles mount is added by :func:`~friday.app.create_app` only when enabled.

The LLM output that drives this route is validated JSON (a
:class:`~friday.studio.scene.Scene`), never JavaScript — there is no eval of
model output anywhere on this path.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from friday.errors import FridayError, ProviderError
from friday.logging import get_logger
from friday.studio.generator import Quality, StudioService

logger = get_logger("friday.api.routes_studio")

router = APIRouter()

#: The no-build frontend asset directory (the Stage-2 agent creates files here).
STATIC_DIR = Path(__file__).resolve().parent.parent / "studio" / "static"
#: The studio single-page entrypoint served by ``GET /studio``.
INDEX_PATH = STATIC_DIR / "index.html"


class StudioGenerateRequest(BaseModel):
    """JSON body for ``POST /studio/generate``."""

    description: str = Field(min_length=1, max_length=2000)
    quality: Quality = "fast"


def _studio_enabled(request: Request) -> bool:
    """Whether the studio is enabled, read off the startup settings on app state."""
    settings = getattr(request.app.state, "settings", None)
    return bool(getattr(settings, "enable_studio", False))


def _disabled() -> JSONResponse:
    """The canonical ``studio disabled`` 404 response."""
    return JSONResponse(status_code=404, content={"detail": "studio disabled"})


def _get_service(request: Request) -> StudioService:
    """Pull the process-wide :class:`StudioService` off ``app.state`` (startup)."""
    service = getattr(request.app.state, "studio", None)
    if not isinstance(service, StudioService):  # pragma: no cover - startup guard
        raise RuntimeError("studio service is not initialized on app.state")
    return service


@router.post("/studio/generate", response_model=None)
async def studio_generate(request: Request) -> JSONResponse:
    """Generate a scene (or mesh) for a description; 404 when studio is disabled.

    The body is validated (``description`` non-empty, ``quality`` in
    ``{fast, hifi}``); a malformed body is mapped to ``422``. A raised
    :class:`~friday.errors.FridayError` (e.g. the procedural generator failing to
    produce valid JSON after its bounded repair) becomes a clean ``502`` JSON
    error rather than a leaked 500.
    """
    if not _studio_enabled(request):
        return _disabled()

    try:
        body = await request.json()
    except (ValueError, UnicodeDecodeError):
        return JSONResponse(status_code=422, content={"detail": "expected a JSON body"})

    try:
        parsed = StudioGenerateRequest.model_validate(body)
    except ValueError as exc:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    service = _get_service(request)
    try:
        result = await service.generate(parsed.description, quality=parsed.quality)
    except FridayError as exc:
        status = 502 if isinstance(exc, ProviderError) else 400
        logger.warning(
            "studio generation raised FridayError",
            extra={"error_type": type(exc).__name__, "status": status},
        )
        return JSONResponse(
            status_code=status,
            content={"error": str(exc), "type": type(exc).__name__},
        )

    return JSONResponse(status_code=200, content=result)


@router.get("/studio", response_model=None)
async def studio_index(request: Request) -> FileResponse | JSONResponse:
    """Serve the studio ``index.html`` single-page app; 404 when disabled/missing."""
    if not _studio_enabled(request):
        return _disabled()
    if not INDEX_PATH.is_file():
        return JSONResponse(
            status_code=404,
            content={"detail": "studio UI not found (frontend assets missing)"},
        )
    return FileResponse(INDEX_PATH, media_type="text/html")
