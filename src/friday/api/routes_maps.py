"""``/maps`` — the flagged Photorealistic-3D Maps surface (Tier 3; default off).

Three surfaces, all gated behind ``FRIDAY_ENABLE_MAPS`` (read lazily off
:func:`~friday.config.get_settings` so the router works mounted on a bare
``FastAPI()`` app, before ``app.py`` wiring exists); when the flag is off every
one of them is ``404`` so the feature simply does not exist for callers
(mirroring ``/studio`` and ``/reminders``):

* ``GET /maps`` — serves the no-build ``index.html`` (``FileResponse``) that
  loads the Google Maps JS API and renders a Photorealistic 3D globe.
* ``GET /maps/config`` — returns ``{"apiKey": <key or "">, "enabled": True}``.
  The page fetches this at runtime so **no key is ever baked into the HTML**.
* ``GET /maps/static/{filename}`` — serves the frontend assets (``maps.js``)
  with a strict no-traversal guard (only files inside ``static/`` are served).

The Google Maps API key is a :class:`~pydantic.SecretStr` on
:class:`~friday.config.Settings`; it is read via ``get_secret_value()`` ONLY to
return it to the same-origin page over ``/maps/config`` (the documented runtime
delivery for a browser Maps key) and is never logged.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse, JSONResponse

from friday.config import get_settings
from friday.logging import get_logger

logger = get_logger("friday.api.routes_maps")

router = APIRouter()

#: The no-build frontend asset directory for the maps surface.
STATIC_DIR = Path(__file__).resolve().parent.parent / "maps" / "static"
#: The maps single-page entrypoint served by ``GET /maps``.
INDEX_PATH = STATIC_DIR / "index.html"

#: Asset content types we serve from ``/maps/static`` (no executable fallthrough).
_MEDIA_TYPES = {
    ".html": "text/html",
    ".js": "text/javascript",
    ".css": "text/css",
    ".map": "application/json",
    ".json": "application/json",
}


def _maps_enabled() -> bool:
    """Whether the maps surface is enabled, read lazily from settings."""
    return bool(getattr(get_settings(), "enable_maps", False))


def _disabled() -> JSONResponse:
    """The canonical ``maps disabled`` 404 response."""
    return JSONResponse(status_code=404, content={"detail": "maps disabled"})


@router.get("/maps", response_model=None)
async def maps_index() -> FileResponse | JSONResponse:
    """Serve the maps ``index.html`` single-page app; 404 when disabled/missing."""
    if not _maps_enabled():
        return _disabled()
    if not INDEX_PATH.is_file():  # pragma: no cover - frontend asset always present
        return JSONResponse(
            status_code=404,
            content={"detail": "maps UI not found (frontend assets missing)"},
        )
    return FileResponse(INDEX_PATH, media_type="text/html")


@router.get("/maps/config", response_model=None)
async def maps_config() -> JSONResponse:
    """Return the runtime Maps config; 404 when disabled.

    The browser fetches this so the API key is delivered at runtime rather than
    baked into the served HTML. ``apiKey`` is ``""`` when no key is configured
    (the page degrades to a friendly notice rather than crashing).
    """
    if not _maps_enabled():
        return _disabled()
    key = get_settings().google_maps_api_key
    api_key = key.get_secret_value() if key is not None else ""
    return JSONResponse(status_code=200, content={"apiKey": api_key, "enabled": True})


@router.get("/maps/static/{filename:path}", response_model=None)
async def maps_static(filename: str) -> FileResponse | JSONResponse:
    """Serve a single asset from ``static/``; 404 when disabled/missing/escaping."""
    if not _maps_enabled():
        return _disabled()
    # Resolve and confine strictly inside STATIC_DIR (path-traversal guard).
    candidate = (STATIC_DIR / filename).resolve()
    try:
        candidate.relative_to(STATIC_DIR.resolve())
    except ValueError:
        return JSONResponse(status_code=404, content={"detail": "not found"})
    if not candidate.is_file():
        return JSONResponse(status_code=404, content={"detail": "not found"})
    media_type = _MEDIA_TYPES.get(candidate.suffix.lower(), "application/octet-stream")
    return FileResponse(candidate, media_type=media_type)
