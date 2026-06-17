"""``/hud`` — the flagged Tier-3 HUD "cockpit" surface (default off).

A no-build static cockpit page: an arc-reactor-style boot sequence animation
with particle/glow styling, plus a Cmd/Ctrl-K command palette that calls the
existing same-origin ``/chat`` and ``/admin`` endpoints. Three surfaces, all
gated behind ``FRIDAY_ENABLE_HUD`` (read lazily off
:func:`~friday.config.get_settings` so the router works mounted on a bare
``FastAPI()`` app, before ``app.py`` wiring exists); when the flag is off every
one of them is ``404`` so the feature simply does not exist for callers
(mirroring ``/maps`` and ``/studio``):

* ``GET /hud`` — serves the no-build ``index.html`` (``FileResponse``).
* ``GET /hud/static/{filename}`` — serves the frontend assets (``hud.js`` /
  ``hud.css``) with a strict no-traversal guard (only files inside ``static/``
  are served, and only known asset suffixes).

The page is pure vanilla JS/CSS/HTML — no bundler, no eval of any server output.
It carries no secrets: the command palette calls the existing endpoints over the
same origin, so authentication (if any) is the gateway's concern, not the HUD's.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse, JSONResponse

from friday.config import get_settings
from friday.logging import get_logger

logger = get_logger("friday.api.routes_hud")

router = APIRouter()

#: The no-build frontend asset directory for the HUD surface.
STATIC_DIR = Path(__file__).resolve().parent.parent / "hud" / "static"
#: The HUD single-page entrypoint served by ``GET /hud``.
INDEX_PATH = STATIC_DIR / "index.html"

#: Asset content types we serve from ``/hud/static`` (no executable fallthrough).
_MEDIA_TYPES = {
    ".html": "text/html",
    ".js": "text/javascript",
    ".css": "text/css",
    ".map": "application/json",
    ".json": "application/json",
}


def _hud_enabled() -> bool:
    """Whether the HUD surface is enabled, read lazily from settings."""
    return bool(getattr(get_settings(), "enable_hud", False))


def _disabled() -> JSONResponse:
    """The canonical ``hud disabled`` 404 response."""
    return JSONResponse(status_code=404, content={"detail": "hud disabled"})


#: The command-palette catalog the HUD offers (and a reusable surface contract:
#: the TUI and any other front-end can read the same list rather than hard-coding
#: it). Kept in sync with the ``COMMANDS`` array in ``hud.js``.
HUD_COMMANDS: tuple[dict[str, str], ...] = (
    {"id": "ask", "title": "Ask FRIDAY", "hint": "POST /chat"},
    {"id": "dossier", "title": "Open dossier", "hint": "POST /chat"},
    {"id": "roster", "title": "Show roster", "hint": "GET /roster"},
    {"id": "theme", "title": "Cycle theme", "hint": "appearance"},
    {"id": "view-command", "title": "Go to Command", "hint": "view"},
    {"id": "view-arena", "title": "Go to Arena", "hint": "view"},
    {"id": "view-agents", "title": "Go to Agents", "hint": "view"},
    {"id": "view-memory", "title": "Go to Memory", "hint": "view"},
    {"id": "view-system", "title": "Go to System", "hint": "view"},
)


@router.get("/hud", response_model=None)
async def hud_index() -> FileResponse | JSONResponse:
    """Serve the HUD ``index.html`` cockpit page; 404 when disabled/missing."""
    if not _hud_enabled():
        return _disabled()
    if not INDEX_PATH.is_file():  # pragma: no cover - frontend asset always present
        return JSONResponse(
            status_code=404,
            content={"detail": "hud UI not found (frontend assets missing)"},
        )
    return FileResponse(INDEX_PATH, media_type="text/html")


@router.get("/hud/commands", response_model=None)
async def hud_commands() -> JSONResponse:
    """Return the command-palette catalog (id/title/hint); 404 when HUD disabled."""
    if not _hud_enabled():
        return _disabled()
    return JSONResponse(status_code=200, content={"commands": list(HUD_COMMANDS)})


@router.get("/hud/static/{filename:path}", response_model=None)
async def hud_static(filename: str) -> FileResponse | JSONResponse:
    """Serve a single asset from ``static/``; 404 when disabled/missing/escaping."""
    if not _hud_enabled():
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
