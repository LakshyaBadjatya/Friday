"""``/plugins`` — the flagged plugin-inventory API (Tier 2).

A single read-only surface, gated behind ``FRIDAY_ENABLE_PLUGINS`` (read off the
startup settings on ``app.state``); when the flag is off it is ``404`` so the
feature simply does not exist for callers (mirroring ``/reminders`` /
``/protocols`` / ``/rag`` / ``/studio``):

* ``GET /plugins`` -> the list of :class:`~friday.plugins.loader.PluginInfo`
  recorded at startup (each ``{name, path, tools, error}``), so the owner can see
  which plugins loaded, which tools they contributed, and which failed (and why).

The route reads the :class:`PluginInfo` list off ``app.state.plugins`` —
``app.py`` runs :func:`~friday.plugins.loader.load_into` once at startup (after
the built-in tools are registered) and stashes the result there. The endpoint
introduces no execution path of its own; it only reports what loading produced.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from friday.logging import get_logger
from friday.plugins.loader import PluginInfo

logger = get_logger("friday.api.routes_plugins")

router = APIRouter()


def _plugins_enabled(request: Request) -> bool:
    """Whether plugins are enabled, read off the startup settings on app state."""
    settings = getattr(request.app.state, "settings", None)
    return bool(getattr(settings, "enable_plugins", False))


def _disabled() -> JSONResponse:
    """The canonical ``plugins disabled`` 404 response."""
    return JSONResponse(status_code=404, content={"detail": "plugins disabled"})


def _get_plugins(request: Request) -> list[PluginInfo]:
    """Pull the startup-recorded :class:`PluginInfo` list off ``app.state``.

    ``app.py`` stashes the :func:`load_into` result on ``app.state.plugins`` when
    the flag is on; default to an empty list so an enabled app with no plugins
    dir still returns ``200`` with ``[]`` rather than erroring.
    """
    plugins = getattr(request.app.state, "plugins", None)
    if not isinstance(plugins, list):
        return []
    return plugins


@router.get("/plugins", response_model=None)
async def list_plugins(request: Request) -> JSONResponse:
    """List the plugins recorded at startup; 404 when disabled."""
    if not _plugins_enabled(request):
        return _disabled()
    plugins = _get_plugins(request)
    return JSONResponse(
        status_code=200, content=[info.model_dump() for info in plugins]
    )
