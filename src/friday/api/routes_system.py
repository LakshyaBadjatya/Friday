"""``/system`` — the flagged hardware/system-monitor API (Tier 2).

Two read-only surfaces, gated behind ``FRIDAY_ENABLE_SYSTEM_MONITOR`` (read off
the startup settings on ``app.state``); when the flag is off both are ``404`` so
the feature simply does not exist for callers (mirroring ``/briefing`` /
``/study``):

* ``GET /system/stats`` -> the current :class:`~friday.system.monitor.SystemStats`
  snapshot (cpu/mem/disk percentages plus optional temperature + load + ts).
* ``GET /system/check`` -> ``{alerts, count}`` — the active
  :class:`~friday.system.monitor.Alert` list (one per breached threshold, empty
  when healthy).

The route reads the shared :class:`~friday.system.monitor.SystemMonitor` off
``app.state.system_monitor`` (``app.py`` builds and stashes it when the flag is
on, over the real :class:`~friday.system.monitor.PsutilSampler`), so the HTTP
surface and the scheduler ``"system_check"`` action read the *same* monitor.
Sampling happens at request time; tests inject a monitor built over a fake
sampler so they stay deterministic and never touch ``psutil``.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from friday.logging import get_logger
from friday.system.monitor import SystemMonitor

logger = get_logger("friday.api.routes_system")

router = APIRouter()


def _system_enabled(request: Request) -> bool:
    """Whether the system monitor is enabled, read off the startup settings."""
    settings = getattr(request.app.state, "settings", None)
    return bool(getattr(settings, "enable_system_monitor", False))


def _disabled() -> JSONResponse:
    """The canonical ``system monitor disabled`` 404 response."""
    return JSONResponse(
        status_code=404, content={"detail": "system monitor disabled"}
    )


def _get_monitor(request: Request) -> SystemMonitor:
    """Pull the process-wide :class:`SystemMonitor` off ``app.state`` (startup)."""
    monitor = getattr(request.app.state, "system_monitor", None)
    if not isinstance(monitor, SystemMonitor):  # pragma: no cover - startup guard
        raise RuntimeError("system monitor is not initialized on app.state")
    return monitor


@router.get("/system/stats", response_model=None)
async def get_stats(request: Request) -> JSONResponse:
    """Return the current resource snapshot; 404 when disabled."""
    if not _system_enabled(request):
        return _disabled()
    monitor = _get_monitor(request)
    stats = monitor.stats()
    return JSONResponse(status_code=200, content=stats.model_dump())


@router.get("/system/check", response_model=None)
async def get_check(request: Request) -> JSONResponse:
    """Return the active threshold alerts (``[]`` when healthy); 404 when disabled."""
    if not _system_enabled(request):
        return _disabled()
    monitor = _get_monitor(request)
    alerts = monitor.check()
    return JSONResponse(
        status_code=200,
        content={"alerts": [a.model_dump() for a in alerts], "count": len(alerts)},
    )
