"""``/presence`` — the flagged BLE presence API (Tier 3).

One read-only surface, gated behind ``FRIDAY_ENABLE_PRESENCE`` (read from
:func:`~friday.config.get_settings`); when the flag is off it is ``404`` so the
feature simply does not exist for callers (mirroring ``/system`` / ``/study``):

* ``GET /presence`` -> the current present/absent/arrived/departed split of the
  *known* devices (the ``FRIDAY_PRESENCE_KNOWN_DEVICES`` ``MAC=Name`` map).

The known-device map and the flag are read off ``get_settings()`` at request
time (NOT ``app.state``), so this router works mounted on a bare ``FastAPI()``
app with settings monkeypatched — no ``create_app`` wiring is required for it to
serve. The BLE scanner is built **lazily inside the handler**: a process-wide one
stashed on ``app.state.presence_scanner`` is used if present (e.g. a real
:class:`~friday.presence.scanner.BleakScanner` wired at startup, or a test fake),
otherwise a stateless :class:`~friday.presence.scanner.FakePresenceScanner` that
sees nothing — so an unconfigured but enabled build answers (everyone absent)
rather than touching Bluetooth.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from friday.config import Settings, get_settings
from friday.logging import get_logger
from friday.presence.scanner import FakePresenceScanner, PresenceScanner
from friday.presence.service import PresenceService

logger = get_logger("friday.api.routes_presence")

router = APIRouter()


def _disabled() -> JSONResponse:
    """The canonical ``presence disabled`` 404 response."""
    return JSONResponse(status_code=404, content={"detail": "presence disabled"})


def _parse_known(entries: list[str]) -> dict[str, str]:
    """Parse ``["MAC=Name", ...]`` (from settings) into a ``{MAC: Name}`` map.

    Each entry is split on the first ``=`` only (a name may legitimately contain
    further ``=``); an entry without ``=`` or with an empty MAC is skipped rather
    than raising, so a stray config value degrades cleanly.
    """
    known: dict[str, str] = {}
    for entry in entries:
        mac, sep, name = entry.partition("=")
        mac = mac.strip()
        name = name.strip()
        if sep and mac and name:
            known[mac] = name
    return known


def _get_scanner(request: Request) -> PresenceScanner:
    """Use the configured scanner on ``app.state`` if any, else a stateless fake.

    The fake sees nothing (empty script), so an enabled-but-unconfigured build
    answers with everyone absent rather than reaching for Bluetooth hardware.
    """
    scanner = getattr(request.app.state, "presence_scanner", None)
    if isinstance(scanner, PresenceScanner):
        return scanner
    return FakePresenceScanner(scans=[])


@router.get("/presence", response_model=None)
async def get_presence(request: Request) -> JSONResponse:
    """Return the current presence split; 404 when presence is disabled."""
    settings: Settings = get_settings()
    if not settings.enable_presence:
        return _disabled()

    known = _parse_known(settings.presence_known_devices)
    scanner = _get_scanner(request)
    service = PresenceService(scanner, known)
    update = await service.update()
    return JSONResponse(status_code=200, content=update.model_dump())
