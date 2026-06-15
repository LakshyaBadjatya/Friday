"""The always-available PWA shell — makes FRIDAY installable + offline-capable.

Three static surfaces, all at ROOT scope (no ``/pwa`` prefix) and with NO feature
flag — installing a Progressive Web App is harmless and the shell carries no
secrets, so it is always reachable (mirroring a service worker's need to live at
the root to control the whole origin):

* ``GET /manifest.webmanifest`` — the web app manifest (``application/manifest+json``)
  declaring the installable ``FRIDAY`` app: ``display: standalone``, ``start_url:
  /hud``. No ``friday-orb`` icon ships in the tree, so ``icons`` is an empty list
  rather than a reference to a missing file.
* ``GET /service-worker.js`` — the service worker (``text/javascript``). Served at
  the ROOT path so its default control scope is the whole origin: it precaches a
  tiny app shell and serves the cached ``/offline.html`` when a navigation fails
  offline. It MUST be served from the root (not ``/pwa/static/...``) for the scope
  to cover the dashboard/HUD.
* ``GET /offline.html`` — the offline fallback page (``text/html``): a minimal
  shell that registers the worker, links back to ``/hud``, and offers a retry.

Everything is no-build vanilla HTML/CSS/JS — no bundler, no eval of any server
output. The router is included unconditionally by :func:`~friday.app.create_app`.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse, JSONResponse

from friday.logging import get_logger

logger = get_logger("friday.api.routes_pwa")

router = APIRouter()

#: The no-build PWA asset directory (manifest + worker + offline shell).
STATIC_DIR = Path(__file__).resolve().parent.parent / "pwa" / "static"
#: The web app manifest served at ``/manifest.webmanifest``.
MANIFEST_PATH = STATIC_DIR / "manifest.webmanifest"
#: The service worker served at the ROOT ``/service-worker.js`` (origin scope).
SERVICE_WORKER_PATH = STATIC_DIR / "service-worker.js"
#: The offline fallback page served at ``/offline.html``.
OFFLINE_PATH = STATIC_DIR / "offline.html"

#: A missing asset becomes a clean 404 rather than a leaked 500.
_NOT_FOUND = JSONResponse(status_code=404, content={"detail": "not found"})


@router.get("/manifest.webmanifest", response_model=None)
async def pwa_manifest() -> FileResponse | JSONResponse:
    """Serve the web app manifest (``application/manifest+json``).

    Declares the installable ``FRIDAY`` app (``display: standalone``,
    ``start_url: /hud``). The slightly unusual media type is the one browsers
    expect for ``.webmanifest``; ``Cache-Control`` is left to the default.
    """
    if not MANIFEST_PATH.is_file():  # pragma: no cover - asset always present
        return _NOT_FOUND
    return FileResponse(MANIFEST_PATH, media_type="application/manifest+json")


@router.get("/service-worker.js", response_model=None)
async def pwa_service_worker() -> FileResponse | JSONResponse:
    """Serve the service worker as ``text/javascript`` at the ROOT scope.

    Served from ``/service-worker.js`` (not a sub-path) so the worker's default
    control scope is the whole origin and it can intercept navigations for the
    dashboard/HUD. ``Service-Worker-Allowed: /`` is set explicitly so a worker
    registered with ``scope: "/"`` is accepted even if the file were ever served
    from elsewhere.
    """
    if not SERVICE_WORKER_PATH.is_file():  # pragma: no cover - asset always present
        return _NOT_FOUND
    return FileResponse(
        SERVICE_WORKER_PATH,
        media_type="text/javascript",
        headers={"Service-Worker-Allowed": "/"},
    )


@router.get("/offline.html", response_model=None)
async def pwa_offline() -> FileResponse | JSONResponse:
    """Serve the offline fallback shell (``text/html``).

    A minimal page that registers the root-scope worker, links back to ``/hud``,
    and offers a retry; it is the page the worker returns when a navigation fails
    while the network is down.
    """
    if not OFFLINE_PATH.is_file():  # pragma: no cover - asset always present
        return _NOT_FOUND
    return FileResponse(OFFLINE_PATH, media_type="text/html")
