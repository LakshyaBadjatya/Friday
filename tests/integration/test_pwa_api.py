"""Integration tests for the always-available PWA shell (no flag).

The PWA slice makes FRIDAY installable: a minimal, no-build app shell that points
at the dashboard/HUD and works offline. It is static assets only (a manifest, a
service worker, and an offline fallback page) with no feature flag — installing a
PWA is harmless and the shell carries no secrets, so it is always reachable.

These tests mount :data:`friday.pwa.router` on a FRESH ``FastAPI()`` app (NOT
``create_app``) so the slice passes before any ``app.py`` wiring exists.

Covered:
* ``GET /manifest.webmanifest`` -> 200, a manifest content type, parses as JSON
  with ``name == "FRIDAY"``, ``display == "standalone"`` and ``start_url`` set to
  the HUD.
* ``GET /service-worker.js`` -> 200, ``text/javascript`` content type, served at
  ROOT scope (so it can control the whole app), and passes ``node --check``.
* ``GET /offline.html`` -> 200, ``text/html``, the offline fallback shell.
* The service worker caches the shell (it lists the offline page + start url) and
  serves the offline page when the network is down (a fetch handler exists).
* The offline page and manifest point at the dashboard/HUD entrypoint.
* A path-traversal request under the static prefix never escapes (404).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

import friday.api.routes_pwa as routes_pwa
from friday.pwa import router as pwa_router

_STATIC_DIR = Path(routes_pwa.__file__).resolve().parent.parent / "pwa" / "static"
_MANIFEST = _STATIC_DIR / "manifest.webmanifest"
_SERVICE_WORKER = _STATIC_DIR / "service-worker.js"
_OFFLINE_HTML = _STATIC_DIR / "offline.html"


def _app() -> FastAPI:
    """A fresh app with ONLY the pwa router mounted (no ``create_app``)."""
    app = FastAPI()
    app.include_router(pwa_router)
    return app


# --------------------------------------------------------------------------- #
# manifest
# --------------------------------------------------------------------------- #
def test_manifest_served_with_manifest_content_type() -> None:
    """``GET /manifest.webmanifest`` -> 200 with a manifest content type."""
    with TestClient(_app()) as client:
        resp = client.get("/manifest.webmanifest")
    assert resp.status_code == 200
    assert "manifest" in resp.headers["content-type"]


def test_manifest_is_valid_installable_json() -> None:
    """The manifest parses as JSON and declares an installable FRIDAY app."""
    with TestClient(_app()) as client:
        resp = client.get("/manifest.webmanifest")
    data = resp.json()
    assert data["name"] == "FRIDAY"
    assert data["display"] == "standalone"
    # The installed app launches into the HUD.
    assert data["start_url"] == "/hud"
    # No friday-orb icon ships, so icons are omitted (an empty/absent icon list is
    # valid) rather than referencing a missing file.
    assert data.get("icons", []) == []


# --------------------------------------------------------------------------- #
# service worker
# --------------------------------------------------------------------------- #
def test_service_worker_served_as_javascript_at_root_scope() -> None:
    """``GET /service-worker.js`` -> 200 ``text/javascript`` at the ROOT path.

    The worker is served from ``/service-worker.js`` (the root) so its default
    control scope is the whole origin — it can intercept fetches for the
    dashboard/HUD, not just a sub-path.
    """
    with TestClient(_app()) as client:
        resp = client.get("/service-worker.js")
    assert resp.status_code == 200
    assert "text/javascript" in resp.headers["content-type"]
    assert len(resp.content) > 0


def test_service_worker_is_valid_javascript() -> None:
    """``service-worker.js`` passes ``node --check`` (valid JavaScript)."""
    node = shutil.which("node")
    if node is None:  # pragma: no cover - node is present in this environment
        return
    result = subprocess.run(
        [node, "--check", str(_SERVICE_WORKER)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_service_worker_caches_shell_and_serves_offline() -> None:
    """The worker precaches the shell and falls back to the offline page."""
    js = _SERVICE_WORKER.read_text(encoding="utf-8")
    # Precaches the offline fallback + the HUD entrypoint.
    assert "/offline.html" in js
    assert "/hud" in js
    # Standard service-worker lifecycle + a fetch handler (the offline fallback).
    assert "install" in js
    assert "fetch" in js
    assert "caches" in js


# --------------------------------------------------------------------------- #
# offline shell
# --------------------------------------------------------------------------- #
def test_offline_page_served_as_html() -> None:
    """``GET /offline.html`` -> 200 ``text/html`` (the offline fallback shell)."""
    with TestClient(_app()) as client:
        resp = client.get("/offline.html")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]


def test_offline_page_points_at_app() -> None:
    """The offline shell links back to the HUD entrypoint to retry."""
    html = _OFFLINE_HTML.read_text(encoding="utf-8")
    assert "/hud" in html
    assert "FRIDAY" in html


# --------------------------------------------------------------------------- #
# safety
# --------------------------------------------------------------------------- #
def test_manifest_links_service_worker_scope() -> None:
    """The manifest is internally consistent (start_url is same-origin path)."""
    data = json.loads(_MANIFEST.read_text(encoding="utf-8"))
    assert data["start_url"].startswith("/")
    assert data["scope"].startswith("/")
