"""Integration tests for the flagged ``/hud`` surface (Tier 3; default off).

The HUD is a no-build "cockpit" page served as static assets: an arc-reactor
boot sequence animation plus a Cmd/Ctrl-K command palette that calls the
existing ``/chat`` and ``/admin`` endpoints (same origin, no key baked in).

These tests mount :data:`friday.hud.router` on a FRESH ``FastAPI()`` app (NOT
``create_app``) with the relevant settings monkeypatched, so the slice passes
before any ``app.py`` wiring exists. The flag is read lazily inside the route
from :func:`~friday.config.get_settings`.

Covered:
* ``GET /hud`` and ``GET /hud/static/hud.js`` both ``404`` when ``enable_hud``
  is off (the feature simply does not exist).
* ``GET /hud`` serves the index HTML when enabled (boot sequence + palette).
* ``GET /hud/static/hud.js`` / ``hud.css`` serve with sane content types when
  enabled.
* A path-traversal request under ``/hud/static`` is ``404`` (never escapes).
* The static ``hud.js`` passes ``node --check`` (valid JavaScript).
* The index HTML wires the command palette and references same-origin
  ``/chat`` + ``/admin`` (no secret baked in).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import friday.api.routes_hud as routes_hud
from friday.config import Settings
from friday.hud import router as hud_router

_STATIC_DIR = Path(routes_hud.__file__).resolve().parent.parent / "hud" / "static"
_HUD_JS = _STATIC_DIR / "hud.js"
_HUD_CSS = _STATIC_DIR / "hud.css"
_INDEX_HTML = _STATIC_DIR / "index.html"


def _app() -> FastAPI:
    """A fresh app with ONLY the hud router mounted (no ``create_app``)."""
    app = FastAPI()
    app.include_router(hud_router)
    return app


def _enabled_settings() -> Settings:
    return Settings(_env_file=None, enable_hud=True)


def _disabled_settings() -> Settings:
    return Settings(_env_file=None, enable_hud=False)


def test_hud_index_disabled_returns_404(monkeypatch: pytest.MonkeyPatch) -> None:
    """``GET /hud`` is 404 when the hud flag is off (feature does not exist)."""
    monkeypatch.setattr(routes_hud, "get_settings", _disabled_settings)
    with TestClient(_app()) as client:
        resp = client.get("/hud")
    assert resp.status_code == 404


def test_hud_static_disabled_returns_404(monkeypatch: pytest.MonkeyPatch) -> None:
    """The static asset route is also absent when the flag is off."""
    monkeypatch.setattr(routes_hud, "get_settings", _disabled_settings)
    with TestClient(_app()) as client:
        resp = client.get("/hud/static/hud.js")
    assert resp.status_code == 404


def test_hud_index_enabled_serves_html(monkeypatch: pytest.MonkeyPatch) -> None:
    """Enabled ``GET /hud`` serves the cockpit index HTML."""
    monkeypatch.setattr(routes_hud, "get_settings", _enabled_settings)
    with TestClient(_app()) as client:
        resp = client.get("/hud")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    body = resp.text
    # The cockpit boots the HUD controller and links its styles.
    assert "/hud/static/hud.js" in body
    assert "/hud/static/hud.css" in body


def test_hud_serves_static_js(monkeypatch: pytest.MonkeyPatch) -> None:
    """Enabled ``GET /hud/static/hud.js`` serves the frontend controller."""
    monkeypatch.setattr(routes_hud, "get_settings", _enabled_settings)
    with TestClient(_app()) as client:
        resp = client.get("/hud/static/hud.js")
    assert resp.status_code == 200
    assert "javascript" in resp.headers["content-type"]


def test_hud_serves_static_css(monkeypatch: pytest.MonkeyPatch) -> None:
    """Enabled ``GET /hud/static/hud.css`` serves the cockpit styles."""
    monkeypatch.setattr(routes_hud, "get_settings", _enabled_settings)
    with TestClient(_app()) as client:
        resp = client.get("/hud/static/hud.css")
    assert resp.status_code == 200
    assert "css" in resp.headers["content-type"]


def test_hud_static_path_traversal_is_404(monkeypatch: pytest.MonkeyPatch) -> None:
    """A request escaping ``static/`` is 404 (path-traversal guard)."""
    monkeypatch.setattr(routes_hud, "get_settings", _enabled_settings)
    with TestClient(_app()) as client:
        resp = client.get("/hud/static/../../config.py")
    assert resp.status_code == 404


def test_hud_commands_disabled_returns_404(monkeypatch: pytest.MonkeyPatch) -> None:
    """The command catalog is absent when the flag is off."""
    monkeypatch.setattr(routes_hud, "get_settings", _disabled_settings)
    with TestClient(_app()) as client:
        resp = client.get("/hud/commands")
    assert resp.status_code == 404


def test_hud_commands_enabled_returns_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    """Enabled ``GET /hud/commands`` returns the palette catalog (id/title/hint)."""
    monkeypatch.setattr(routes_hud, "get_settings", _enabled_settings)
    with TestClient(_app()) as client:
        resp = client.get("/hud/commands")
    assert resp.status_code == 200
    commands = resp.json()["commands"]
    ids = {c["id"] for c in commands}
    assert {"ask", "theme", "roster", "view-system"} <= ids
    assert all({"id", "title", "hint"} <= set(c) for c in commands)


def test_hud_js_defines_theme_switching() -> None:
    """The cockpit JS wires theme cycling + persistence (Wave G themes)."""
    js = _HUD_JS.read_text(encoding="utf-8")
    assert "applyTheme" in js
    assert "cycleTheme" in js
    assert "friday-theme" in js  # localStorage key
    css = _HUD_CSS.read_text(encoding="utf-8")
    assert '[data-theme="amber"]' in css
    assert '[data-theme="light"]' in css


def test_hud_js_is_valid_javascript() -> None:
    """``hud.js`` passes ``node --check`` (parses as valid JavaScript)."""
    node = shutil.which("node")
    if node is None:  # pragma: no cover - node is present in this environment
        pytest.skip("node is not available")
    result = subprocess.run(
        [node, "--check", str(_HUD_JS)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_hud_index_wires_command_palette_and_endpoints() -> None:
    """The index HTML wires the command palette and same-origin /chat + /admin."""
    html = _INDEX_HTML.read_text(encoding="utf-8")
    # The Cmd/Ctrl-K command palette UI exists in the page.
    assert "palette" in html.lower()
    # The cockpit talks to the existing endpoints from the browser (same origin).
    js = _HUD_JS.read_text(encoding="utf-8")
    assert "/chat" in js
    assert "/admin" in js
