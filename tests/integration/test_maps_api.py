"""Integration tests for the flagged ``/maps`` surface (Tier 3; default off).

The Maps feature is a Photorealistic 3D globe served as a no-build page that
loads the Google Maps JS API. The API key is NEVER baked into the HTML; the
page fetches it at runtime from ``GET /maps/config``.

These tests mount :data:`friday.maps.router` on a FRESH ``FastAPI()`` app (NOT
``create_app``) with the relevant settings monkeypatched, so the slice passes
before any ``app.py`` wiring exists. The maps client/key is read lazily inside
the route from :func:`~friday.config.get_settings`.

Covered:
* ``GET /maps`` and ``GET /maps/config`` both ``404`` when ``enable_maps`` off.
* ``GET /maps`` serves the index HTML when enabled (loads the Maps JS API; no
  key baked in).
* ``GET /maps/config`` returns ``{"apiKey": <key or "">, "enabled": True}`` when
  enabled — the key field is present and carries the configured secret.
* ``GET /maps/config`` returns ``apiKey == ""`` when enabled but no key is set
  (never crashes on a missing key).
* The static ``maps.js`` passes ``node --check`` (valid JavaScript).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import friday.api.routes_maps as routes_maps
from friday.config import Settings
from friday.maps import router as maps_router

_STATIC_DIR = Path(routes_maps.__file__).resolve().parent.parent / "maps" / "static"
_MAPS_JS = _STATIC_DIR / "maps.js"
_INDEX_HTML = _STATIC_DIR / "index.html"


def _app() -> FastAPI:
    """A fresh app with ONLY the maps router mounted (no ``create_app``)."""
    app = FastAPI()
    app.include_router(maps_router)
    return app


def _enabled_settings(key: str | None = "test-key-123") -> Settings:
    return Settings(
        _env_file=None,
        enable_maps=True,
        google_maps_api_key=key,
    )


def _disabled_settings() -> Settings:
    return Settings(_env_file=None, enable_maps=False)


def test_maps_index_disabled_returns_404(monkeypatch: pytest.MonkeyPatch) -> None:
    """``GET /maps`` is 404 when the maps flag is off (feature does not exist)."""
    monkeypatch.setattr(routes_maps, "get_settings", _disabled_settings)
    with TestClient(_app()) as client:
        resp = client.get("/maps")
    assert resp.status_code == 404


def test_maps_config_disabled_returns_404(monkeypatch: pytest.MonkeyPatch) -> None:
    """``GET /maps/config`` is 404 when the maps flag is off."""
    monkeypatch.setattr(routes_maps, "get_settings", _disabled_settings)
    with TestClient(_app()) as client:
        resp = client.get("/maps/config")
    assert resp.status_code == 404


def test_maps_index_enabled_serves_html(monkeypatch: pytest.MonkeyPatch) -> None:
    """Enabled ``GET /maps`` serves the index HTML; loads Maps JS, no key baked."""
    monkeypatch.setattr(routes_maps, "get_settings", lambda: _enabled_settings())
    with TestClient(_app()) as client:
        resp = client.get("/maps")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    body = resp.text
    # The page loads the Google Maps JS API and fetches its key at runtime.
    assert "maps.googleapis.com" in body
    # The secret key must NOT be baked into the served HTML.
    assert "test-key-123" not in body


def test_maps_config_enabled_returns_key_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Enabled ``GET /maps/config`` returns the apiKey + enabled fields."""
    monkeypatch.setattr(routes_maps, "get_settings", lambda: _enabled_settings())
    with TestClient(_app()) as client:
        resp = client.get("/maps/config")
    assert resp.status_code == 200
    body = resp.json()
    assert body == {"apiKey": "test-key-123", "enabled": True}


def test_maps_config_enabled_no_key_returns_empty_string(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Enabled but no key set -> ``apiKey == ""`` (never crashes on missing key)."""
    monkeypatch.setattr(
        routes_maps, "get_settings", lambda: _enabled_settings(key=None)
    )
    with TestClient(_app()) as client:
        resp = client.get("/maps/config")
    assert resp.status_code == 200
    body = resp.json()
    assert body == {"apiKey": "", "enabled": True}


def test_maps_serves_static_js(monkeypatch: pytest.MonkeyPatch) -> None:
    """Enabled ``GET /maps/static/maps.js`` serves the frontend controller."""
    monkeypatch.setattr(routes_maps, "get_settings", lambda: _enabled_settings())
    with TestClient(_app()) as client:
        resp = client.get("/maps/static/maps.js")
    assert resp.status_code == 200
    assert "javascript" in resp.headers["content-type"]
    # No key is baked into the JS either; it is fetched from /maps/config.
    assert "test-key-123" not in resp.text


def test_maps_static_disabled_returns_404(monkeypatch: pytest.MonkeyPatch) -> None:
    """The static asset route is also absent when the flag is off."""
    monkeypatch.setattr(routes_maps, "get_settings", _disabled_settings)
    with TestClient(_app()) as client:
        resp = client.get("/maps/static/maps.js")
    assert resp.status_code == 404


def test_maps_js_is_valid_javascript() -> None:
    """``maps.js`` passes ``node --check`` (parses as valid JavaScript)."""
    node = shutil.which("node")
    if node is None:  # pragma: no cover - node is present in this environment
        pytest.skip("node is not available")
    result = subprocess.run(
        [node, "--check", str(_MAPS_JS)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_maps_index_references_runtime_config() -> None:
    """The index HTML wires the runtime ``/maps/config`` fetch (no baked key)."""
    html = _INDEX_HTML.read_text(encoding="utf-8")
    assert "/maps/config" in html
    # The Maps JS API <script> is bootstrapped, not hardcoded with a key.
    assert "maps.googleapis.com" in html
