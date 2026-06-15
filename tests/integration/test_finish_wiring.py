"""Integration tests for the FINISH wiring in :mod:`friday.app`.

This is the "everywhere" integration slice — the last seam that turns the built
pieces into a coherent app:

* The always-available PWA shell (:data:`friday.pwa.router`) is included
  UNCONDITIONALLY by :func:`friday.app.create_app` (no feature flag), so a fresh
  install is reachable and installable even on the offline default. The three
  static surfaces (``GET /manifest.webmanifest`` / ``GET /service-worker.js`` /
  ``GET /offline.html``) all return 200 through the real ``create_app`` (not a
  hand-mounted router), proving the wiring — not just the router in isolation.

* The exposure safety nudge: when the configured uvicorn bind host is
  non-loopback (``FRIDAY_BIND_HOST != 127.0.0.1/localhost``) AND ``require_auth``
  is off, startup logs a prominent WARNING that FRIDAY is exposed to the network
  without auth. It is ADVISORY — boot is NEVER refused (local dev keeps working).
  On the loopback default (or with auth on) it is SILENT.

Fully offline (``llm_provider="fake"``, ``memory_db_path=":memory:"``) and run
through the real :func:`create_app` with ``get_settings`` monkeypatched to the
pinned settings (mirroring the other ``*_wiring`` tests), so the eager runtime
install reads them.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from friday import app as app_mod
from friday.app import create_app
from friday.config import Settings


def _settings(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "_env_file": None,
        "llm_provider": "fake",
        "memory_db_path": ":memory:",
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def _app(monkeypatch: pytest.MonkeyPatch, settings: Settings) -> FastAPI:
    """Build the real app with ``get_settings`` pinned to ``settings``.

    ``create_app`` reads the module-level ``get_settings`` for its eager runtime
    install + startup checks, so patching it here pins the whole build.
    """
    monkeypatch.setattr(app_mod, "get_settings", lambda: settings)
    return create_app()


# --------------------------------------------------------------------------- #
# PWA shell is wired into create_app (always-on, no flag)
# --------------------------------------------------------------------------- #
def test_pwa_manifest_served_through_create_app(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``GET /manifest.webmanifest`` -> 200 through the REAL create_app wiring."""
    app = _app(monkeypatch, _settings())
    with TestClient(app) as client:
        resp = client.get("/manifest.webmanifest")
    assert resp.status_code == 200
    assert "manifest" in resp.headers["content-type"]
    assert resp.json()["name"] == "FRIDAY"


def test_pwa_service_worker_served_through_create_app(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``GET /service-worker.js`` -> 200 ``text/javascript`` at the ROOT scope."""
    app = _app(monkeypatch, _settings())
    with TestClient(app) as client:
        resp = client.get("/service-worker.js")
    assert resp.status_code == 200
    assert "text/javascript" in resp.headers["content-type"]
    assert len(resp.content) > 0


def test_pwa_offline_page_served_through_create_app(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``GET /offline.html`` -> 200 ``text/html`` (the offline fallback shell)."""
    app = _app(monkeypatch, _settings())
    with TestClient(app) as client:
        resp = client.get("/offline.html")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]


def test_pwa_shell_available_even_with_all_flags_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The PWA shell has NO feature flag — it is reachable on the offline default.

    Even the HUD flag (which the manifest's ``start_url`` points at) is off here;
    the shell is still served because installing a PWA is harmless and carries no
    secrets.
    """
    app = _app(monkeypatch, _settings(enable_hud=False))
    with TestClient(app) as client:
        for path in ("/manifest.webmanifest", "/service-worker.js", "/offline.html"):
            assert client.get(path).status_code == 200, path


# --------------------------------------------------------------------------- #
# Exposure safety nudge (advisory WARNING; never refuses boot)
# --------------------------------------------------------------------------- #
def test_bind_host_defaults_to_loopback() -> None:
    """The bind host defaults to the loopback so the gateway is local-only."""
    assert _settings().bind_host == "127.0.0.1"


def test_warns_when_public_bind_without_auth(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A non-loopback bind with auth off logs a prominent exposure WARNING.

    The warning is ADVISORY: ``create_app`` still returns a working app (boot is
    never refused), so local dev that binds ``0.0.0.0`` keeps running.
    """
    settings = _settings(bind_host="0.0.0.0", require_auth=False)  # noqa: S104
    with caplog.at_level("WARNING"):
        app = _app(monkeypatch, settings)
    # Boot was NOT refused.
    assert isinstance(app, FastAPI)
    warnings = [
        rec
        for rec in caplog.records
        if rec.levelname == "WARNING" and "without auth" in rec.getMessage().lower()
    ]
    assert warnings, "expected an exposure WARNING for public bind + no auth"
    # The offending host rides the message so the operator can see it.
    assert any("0.0.0.0" in rec.getMessage() for rec in warnings)
    # The app still serves the PWA shell (it really did boot).
    with TestClient(app) as client:
        assert client.get("/manifest.webmanifest").status_code == 200


def test_silent_on_loopback_default(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The loopback default never fires the exposure warning."""
    with caplog.at_level("WARNING"):
        _app(monkeypatch, _settings())
    exposure = [
        rec for rec in caplog.records if "without auth" in rec.getMessage().lower()
    ]
    assert exposure == []


def test_silent_on_localhost_bind(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """``localhost`` is treated as loopback — no exposure warning."""
    with caplog.at_level("WARNING"):
        _app(monkeypatch, _settings(bind_host="localhost"))
    exposure = [
        rec for rec in caplog.records if "without auth" in rec.getMessage().lower()
    ]
    assert exposure == []


def test_silent_on_public_bind_with_auth_on(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A public bind is fine when auth is on — the nudge stays silent."""
    settings = _settings(
        bind_host="0.0.0.0",  # noqa: S104
        require_auth=True,
        api_keys=["k1"],
    )
    with caplog.at_level("WARNING"):
        _app(monkeypatch, settings)
    exposure = [
        rec for rec in caplog.records if "without auth" in rec.getMessage().lower()
    ]
    assert exposure == []
