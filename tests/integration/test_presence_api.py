"""Integration tests for the ``/presence`` REST API (Tier 3).

Fully offline: the router is mounted on a **fresh** ``FastAPI()`` app (NOT
``create_app``), so it passes before any ``app.py`` wiring exists. The presence
flag and known-device map are forced via a monkeypatched ``get_settings`` in the
route module; the route builds a :class:`~friday.presence.scanner.FakePresenceScanner`
lazily (no ``bleak``, no Bluetooth) so a ``GET /presence`` exercises the full
service path with zero hardware.

Covered:
* ``GET /presence`` is ``404`` when ``FRIDAY_ENABLE_PRESENCE`` is off.
* ``GET /presence`` returns the present/absent/arrived/departed split when on.
* A test scanner can be injected on ``app.state`` so the present set is asserted.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import friday.api.routes_presence as routes_presence
from friday.config import Settings
from friday.presence.scanner import Device, FakePresenceScanner


def _settings(*, enabled: bool, known: list[str] | None = None) -> Settings:
    return Settings(
        _env_file=None,
        enable_presence=enabled,
        presence_known_devices=known or [],
    )


def _app(monkeypatch: pytest.MonkeyPatch, settings: Settings) -> FastAPI:
    """A fresh app with ONLY the presence router mounted, settings patched."""
    monkeypatch.setattr(routes_presence, "get_settings", lambda: settings)
    app = FastAPI()
    app.include_router(routes_presence.router)
    return app


def test_presence_disabled_is_404(monkeypatch: pytest.MonkeyPatch) -> None:
    app = _app(monkeypatch, _settings(enabled=False))
    with TestClient(app) as client:
        resp = client.get("/presence")
    assert resp.status_code == 404
    assert resp.json()["detail"] == "presence disabled"


def test_presence_enabled_returns_state(monkeypatch: pytest.MonkeyPatch) -> None:
    known = ["AA:BB:CC:DD:EE:FF=Phone", "11:22:33:44:55:66=Laptop"]
    app = _app(monkeypatch, _settings(enabled=True, known=known))
    with TestClient(app) as client:
        resp = client.get("/presence")
    assert resp.status_code == 200
    body = resp.json()
    # The default fake scanner sees nothing, so both known devices are absent.
    assert body["present"] == []
    assert sorted(body["absent"]) == ["Laptop", "Phone"]
    assert body["arrived"] == []
    assert body["departed"] == []
    assert isinstance(body["ts"], str)


def test_presence_enabled_with_injected_scanner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A scanner stashed on ``app.state`` overrides the default fake."""
    known = ["AA:BB:CC:DD:EE:FF=Phone", "11:22:33:44:55:66=Laptop"]
    app = _app(monkeypatch, _settings(enabled=True, known=known))
    phone = Device(address="AA:BB:CC:DD:EE:FF", name="bt", rssi=-50)
    app.state.presence_scanner = FakePresenceScanner(scans=[[phone]])

    with TestClient(app) as client:
        resp = client.get("/presence")
    assert resp.status_code == 200
    body = resp.json()
    assert body["present"] == ["Phone"]
    assert body["absent"] == ["Laptop"]


def test_presence_default_off_is_404(monkeypatch: pytest.MonkeyPatch) -> None:
    """With pristine env-default settings (flag off), the route is 404."""
    app = _app(monkeypatch, _settings(enabled=False))
    with TestClient(app) as client:
        resp = client.get("/presence")
    assert resp.status_code == 404
