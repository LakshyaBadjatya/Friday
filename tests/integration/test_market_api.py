"""Integration tests for the flagged ``/market`` surface (Tier 3; default off).

The market feature talks to the Dhan broker REST API over httpx. These tests
mount :data:`friday.market.router` on a FRESH ``FastAPI()`` app (NOT
``create_app``) with the relevant settings monkeypatched, so the slice passes
before any ``app.py`` wiring exists. The Dhan client is built lazily inside the
route from :func:`~friday.config.get_settings`; every Dhan call is ``respx``-
mocked, so the suite is fully offline.

Covered:
* ``GET /market/quote`` is ``404`` when ``FRIDAY_ENABLE_MARKET_DATA`` is off.
* Enabled + configured: ``GET /market/quote?symbol=`` returns the real parsed
  quote (proving the auth headers reach Dhan, and that nothing is fabricated).
* Enabled but creds missing -> a clear "configure Dhan" error (never a fake
  number).
* A missing ``symbol`` is a ``422`` (validation), not a 500.
"""

from __future__ import annotations

import httpx
import pytest
import respx
from fastapi import FastAPI
from fastapi.testclient import TestClient

import friday.api.routes_market as routes_market
from friday.config import Settings
from friday.market import router as market_router

BASE_URL = "https://api.dhan.co"

_QUOTE_BODY = {
    "data": {
        "NSE_EQ": {
            "11536": {
                "last_price": 4525.55,
                "ohlc": {"open": 4521.45, "close": 4507.85, "high": 4530, "low": 4500},
            }
        }
    },
    "status": "success",
}


def _app() -> FastAPI:
    """A fresh app with ONLY the market router mounted (no ``create_app``)."""
    app = FastAPI()
    app.include_router(market_router)
    return app


def _settings(*, enabled: bool, creds: bool = True) -> Settings:
    return Settings(
        _env_file=None,
        enable_market_data=enabled,
        dhan_client_id="1000000001" if creds else None,
        dhan_access_token="JWT-TOKEN" if creds else None,
    )


def test_market_quote_disabled_returns_404(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(routes_market, "get_settings", lambda: _settings(enabled=False))
    with TestClient(_app()) as client:
        resp = client.get("/market/quote", params={"symbol": "NSE_EQ:11536"})
    assert resp.status_code == 404
    assert resp.json()["detail"] == "market data disabled"


def test_market_quote_default_off_is_404(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pristine env-default settings (flag off) -> the route does not exist."""
    monkeypatch.setattr(routes_market, "get_settings", lambda: Settings(_env_file=None))
    with TestClient(_app()) as client:
        resp = client.get("/market/quote", params={"symbol": "NSE_EQ:11536"})
    assert resp.status_code == 404


@respx.mock
def test_market_quote_enabled_returns_real_quote(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(routes_market, "get_settings", lambda: _settings(enabled=True))
    route = respx.post(f"{BASE_URL}/v2/marketfeed/quote").mock(
        return_value=httpx.Response(200, json=_QUOTE_BODY)
    )
    with TestClient(_app()) as client:
        resp = client.get("/market/quote", params={"symbol": "NSE_EQ:11536"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["symbol"] == "NSE_EQ:11536"
    assert body["quote"]["last_price"] == 4525.55
    # The configured Dhan secret reached the broker as the documented header.
    assert route.calls.last.request.headers["access-token"] == "JWT-TOKEN"


def test_market_quote_enabled_no_creds_is_configure_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        routes_market, "get_settings", lambda: _settings(enabled=True, creds=False)
    )
    with TestClient(_app()) as client:
        resp = client.get("/market/quote", params={"symbol": "NSE_EQ:11536"})
    assert resp.status_code == 502
    body = resp.json()
    assert "configure dhan" in body["error"].lower()


def test_market_quote_missing_symbol_is_422(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(routes_market, "get_settings", lambda: _settings(enabled=True))
    with TestClient(_app()) as client:
        resp = client.get("/market/quote")
    assert resp.status_code == 422


@respx.mock
def test_market_quote_dhan_error_maps_to_502(monkeypatch: pytest.MonkeyPatch) -> None:
    """A Dhan non-2xx surfaces as a clean 502, never a leaked 500 or fake data."""
    monkeypatch.setattr(routes_market, "get_settings", lambda: _settings(enabled=True))
    respx.post(f"{BASE_URL}/v2/marketfeed/quote").mock(
        return_value=httpx.Response(503, text="upstream down")
    )
    with TestClient(_app()) as client:
        resp = client.get("/market/quote", params={"symbol": "NSE_EQ:11536"})
    assert resp.status_code == 502
    assert "quote" not in resp.json()
