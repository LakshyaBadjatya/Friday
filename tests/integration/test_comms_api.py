"""Integration tests for the flagged ``/comms`` surface (Tier 3; default off).

These tests mount the comms ``APIRouter`` on a FRESH ``FastAPI()`` app (NOT
``create_app``) with the relevant settings monkeypatched, so the slice passes
before any ``app.py`` wiring exists. The Twilio client + credentials are read
lazily inside the route from :func:`~friday.config.get_settings`; the outbound
Twilio HTTP is mocked with ``respx`` — no live network, no real credentials.

Covered:
* ``POST /comms/sms`` and ``POST /comms/whatsapp`` both ``404`` when
  ``enable_comms`` is off (the feature simply does not exist).
* Enabled ``POST /comms/sms`` sends the SMS and returns the message JSON; the
  auth token rides only in the basic-auth header and is never echoed.
* Enabled ``POST /comms/whatsapp`` prefixes the numbers with ``whatsapp:``.
* A bad body -> ``422`` (not a leaked 500).
* Enabled but NO credentials -> a clear error status (never a leaked 500), and
  the secret token is never echoed in any response body.
"""

from __future__ import annotations

import httpx
import pytest
import respx
from fastapi import FastAPI
from fastapi.testclient import TestClient

import friday.api.routes_comms as routes_comms
from friday.config import Settings

_SID = "AC0123456789abcdef"
_TOKEN = "tok-secret-abcdef"
_FROM = "+15005550006"
_MESSAGES_URL = f"https://api.twilio.com/2010-04-01/Accounts/{_SID}/Messages.json"

_SMS_BODY = {
    "sid": "SMxxxx",
    "status": "queued",
    "to": "+15551230000",
    "from": _FROM,
}


def _app() -> FastAPI:
    """A fresh app with ONLY the comms router mounted (no ``create_app``)."""
    app = FastAPI()
    app.include_router(routes_comms.router)
    return app


def _enabled_settings(
    sid: str | None = _SID,
    token: str | None = _TOKEN,
    from_number: str = _FROM,
) -> Settings:
    return Settings(
        _env_file=None,
        enable_comms=True,
        twilio_account_sid=sid,
        twilio_auth_token=token,
        twilio_from_number=from_number,
    )


def _disabled_settings() -> Settings:
    return Settings(_env_file=None, enable_comms=False)


def test_sms_disabled_returns_404(monkeypatch: pytest.MonkeyPatch) -> None:
    """``POST /comms/sms`` is 404 when the comms flag is off."""
    monkeypatch.setattr(routes_comms, "get_settings", _disabled_settings)
    with TestClient(_app()) as client:
        resp = client.post("/comms/sms", json={"to": "+15551230000", "body": "hi"})
    assert resp.status_code == 404


def test_whatsapp_disabled_returns_404(monkeypatch: pytest.MonkeyPatch) -> None:
    """``POST /comms/whatsapp`` is 404 when the comms flag is off."""
    monkeypatch.setattr(routes_comms, "get_settings", _disabled_settings)
    with TestClient(_app()) as client:
        resp = client.post(
            "/comms/whatsapp", json={"to": "+15551230000", "body": "hi"}
        )
    assert resp.status_code == 404


@respx.mock
def test_sms_enabled_sends(monkeypatch: pytest.MonkeyPatch) -> None:
    """Enabled ``POST /comms/sms`` sends the SMS and returns the message JSON."""
    monkeypatch.setattr(routes_comms, "get_settings", lambda: _enabled_settings())
    route = respx.post(_MESSAGES_URL).mock(
        return_value=httpx.Response(201, json=_SMS_BODY)
    )

    with TestClient(_app()) as client:
        resp = client.post(
            "/comms/sms", json={"to": "+15551230000", "body": "hello there"}
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["message"]["sid"] == "SMxxxx"
    sent = route.calls.last.request
    assert sent.headers["Authorization"].startswith("Basic ")
    form = sent.content.decode()
    assert "To=%2B15551230000" in form
    assert "Body=hello+there" in form
    # The secret token is never echoed in the response.
    assert _TOKEN not in resp.text


@respx.mock
def test_whatsapp_enabled_sends_with_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Enabled ``POST /comms/whatsapp`` prefixes the numbers with ``whatsapp:``."""
    monkeypatch.setattr(routes_comms, "get_settings", lambda: _enabled_settings())
    route = respx.post(_MESSAGES_URL).mock(
        return_value=httpx.Response(201, json=_SMS_BODY)
    )

    with TestClient(_app()) as client:
        resp = client.post(
            "/comms/whatsapp", json={"to": "+15551230000", "body": "hi wa"}
        )

    assert resp.status_code == 200
    form = route.calls.last.request.content.decode()
    assert "To=whatsapp%3A%2B15551230000" in form
    assert _TOKEN not in resp.text


def test_sms_bad_body_returns_422(monkeypatch: pytest.MonkeyPatch) -> None:
    """A body missing ``to``/``body`` is 422, not a leaked 500."""
    monkeypatch.setattr(routes_comms, "get_settings", lambda: _enabled_settings())
    with TestClient(_app()) as client:
        resp = client.post("/comms/sms", json={"to": "+15551230000"})
    assert resp.status_code == 422


def test_sms_not_json_returns_422(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-JSON body is 422, not a leaked 500."""
    monkeypatch.setattr(routes_comms, "get_settings", lambda: _enabled_settings())
    with TestClient(_app()) as client:
        resp = client.post(
            "/comms/sms",
            content=b"not json",
            headers={"Content-Type": "application/json"},
        )
    assert resp.status_code == 422


def test_sms_enabled_no_creds_returns_clear_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Enabled but unconfigured -> a clear error status, token never echoed."""
    monkeypatch.setattr(
        routes_comms,
        "get_settings",
        lambda: _enabled_settings(sid=None, token=None, from_number=""),
    )
    with TestClient(_app()) as client:
        resp = client.post(
            "/comms/sms", json={"to": "+15551230000", "body": "hi"}
        )
    assert resp.status_code == 400
    assert resp.status_code != 500
    assert _TOKEN not in resp.text
