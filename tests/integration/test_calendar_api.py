"""Integration tests for the flagged ``/calendar`` surface (Tier 3; default off).

These tests mount :data:`friday.integrations.calendar_router` on a FRESH
``FastAPI()`` app (NOT ``create_app``) with the relevant settings monkeypatched,
so the slice passes before any ``app.py`` wiring exists. The Google Calendar
client + OAuth token are read lazily inside the route from
:func:`~friday.config.get_settings`; the outbound Calendar v3 HTTP is mocked with
``respx`` — no live network, no real OAuth.

Covered:
* ``GET /calendar/events`` and ``POST /calendar/events`` both ``404`` when
  ``enable_calendar`` is off (the feature simply does not exist).
* Enabled ``GET /calendar/events?day=`` returns the day's parsed events.
* Enabled ``POST /calendar/events`` creates an event and returns it.
* Enabled but NO token configured -> a clear error status (never a leaked 500),
  and the secret token is never echoed in any response body.
"""

from __future__ import annotations

import httpx
import pytest
import respx
from fastapi import FastAPI
from fastapi.testclient import TestClient

import friday.api.routes_calendar as routes_calendar
from friday.config import Settings
from friday.integrations import calendar_router

_PRIMARY = "https://www.googleapis.com/calendar/v3/calendars/primary/events"

_LIST_BODY = {
    "items": [
        {"id": "ev1", "summary": "Standup"},
        {"id": "ev2", "summary": "Lunch"},
    ]
}


def _app() -> FastAPI:
    """A fresh app with ONLY the calendar router mounted (no ``create_app``)."""
    app = FastAPI()
    app.include_router(calendar_router)
    return app


def _enabled_settings(token: str | None = "oauth-secret-123") -> Settings:
    return Settings(
        _env_file=None,
        enable_calendar=True,
        google_oauth_token=token,
    )


def _disabled_settings() -> Settings:
    return Settings(_env_file=None, enable_calendar=False)


def test_events_get_disabled_returns_404(monkeypatch: pytest.MonkeyPatch) -> None:
    """``GET /calendar/events`` is 404 when the calendar flag is off."""
    monkeypatch.setattr(routes_calendar, "get_settings", _disabled_settings)
    with TestClient(_app()) as client:
        resp = client.get("/calendar/events")
    assert resp.status_code == 404


def test_events_post_disabled_returns_404(monkeypatch: pytest.MonkeyPatch) -> None:
    """``POST /calendar/events`` is 404 when the calendar flag is off."""
    monkeypatch.setattr(routes_calendar, "get_settings", _disabled_settings)
    with TestClient(_app()) as client:
        resp = client.post("/calendar/events", json={"event": {"summary": "x"}})
    assert resp.status_code == 404


@respx.mock
def test_events_get_enabled_lists_day(monkeypatch: pytest.MonkeyPatch) -> None:
    """Enabled ``GET /calendar/events?day=`` returns the day's parsed events."""
    monkeypatch.setattr(routes_calendar, "get_settings", lambda: _enabled_settings())
    route = respx.get(_PRIMARY).mock(return_value=httpx.Response(200, json=_LIST_BODY))

    with TestClient(_app()) as client:
        resp = client.get("/calendar/events", params={"day": "2026-06-15"})

    assert resp.status_code == 200
    body = resp.json()
    assert [e["id"] for e in body["events"]] == ["ev1", "ev2"]
    assert body["count"] == 2
    # The token rode only in the Authorization header (not echoed in the body).
    sent = route.calls.last.request
    assert sent.headers["Authorization"] == "Bearer oauth-secret-123"
    assert sent.url.params["timeMin"] == "2026-06-15T00:00:00Z"
    assert sent.url.params["timeMax"] == "2026-06-16T00:00:00Z"
    assert "oauth-secret-123" not in resp.text


@respx.mock
def test_events_post_enabled_creates_event(monkeypatch: pytest.MonkeyPatch) -> None:
    """Enabled ``POST /calendar/events`` creates an event and returns it."""
    monkeypatch.setattr(routes_calendar, "get_settings", lambda: _enabled_settings())
    created = {"id": "new1", "summary": "Coffee", "status": "confirmed"}
    route = respx.post(_PRIMARY).mock(return_value=httpx.Response(200, json=created))

    event = {
        "summary": "Coffee",
        "start": {"dateTime": "2026-06-15T15:00:00Z"},
        "end": {"dateTime": "2026-06-15T15:30:00Z"},
    }
    with TestClient(_app()) as client:
        resp = client.post("/calendar/events", json={"event": event})

    assert resp.status_code == 200
    body = resp.json()
    assert body["event"]["id"] == "new1"
    sent = route.calls.last.request
    assert sent.headers["Authorization"] == "Bearer oauth-secret-123"
    assert b"Coffee" in sent.content


def test_events_get_enabled_no_token_returns_clear_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Enabled but NO token -> a clear error (never a leaked 500), token unechoed."""
    monkeypatch.setattr(
        routes_calendar, "get_settings", lambda: _enabled_settings(token=None)
    )
    with TestClient(_app()) as client:
        resp = client.get("/calendar/events", params={"day": "2026-06-15"})

    assert resp.status_code == 400
    body = resp.json()
    assert "token" in body["detail"].lower()


def test_events_post_bad_body_returns_422(monkeypatch: pytest.MonkeyPatch) -> None:
    """A malformed ``POST`` body is mapped to 422 (not a leaked 500)."""
    monkeypatch.setattr(routes_calendar, "get_settings", lambda: _enabled_settings())
    with TestClient(_app()) as client:
        resp = client.post("/calendar/events", json={"not_event": 1})
    assert resp.status_code == 422
