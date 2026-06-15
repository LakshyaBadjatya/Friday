"""Integration tests for the flagged ``/email`` surface (Tier 3; default off).

These tests mount the ``/email`` router on a FRESH ``FastAPI()`` app (NOT
``create_app``) with the relevant settings monkeypatched, so the slice passes
before any ``app.py`` wiring exists. The :class:`~friday.integrations.email.GmailClient`
+ OAuth token are read lazily inside the route from
:func:`~friday.config.get_settings`; the outbound Gmail v1 HTTP is mocked with
``respx`` and the summary LLM is a :class:`~friday.providers.llm.FakeLLM` — no
live network, no real OAuth.

Covered:
* ``GET /email/inbox`` and ``POST /email/draft`` both ``404`` when
  ``enable_email`` is off (the feature simply does not exist).
* Enabled ``GET /email/inbox`` returns the listed messages plus a summary.
* Enabled ``POST /email/draft`` creates a DRAFT (never auto-sends) and returns it.
* Enabled but NO token configured -> a clear error status (never a leaked 500),
  and the secret token is never echoed in any response body.
* A malformed ``POST /email/draft`` body is mapped to 422 (not a leaked 500).
"""

from __future__ import annotations

import httpx
import pytest
import respx
from fastapi import FastAPI
from fastapi.testclient import TestClient

import friday.api.routes_email as routes_email
from friday.config import Settings
from friday.providers.llm import FakeLLM, LLMResponse

_LIST_URL = "https://gmail.googleapis.com/gmail/v1/users/me/messages"
_DRAFTS_URL = "https://gmail.googleapis.com/gmail/v1/users/me/drafts"
_SEND_URL = "https://gmail.googleapis.com/gmail/v1/users/me/messages/send"

_LIST_BODY = {
    "messages": [
        {"id": "m1", "threadId": "t1"},
        {"id": "m2", "threadId": "t2"},
    ]
}


def _app() -> FastAPI:
    """A fresh app with ONLY the email router mounted (no ``create_app``)."""
    app = FastAPI()
    app.include_router(routes_email.router)
    return app


def _enabled_settings(token: str | None = "gmail-secret-123") -> Settings:
    return Settings(_env_file=None, enable_email=True, gmail_oauth_token=token)


def _disabled_settings() -> Settings:
    return Settings(_env_file=None, enable_email=False)


def _patch_llm(monkeypatch: pytest.MonkeyPatch, text: str) -> None:
    """Make the route build a scripted FakeLLM (offline summary)."""
    monkeypatch.setattr(
        routes_email, "build_llm", lambda settings: FakeLLM([LLMResponse(text=text)])
    )


def test_inbox_disabled_returns_404(monkeypatch: pytest.MonkeyPatch) -> None:
    """``GET /email/inbox`` is 404 when the email flag is off."""
    monkeypatch.setattr(routes_email, "get_settings", _disabled_settings)
    with TestClient(_app()) as client:
        resp = client.get("/email/inbox")
    assert resp.status_code == 404


def test_draft_disabled_returns_404(monkeypatch: pytest.MonkeyPatch) -> None:
    """``POST /email/draft`` is 404 when the email flag is off."""
    monkeypatch.setattr(routes_email, "get_settings", _disabled_settings)
    with TestClient(_app()) as client:
        resp = client.post(
            "/email/draft", json={"message_id": "m1", "body": "hi"}
        )
    assert resp.status_code == 404


@respx.mock
def test_inbox_enabled_lists_and_summarizes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Enabled ``GET /email/inbox`` returns the messages and an LLM summary."""
    monkeypatch.setattr(routes_email, "get_settings", lambda: _enabled_settings())
    _patch_llm(monkeypatch, "2 unread, nothing urgent.")
    route = respx.get(_LIST_URL).mock(
        return_value=httpx.Response(200, json=_LIST_BODY)
    )

    with TestClient(_app()) as client:
        resp = client.get("/email/inbox")

    assert resp.status_code == 200
    body = resp.json()
    assert [m["id"] for m in body["messages"]] == ["m1", "m2"]
    assert body["count"] == 2
    assert body["summary"] == "2 unread, nothing urgent."
    sent = route.calls.last.request
    assert sent.headers["Authorization"] == "Bearer gmail-secret-123"
    # The token never leaks into the response body.
    assert "gmail-secret-123" not in resp.text


@respx.mock
def test_draft_enabled_creates_draft_never_sends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Enabled ``POST /email/draft`` creates a DRAFT and never auto-sends."""
    monkeypatch.setattr(routes_email, "get_settings", lambda: _enabled_settings())
    respx.get(f"{_LIST_URL}/m1").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "m1",
                "threadId": "t1",
                "payload": {
                    "headers": [
                        {"name": "Subject", "value": "Hello"},
                        {"name": "From", "value": "alice@example.com"},
                    ]
                },
            },
        )
    )
    created = {"id": "d1", "message": {"id": "msgd1", "threadId": "t1"}}
    drafts_route = respx.post(_DRAFTS_URL).mock(
        return_value=httpx.Response(200, json=created)
    )
    send_route = respx.post(_SEND_URL).mock(
        return_value=httpx.Response(200, json={"id": "nope"})
    )

    with TestClient(_app()) as client:
        resp = client.post(
            "/email/draft", json={"message_id": "m1", "body": "Thanks!"}
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["draft"]["id"] == "d1"
    assert drafts_route.called
    # CRITICAL: confirm-only — a draft is NEVER auto-sent.
    assert not send_route.called
    sent = drafts_route.calls.last.request
    assert sent.headers["Authorization"] == "Bearer gmail-secret-123"


def test_inbox_enabled_no_token_returns_clear_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Enabled but NO token -> a clear error (never a leaked 500), token unechoed."""
    monkeypatch.setattr(
        routes_email, "get_settings", lambda: _enabled_settings(token=None)
    )
    _patch_llm(monkeypatch, "unused")
    with TestClient(_app()) as client:
        resp = client.get("/email/inbox")

    assert resp.status_code == 400
    body = resp.json()
    assert "token" in body["detail"].lower()


def test_draft_bad_body_returns_422(monkeypatch: pytest.MonkeyPatch) -> None:
    """A malformed ``POST /email/draft`` body is mapped to 422 (not a leaked 500)."""
    monkeypatch.setattr(routes_email, "get_settings", lambda: _enabled_settings())
    with TestClient(_app()) as client:
        resp = client.post("/email/draft", json={"message_id": "m1"})
    assert resp.status_code == 422
