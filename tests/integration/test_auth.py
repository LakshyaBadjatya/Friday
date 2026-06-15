"""Gateway auth integration tests (Phase 6, Stage 1).

Drives the real FastAPI app through ``TestClient`` with auth toggled on via an
env override, asserting the :class:`~friday.api.middleware.AuthMiddleware`
behaviour:

* with ``require_auth`` on, a missing or malformed ``Authorization`` header ->
  401 JSON ``{"detail": "unauthorized"}``;
* a wrong bearer key -> 401;
* a valid bearer key (in ``settings.api_keys``) -> 200;
* ``/health`` is always exempt, even with auth on;
* with ``require_auth`` off (the default), every route passes through.

Everything is offline: the orchestrator runs against the default in-process
``FakeLLM`` (no network) and ``/chat`` receives a plain conversation turn.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from friday.app import create_app
from friday.config import get_settings


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Pin the offline ``fake`` provider and drop the settings cache per test.

    The developer's real ``.env`` may select the live NVIDIA provider; force
    ``fake`` so every gateway test runs offline with zero network. The app reads
    ``get_settings()`` (an ``lru_cache``) at construction, so each test starts
    from a clean cache and leaves one behind for the next.
    """
    monkeypatch.setenv("FRIDAY_LLM_PROVIDER", "fake")
    monkeypatch.setenv("FRIDAY_LLM_FALLBACK_PROVIDER", "none")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _auth_app(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Build a TestClient for an app with auth required and one valid key."""
    monkeypatch.setenv("FRIDAY_REQUIRE_AUTH", "true")
    monkeypatch.setenv("FRIDAY_API_KEYS", "s3cret, other-key")
    monkeypatch.setenv("FRIDAY_RATE_LIMIT_ENABLED", "false")
    get_settings.cache_clear()
    return TestClient(create_app())


def test_chat_without_token_is_401(monkeypatch: pytest.MonkeyPatch) -> None:
    with _auth_app(monkeypatch) as client:
        resp = client.post("/chat", json={"session_id": "a", "text": "hi"})
    assert resp.status_code == 401
    assert resp.json() == {"detail": "unauthorized"}


def test_chat_with_malformed_header_is_401(monkeypatch: pytest.MonkeyPatch) -> None:
    with _auth_app(monkeypatch) as client:
        resp = client.post(
            "/chat",
            json={"session_id": "a", "text": "hi"},
            headers={"Authorization": "s3cret"},  # missing 'Bearer ' scheme
        )
    assert resp.status_code == 401
    assert resp.json() == {"detail": "unauthorized"}


def test_chat_with_wrong_token_is_401(monkeypatch: pytest.MonkeyPatch) -> None:
    with _auth_app(monkeypatch) as client:
        resp = client.post(
            "/chat",
            json={"session_id": "a", "text": "hi"},
            headers={"Authorization": "Bearer nope"},
        )
    assert resp.status_code == 401
    assert resp.json() == {"detail": "unauthorized"}


def test_chat_with_valid_token_is_200(monkeypatch: pytest.MonkeyPatch) -> None:
    with _auth_app(monkeypatch) as client:
        resp = client.post(
            "/chat",
            json={"session_id": "a", "text": "what's 2+2"},
            headers={"Authorization": "Bearer s3cret"},
        )
    assert resp.status_code == 200
    assert resp.json()["mode"]


def test_health_is_exempt_even_with_auth_on(monkeypatch: pytest.MonkeyPatch) -> None:
    with _auth_app(monkeypatch) as client:
        resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_auth_off_by_default_passes_through(monkeypatch: pytest.MonkeyPatch) -> None:
    # No FRIDAY_REQUIRE_AUTH set -> default False -> open gateway.
    monkeypatch.delenv("FRIDAY_REQUIRE_AUTH", raising=False)
    monkeypatch.setenv("FRIDAY_RATE_LIMIT_ENABLED", "false")
    get_settings.cache_clear()
    with TestClient(create_app()) as client:
        resp = client.post(
            "/chat", json={"session_id": "a", "text": "what's 2+2"}
        )
    assert resp.status_code == 200
