"""Liveness ``GET /health`` and ``/chat`` input-validation tests (no network).

The health endpoint is a cheap liveness probe: it must report the configured LLM
provider and model *without* making any LLM call. The default app uses the
:class:`FakeLLM` path (``FRIDAY_LLM_PROVIDER=fake``), so a ``TestClient`` boots
fully offline. The validation tests assert FastAPI's automatic 422 for an empty
or oversized ``text`` / ``session_id``.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from friday.app import create_app
from friday.config import get_settings

_LEAKY_ENV_PREFIXES = ("FRIDAY_", "NVIDIA_")


@pytest.fixture
def fake_provider_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Force the offline FakeLLM path regardless of the developer's ``.env``.

    Strips ``FRIDAY_*``/``NVIDIA_*`` from the environment, pins the provider to
    ``fake``, and clears the cached settings so ``create_app`` boots without
    network or credentials. The cache is cleared again on teardown so other
    tests see fresh settings.
    """
    for key in list(os.environ):
        if key.upper().startswith(_LEAKY_ENV_PREFIXES):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("FRIDAY_LLM_PROVIDER", "fake")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_health_returns_ok_with_provider(fake_provider_env: None) -> None:
    app = create_app()
    with TestClient(app) as client:
        resp = client.get("/health")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    # Default config is the offline FakeLLM path.
    assert body["llm_provider"] == "fake"
    # The model key is always present (null for the fake path is acceptable, but
    # the field must exist).
    assert "model" in body


def test_health_makes_no_llm_call(fake_provider_env: None) -> None:
    # An empty-script FakeLLM raises ProviderError if invoked; the default app
    # uses exactly that, so a successful /health proves no LLM call happened.
    app = create_app()
    with TestClient(app) as client:
        resp = client.get("/health")
    assert resp.status_code == 200


def test_chat_rejects_empty_text() -> None:
    app = create_app()
    with TestClient(app) as client:
        resp = client.post("/chat", json={"session_id": "s1", "text": ""})
    assert resp.status_code == 422


def test_chat_rejects_empty_session_id() -> None:
    app = create_app()
    with TestClient(app) as client:
        resp = client.post("/chat", json={"session_id": "", "text": "hi"})
    assert resp.status_code == 422


def test_chat_rejects_oversized_text() -> None:
    app = create_app()
    with TestClient(app) as client:
        resp = client.post(
            "/chat", json={"session_id": "s1", "text": "x" * 8001}
        )
    assert resp.status_code == 422
