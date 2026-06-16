# © Lakshya Badjatya — Author
"""Integration tests for the ``/memory/contradiction`` check surface (flag-gated)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from friday.app import create_app
from friday.config import get_settings


def _client(monkeypatch: pytest.MonkeyPatch, *, enabled: bool) -> TestClient:
    if enabled:
        monkeypatch.setenv("FRIDAY_ENABLE_CONTRADICTION", "true")
    else:
        monkeypatch.delenv("FRIDAY_ENABLE_CONTRADICTION", raising=False)
    monkeypatch.setenv("FRIDAY_LLM_PROVIDER", "fake")
    monkeypatch.setenv("FRIDAY_MEMORY_DB_PATH", ":memory:")
    get_settings.cache_clear()
    return TestClient(create_app())


def test_contradiction_404_when_off(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, enabled=False) as client:
        resp = client.post("/memory/contradiction", json={"fact": "the sky is blue"})
    assert resp.status_code == 404
    assert resp.json()["detail"] == "contradiction check disabled"


def test_contradiction_reachable_when_on(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, enabled=True) as client:
        resp = client.post("/memory/contradiction", json={"fact": "the sky is blue"})
    assert resp.status_code == 200
    body = resp.json()
    assert {"contradicts", "conflicting_source_id", "explanation"} <= body.keys()
    # Offline + empty store: no stored facts to conflict with -> no contradiction.
    assert body["contradicts"] is False


def test_contradiction_validates_body(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, enabled=True) as client:
        resp = client.post("/memory/contradiction", json={"fact": ""})
    assert resp.status_code == 422  # empty fact rejected by the model


def _tag_client(monkeypatch: pytest.MonkeyPatch, *, enabled: bool) -> TestClient:
    if enabled:
        monkeypatch.setenv("FRIDAY_ENABLE_AUTOTAG", "true")
    else:
        monkeypatch.delenv("FRIDAY_ENABLE_AUTOTAG", raising=False)
    monkeypatch.setenv("FRIDAY_LLM_PROVIDER", "fake")
    monkeypatch.setenv("FRIDAY_MEMORY_DB_PATH", ":memory:")
    get_settings.cache_clear()
    return TestClient(create_app())


def test_tag_404_when_off(monkeypatch: pytest.MonkeyPatch) -> None:
    with _tag_client(monkeypatch, enabled=False) as client:
        resp = client.post("/memory/tag", json={"text": "a note about AI"})
    assert resp.status_code == 404
    assert resp.json()["detail"] == "auto-tagging disabled"


def test_tag_reachable_when_on(monkeypatch: pytest.MonkeyPatch) -> None:
    with _tag_client(monkeypatch, enabled=True) as client:
        resp = client.post("/memory/tag", json={"text": "a note about AI"})
    assert resp.status_code == 200
    # Offline FakeLLM has no scripted reply, so tagging degrades to [] (non-fatal).
    assert resp.json()["tags"] == []
