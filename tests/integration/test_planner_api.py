# © Lakshya Badjatya — Author
"""Integration tests for the ``/planner`` task-decomposition surface (flag-gated)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from friday.app import create_app
from friday.config import get_settings


def _client(monkeypatch: pytest.MonkeyPatch, *, enabled: bool) -> TestClient:
    """Build an offline app with the planner flag set as requested."""
    if enabled:
        monkeypatch.setenv("FRIDAY_ENABLE_PLANNER", "true")
    else:
        monkeypatch.delenv("FRIDAY_ENABLE_PLANNER", raising=False)
    monkeypatch.setenv("FRIDAY_LLM_PROVIDER", "fake")
    monkeypatch.setenv("FRIDAY_MEMORY_DB_PATH", ":memory:")
    get_settings.cache_clear()
    return TestClient(create_app())


def test_planner_404_when_off(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, enabled=False) as client:
        resp = client.post("/planner/plan", json={"goal": "x"})
    assert resp.status_code == 404
    assert resp.json()["detail"] == "planner disabled"


def test_planner_reachable_when_on(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, enabled=True) as client:
        resp = client.post("/planner/plan", json={"goal": "research then notify"})
    assert resp.status_code == 200
    body = resp.json()
    assert "plan" in body and "rendered" in body
    # Offline FakeLLM can't decompose, so the planner degrades to a single-step
    # plan restating the goal — the non-fatal fallback, wired end-to-end.
    assert len(body["plan"]["steps"]) >= 1
    assert "research then notify" in body["rendered"]
