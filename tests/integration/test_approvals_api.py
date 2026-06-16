# © Lakshya Badjatya — Author
"""Integration tests for the /approvals workflow surface (flag-gated)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from friday.app import create_app
from friday.config import get_settings


def _client(monkeypatch: pytest.MonkeyPatch, *, enabled: bool) -> TestClient:
    monkeypatch.setenv("FRIDAY_LLM_PROVIDER", "fake")
    monkeypatch.setenv("FRIDAY_MEMORY_DB_PATH", ":memory:")
    if enabled:
        monkeypatch.setenv("FRIDAY_ENABLE_APPROVALS", "true")
    else:
        monkeypatch.delenv("FRIDAY_ENABLE_APPROVALS", raising=False)
    get_settings.cache_clear()
    return TestClient(create_app())


def test_approvals_404_when_off(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, enabled=False) as client:
        assert client.post("/approvals", json={"action": "x"}).status_code == 404


def test_create_pending_then_approve_is_one_shot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _client(monkeypatch, enabled=True) as client:
        created = client.post("/approvals", json={"action": "wire $500"})
        assert created.status_code == 200
        approval_id = created.json()["id"]
        assert created.json()["status"] == "pending"

        # It shows up as pending.
        pending = client.get("/approvals").json()["pending"]
        assert any(p["id"] == approval_id for p in pending)

        # Approve it once...
        approved = client.post(f"/approvals/{approval_id}/approve")
        assert approved.status_code == 200
        assert approved.json()["status"] == "approved"

        # ...and a second decision is rejected (one-shot).
        assert client.post(f"/approvals/{approval_id}/deny").status_code == 409


def test_deny(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, enabled=True) as client:
        approval_id = client.post(
            "/approvals", json={"action": "delete everything"}
        ).json()["id"]
        resp = client.post(f"/approvals/{approval_id}/deny")
    assert resp.status_code == 200
    assert resp.json()["status"] == "denied"


def test_unknown_approval_404(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, enabled=True) as client:
        assert client.post("/approvals/ghost/approve").status_code == 404
