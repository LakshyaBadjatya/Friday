# © Lakshya Badjatya — Author
"""Integration tests for the ``/flows`` Flow Engine surface (flag-gated).

Verifies the route is ``404`` when off, and that the full lifecycle works when on:
plan → get → run → list → events, end-to-end through the real ``create_app`` wiring
(offline ``FakeLLM`` + an in-memory flow store).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from friday.app import create_app
from friday.config import get_settings


def _client(monkeypatch: pytest.MonkeyPatch, *, enabled: bool) -> TestClient:
    if enabled:
        monkeypatch.setenv("FRIDAY_ENABLE_FLOWS", "true")
    else:
        monkeypatch.delenv("FRIDAY_ENABLE_FLOWS", raising=False)
    monkeypatch.setenv("FRIDAY_LLM_PROVIDER", "fake")
    monkeypatch.setenv("FRIDAY_MEMORY_DB_PATH", ":memory:")
    monkeypatch.setenv("FRIDAY_FLOW_DB_PATH", ":memory:")
    get_settings.cache_clear()
    return TestClient(create_app())


def test_flows_404_when_off(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, enabled=False) as client:
        resp = client.post("/flows", json={"goal": "x"})
    assert resp.status_code == 404
    assert resp.json()["detail"] == "flows disabled"


def test_flow_lifecycle_when_on(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, enabled=True) as client:
        created = client.post("/flows", json={"goal": "research then write"})
        assert created.status_code == 200
        flow = created.json()
        flow_id = flow["id"]
        assert flow["status"] == "planned"
        assert len(flow["steps"]) >= 1

        # The flow is retrievable and listed.
        got = client.get(f"/flows/{flow_id}")
        assert got.status_code == 200
        assert got.json()["id"] == flow_id

        listed = client.get("/flows")
        assert listed.status_code == 200
        assert any(f["id"] == flow_id for f in listed.json()["flows"])

        # Running drives it to a terminal/paused state (offline FakeLLM may make a
        # reason step succeed or fail — either way the engine reaches a real,
        # honest state, never half-completes silently).
        ran = client.post(f"/flows/{flow_id}/run", json={"confirmed": False})
        assert ran.status_code == 200
        assert ran.json()["status"] in {
            "succeeded", "failed", "needs_confirmation", "awaiting_approval",
        }

        # Every transition was audited; the 'planned' event is always present.
        events = client.get(f"/flows/{flow_id}/events")
        assert events.status_code == 200
        kinds = [e["kind"] for e in events.json()["events"]]
        assert "planned" in kinds


def test_run_missing_flow_404(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, enabled=True) as client:
        resp = client.post("/flows/nope/run", json={"confirmed": False})
    assert resp.status_code == 404
    assert resp.json()["detail"] == "flow not found"
