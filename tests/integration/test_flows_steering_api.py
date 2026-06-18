# © Lakshya Badjatya — Author
"""Integration tests for the Flow Engine steering + template HTTP surface.

Exercises the Phase 3/4 routes end-to-end over the real ``create_app`` wiring:
cancel / simulate steering and the ``/flow-templates`` save → list → run
lifecycle, plus 404 when the feature is off.
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


def test_steering_routes_404_when_off(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, enabled=False) as client:
        assert client.post("/flows/x/cancel").status_code == 404
        assert client.post("/flows/x/simulate").status_code == 404
        assert client.get("/flow-templates").status_code == 404


def test_cancel_and_simulate(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, enabled=True) as client:
        flow_id = client.post("/flows", json={"goal": "do a thing"}).json()["id"]

        cancelled = client.post(f"/flows/{flow_id}/cancel")
        assert cancelled.status_code == 200
        assert cancelled.json()["status"] == "cancelled"

        other = client.post("/flows", json={"goal": "irreversible send"}).json()["id"]
        sim = client.post(f"/flows/{other}/simulate")
        assert sim.status_code == 200
        assert sim.json()["status"] in {"succeeded", "failed", "needs_confirmation"}

        assert client.post("/flows/nope/cancel").status_code == 404


def test_template_lifecycle(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, enabled=True) as client:
        saved = client.post(
            "/flow-templates",
            json={
                "name": "brief",
                "goal": "brief {topic}",
                "steps": [{"id": "s1", "description": "summarize", "kind": "reason"}],
            },
        )
        assert saved.status_code == 200
        assert saved.json()["name"] == "brief"

        listed = client.get("/flow-templates")
        assert listed.status_code == 200
        assert any(t["name"] == "brief" for t in listed.json()["templates"])

        run = client.post("/flow-templates/brief/run", json={"params": {"topic": "ai"}})
        assert run.status_code == 200
        assert run.json()["template"] == "brief"
        assert run.json()["goal"] == "brief ai"

        assert client.post("/flow-templates/missing/run", json={}).status_code == 404
