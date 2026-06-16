# © Lakshya Badjatya — Author
"""Integration tests for the /rules and /watchers evaluation surfaces."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from friday.app import create_app
from friday.config import get_settings


def _client(monkeypatch: pytest.MonkeyPatch, **flags: str) -> TestClient:
    monkeypatch.setenv("FRIDAY_LLM_PROVIDER", "fake")
    monkeypatch.setenv("FRIDAY_MEMORY_DB_PATH", ":memory:")
    for key, value in flags.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()
    return TestClient(create_app())


def test_rules_404_when_off(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch) as client:
        resp = client.post("/rules/evaluate", json={"event": "e", "rules": []})
    assert resp.status_code == 404


def test_rules_evaluate_fires_matching_rule(monkeypatch: pytest.MonkeyPatch) -> None:
    rule = {
        "name": "hot",
        "trigger": "metric",
        "action": {"name": "notify", "args": {"to": "owner"}},
        "condition": {"field": "cpu", "op": "gt", "value": 90},
    }
    with _client(monkeypatch, FRIDAY_ENABLE_RULES="true") as client:
        resp = client.post(
            "/rules/evaluate",
            json={"event": "metric", "payload": {"cpu": 95}, "rules": [rule]},
        )
    assert resp.status_code == 200
    fired = resp.json()["fired"]
    assert [f["rule"] for f in fired] == ["hot"]
    assert fired[0]["action"]["name"] == "notify"


def test_watchers_404_when_off(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch) as client:
        resp = client.post("/watchers/conflicts", json={"events": []})
    assert resp.status_code == 404


def test_watchers_conflicts(monkeypatch: pytest.MonkeyPatch) -> None:
    events = [
        {"title": "a", "start": 0, "end": 10},
        {"title": "b", "start": 5, "end": 15},
        {"title": "c", "start": 20, "end": 30},
    ]
    with _client(monkeypatch, FRIDAY_ENABLE_WATCHERS="true") as client:
        resp = client.post("/watchers/conflicts", json={"events": events})
    assert resp.status_code == 200
    conflicts = resp.json()["conflicts"]
    assert [(c["a"], c["b"]) for c in conflicts] == [("a", "b")]


def test_watchers_price(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, FRIDAY_ENABLE_WATCHERS="true") as client:
        resp = client.post("/watchers/price", json={"price": 105, "above": 100})
    assert resp.status_code == 200
    assert resp.json()["breach"] is True
