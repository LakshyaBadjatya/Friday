# © Lakshya Badjatya — Author
"""Integration tests for the ``/ensemble`` debate surface (flag-gated)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from friday.app import create_app
from friday.config import get_settings


def _client(monkeypatch: pytest.MonkeyPatch, *, enabled: bool) -> TestClient:
    """Build an offline app with the ensemble flag set as requested."""
    if enabled:
        monkeypatch.setenv("FRIDAY_ENABLE_ENSEMBLE", "true")
    else:
        monkeypatch.delenv("FRIDAY_ENABLE_ENSEMBLE", raising=False)
    monkeypatch.setenv("FRIDAY_LLM_PROVIDER", "fake")
    monkeypatch.setenv("FRIDAY_MEMORY_DB_PATH", ":memory:")
    get_settings.cache_clear()
    return TestClient(create_app())


def test_ensemble_404_when_off(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, enabled=False) as client:
        resp = client.post("/ensemble/debate", json={"question": "x"})
    assert resp.status_code == 404
    assert resp.json()["detail"] == "ensemble disabled"


def test_ensemble_reachable_when_on(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, enabled=True) as client:
        resp = client.post("/ensemble/debate", json={"question": "What's the plan?"})
    assert resp.status_code == 200
    body = resp.json()
    assert {"drafts", "synthesis", "contributors"} <= body.keys()
    # Offline FakeLLM has no scripted replies, so every draft fails gracefully
    # (captured, not raised) — proving the non-fatal path is wired end-to-end.
    assert all(d["ok"] is False for d in body["drafts"])


def test_ensemble_unknown_operators_400(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, enabled=True) as client:
        resp = client.post(
            "/ensemble/debate", json={"question": "x", "operators": ["NOBODY"]}
        )
    assert resp.status_code == 400
