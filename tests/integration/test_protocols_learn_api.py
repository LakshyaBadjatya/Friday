# © Lakshya Badjatya — Author
"""Integration tests for ``POST /protocols/learn`` (macro learning from audit)."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from friday.app import create_app
from friday.config import get_settings


def _make_app(monkeypatch: pytest.MonkeyPatch, *, enabled: bool) -> FastAPI:
    if enabled:
        monkeypatch.setenv("FRIDAY_ENABLE_PROTOCOLS", "true")
    else:
        monkeypatch.delenv("FRIDAY_ENABLE_PROTOCOLS", raising=False)
    monkeypatch.setenv("FRIDAY_LLM_PROVIDER", "fake")
    monkeypatch.setenv("FRIDAY_MEMORY_DB_PATH", ":memory:")
    get_settings.cache_clear()
    return create_app()


def test_learn_404_when_off(monkeypatch: pytest.MonkeyPatch) -> None:
    app = _make_app(monkeypatch, enabled=False)
    with TestClient(app) as client:
        resp = client.post(
            "/protocols/learn", json={"name": "p", "trigger_phrase": "t"}
        )
    assert resp.status_code == 404


def test_learn_400_when_no_history(monkeypatch: pytest.MonkeyPatch) -> None:
    app = _make_app(monkeypatch, enabled=True)
    with TestClient(app) as client:
        resp = client.post(
            "/protocols/learn", json={"name": "p", "trigger_phrase": "t"}
        )
    assert resp.status_code == 400  # nothing in the audit to learn from


def test_learn_creates_disabled_protocol_from_audit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _make_app(monkeypatch, enabled=True)
    with TestClient(app) as client:
        app.state.audit.record(
            correlation_id="c1", tool="notify", args={"text": "hi"}, ok=True,
            error_code=None,
        )
        app.state.audit.record(
            correlation_id="c1", tool="home", args={"device": "lights"}, ok=True,
            error_code=None,
        )
        resp = client.post(
            "/protocols/learn",
            json={"name": "Goodnight", "trigger_phrase": "goodnight"},
        )
    assert resp.status_code == 200
    body = resp.json()
    proto = body["protocol"]
    assert proto["enabled"] is False  # created disabled for review
    assert [s["tool"] for s in proto["steps"]] == ["notify", "home"]
    assert body["has_redacted_args"] is False
