# © Lakshya Badjatya — Author
"""Integration tests for the ``GET /export`` second-brain Markdown surface."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from friday.app import create_app
from friday.config import get_settings


def _make_app(monkeypatch: pytest.MonkeyPatch, *, enabled: bool) -> FastAPI:
    if enabled:
        monkeypatch.setenv("FRIDAY_ENABLE_KB_EXPORT", "true")
    else:
        monkeypatch.delenv("FRIDAY_ENABLE_KB_EXPORT", raising=False)
    monkeypatch.setenv("FRIDAY_LLM_PROVIDER", "fake")
    monkeypatch.setenv("FRIDAY_MEMORY_DB_PATH", ":memory:")
    get_settings.cache_clear()
    return create_app()


def test_export_404_when_off(monkeypatch: pytest.MonkeyPatch) -> None:
    app = _make_app(monkeypatch, enabled=False)
    with TestClient(app) as client:
        resp = client.get("/export")
    assert resp.status_code == 404


def test_export_renders_facts_as_markdown(monkeypatch: pytest.MonkeyPatch) -> None:
    app = _make_app(monkeypatch, enabled=True)
    with TestClient(app) as client:
        app.state.long_term.add_fact("the sky is blue", "src1")
        app.state.long_term.add_fact("water is wet", "src2")
        resp = client.get("/export")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/markdown")
    body = resp.text
    assert "# FRIDAY Knowledge Export" in body
    assert "the sky is blue (src1)" in body
    assert "water is wet (src2)" in body


def test_export_empty_when_no_facts(monkeypatch: pytest.MonkeyPatch) -> None:
    app = _make_app(monkeypatch, enabled=True)
    with TestClient(app) as client:
        resp = client.get("/export")
    assert resp.status_code == 200
    assert "# FRIDAY Knowledge Export" in resp.text
