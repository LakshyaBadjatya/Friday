# © Lakshya Badjatya — Author
"""Integration tests for the flagged ``/sentiment`` route (Wave C; default off)."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import friday.api.routes_sentiment as routes_sentiment
from friday.api.routes_sentiment import router as sentiment_router
from friday.config import Settings


def _app() -> FastAPI:
    app = FastAPI()
    app.include_router(sentiment_router)
    return app


def _enabled() -> Settings:
    return Settings(_env_file=None, enable_sentiment=True)


def _disabled() -> Settings:
    return Settings(_env_file=None, enable_sentiment=False)


def test_sentiment_disabled_returns_404(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(routes_sentiment, "get_settings", _disabled)
    with TestClient(_app()) as client:
        resp = client.post("/sentiment", json={"text": "this is great"})
    assert resp.status_code == 404


def test_sentiment_enabled_scores_text(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(routes_sentiment, "get_settings", _enabled)
    with TestClient(_app()) as client:
        resp = client.post("/sentiment", json={"text": "I love this, works perfectly!"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["label"] == "positive"
    assert body["score"] > 0.1
    assert "love" in body["positive_hits"]


def test_sentiment_validates_empty_text(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(routes_sentiment, "get_settings", _enabled)
    with TestClient(_app()) as client:
        resp = client.post("/sentiment", json={"text": ""})
    assert resp.status_code == 422  # min_length=1
