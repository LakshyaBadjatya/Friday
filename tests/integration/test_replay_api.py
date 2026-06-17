# © Lakshya Badjatya — Author
"""Integration tests for the turn-replay admin API.

Offline: a scripted :class:`FakeLLM` orchestrator sharing the ``app.state`` turn
recorder the lifespan built, so a real ``POST /chat`` lands a transcript that
``GET /admin/turns`` and ``GET /admin/turns/{id}`` read back.

Each test pins a clean offline ``Settings`` (fake LLM, in-memory DB) and clears
the ``get_settings`` cache before ``create_app``, so the slice is hermetic
regardless of any prior test's cached settings (e.g. an auth-enabled build).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from friday.app import create_app
from friday.config import get_settings
from friday.core.orchestrator import Orchestrator
from friday.providers.llm import FakeLLM, LLMResponse, Usage

PERSONA_PATH = (
    Path(__file__).resolve().parents[2] / "src" / "friday" / "persona" / "friday.md"
)


def _app(monkeypatch: pytest.MonkeyPatch) -> FastAPI:
    """Build an app over pinned, pristine offline settings (no cache leakage)."""
    monkeypatch.setenv("FRIDAY_LLM_PROVIDER", "fake")
    monkeypatch.setenv("FRIDAY_MEMORY_DB_PATH", ":memory:")
    get_settings.cache_clear()
    app = create_app()
    return app


def _wire_recording_orchestrator(app: FastAPI) -> None:
    """Wire a FakeLLM orchestrator that shares app.state's turn recorder."""
    llm = FakeLLM(
        responses=[LLMResponse(text="Four, Boss.", tool_calls=[], usage=Usage())]
    )
    app.state.orchestrator = Orchestrator(
        llm=llm,
        registry=app.state.registry,
        memory=app.state.short_term,
        persona_path=PERSONA_PATH,
        tracer=app.state.tracer,
        metrics=app.state.metrics,
        turn_recorder=app.state.turns,
    )


def test_turns_empty_before_any_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    with TestClient(_app(monkeypatch)) as client:
        resp = client.get("/admin/turns")
        assert resp.status_code == 200
        assert resp.json() == {"turns": []}
    get_settings.cache_clear()


def test_turn_is_recorded_and_fetchable_by_id(monkeypatch: pytest.MonkeyPatch) -> None:
    app = _app(monkeypatch)
    with TestClient(app) as client:
        _wire_recording_orchestrator(app)
        chat = client.post("/chat", json={"session_id": "replay-1", "text": "what's 2+2"})
        assert chat.status_code == 200

        listing = client.get("/admin/turns").json()["turns"]
        assert len(listing) >= 1
        latest = listing[-1]
        assert latest["session_id"] == "replay-1"
        assert latest["user_input"] == "what's 2+2"
        assert latest["response"] == "Four, Boss."
        assert latest["mode"]  # a concrete mode string was stamped

        one = client.get(f"/admin/turns/{latest['id']}")
        assert one.status_code == 200
        assert one.json() == latest
    get_settings.cache_clear()


def test_unknown_turn_id_is_404(monkeypatch: pytest.MonkeyPatch) -> None:
    with TestClient(_app(monkeypatch)) as client:
        resp = client.get("/admin/turns/999999")
        assert resp.status_code == 404
        assert resp.json()["type"] == "TurnNotFound"
    get_settings.cache_clear()
