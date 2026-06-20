"""Integration tests for circle status intents over the /siri/ask route.

Offline: the lifespan builds the app, then each test injects an in-memory circle,
a status service, a token->uid map, and a scripted orchestrator on ``app.state``.
Proves the end-to-end loop ("set my status…", "what's X doing") and that
non-circle phrasing / unknown callers fall through to the general assistant.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from friday.app import create_app
from friday.circle.service import CircleService
from friday.circle.status import InMemoryStatusStore, StatusService
from friday.circle.store import InMemoryCircleStore
from friday.config import get_settings
from friday.core.state import GraphState

NOW = datetime(2026, 6, 18, 17, 0, tzinfo=UTC)


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("FRIDAY_LLM_PROVIDER", "fake")
    monkeypatch.setenv("FRIDAY_LLM_FALLBACK_PROVIDER", "none")
    monkeypatch.setenv("FRIDAY_RATE_LIMIT_ENABLED", "false")
    monkeypatch.setenv("FRIDAY_ENABLE_SIRI", "true")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


class _FakeOrchestrator:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def handle(self, state: GraphState) -> GraphState:
        self.calls.append(state.user_input)
        state.response = "general answer"
        return state


def _wire(app: FastAPI) -> None:
    circle = CircleService(InMemoryCircleStore())
    circle.create_group(
        name="Us",
        admin_uid="u-india",
        admin_display_name="Me",
        admin_tz="Asia/Kolkata",
        now=NOW,
        group_id="g1",
    )
    circle.accept_invite(
        code=circle.invite(group_id="g1", by_uid="u-india", now=NOW).code,
        uid="u-us",
        display_name="Bestie",
        tz="America/New_York",
        now=NOW,
    )
    app.state.circle = circle
    app.state.circle_status = StatusService(circle, InMemoryStatusStore())
    app.state.siri_identities = {"tok-india": "u-india", "tok-us": "u-us"}


def test_set_status_then_query_over_siri() -> None:
    with TestClient(create_app()) as client:
        _wire(client.app)
        client.app.state.orchestrator = _FakeOrchestrator()
        r1 = client.post(
            "/siri/ask?q=set my status to having lunch",
            headers={"Authorization": "Bearer tok-us"},
        )
        assert r1.status_code == 200
        assert "lunch" in r1.text.lower()
        r2 = client.post(
            "/siri/ask?q=what's Bestie doing",
            headers={"Authorization": "Bearer tok-india"},
        )
    assert r2.status_code == 200
    assert "Bestie" in r2.text
    assert "having lunch" in r2.text


def test_non_circle_query_falls_through_to_orchestrator() -> None:
    with TestClient(create_app()) as client:
        _wire(client.app)
        fake = _FakeOrchestrator()
        client.app.state.orchestrator = fake
        r = client.post(
            "/siri/ask?q=tell me a joke",
            headers={"Authorization": "Bearer tok-india"},
        )
    assert r.status_code == 200
    assert r.text == "general answer"
    assert fake.calls == ["tell me a joke"]


def test_unknown_caller_falls_through() -> None:
    with TestClient(create_app()) as client:
        _wire(client.app)
        fake = _FakeOrchestrator()
        client.app.state.orchestrator = fake
        r = client.post(
            "/siri/ask?q=what's Bestie doing",
            headers={"Authorization": "Bearer nope"},
        )
    assert r.status_code == 200
    assert r.text == "general answer"
    assert fake.calls == ["what's Bestie doing"]
