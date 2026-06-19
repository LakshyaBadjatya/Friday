"""Integration tests for the ``/siri/ask`` Siri-Shortcuts front door.

Offline: the lifespan builds the real app against the in-process ``FakeLLM``,
then each test injects a tiny scripted orchestrator on ``app.state.orchestrator``
(same pattern the studio tests use for ``app.state.studio``) so the spoken-reply
assertions are deterministic and never touch the network.

Covered:
* flag off (default) -> ``404`` (the feature does not exist);
* flag on -> ``200 text/plain`` whose body is the markdown-stripped reply;
* the spoken query reaches the orchestrator unchanged;
* ``?format=json`` -> a ``{speak, text, mode}`` body;
* a missing query -> ``400``;
* an empty reply -> a non-empty spoken fallback (Siri never reads silence);
* with ``require_auth`` on, no/invalid bearer -> ``401``, a valid key -> ``200``.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from friday.app import create_app
from friday.config import get_settings
from friday.core.state import GraphState


@pytest.fixture(autouse=True)
def _offline_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Force the offline ``fake`` provider, drop rate limiting, reset the cache."""
    monkeypatch.setenv("FRIDAY_LLM_PROVIDER", "fake")
    monkeypatch.setenv("FRIDAY_LLM_FALLBACK_PROVIDER", "none")
    monkeypatch.setenv("FRIDAY_RATE_LIMIT_ENABLED", "false")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


class _FakeOrchestrator:
    """Scripted stand-in: records the turn's input and echoes a fixed reply."""

    def __init__(self, reply: str) -> None:
        self._reply = reply
        self.seen: list[str] = []

    async def handle(self, state: GraphState) -> GraphState:
        self.seen.append(state.user_input)
        state.response = self._reply
        return state


def test_siri_ask_disabled_returns_404(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FRIDAY_ENABLE_SIRI", "false")
    get_settings.cache_clear()
    with TestClient(create_app()) as client:
        resp = client.post("/siri/ask?q=hello")
    assert resp.status_code == 404


def test_siri_ask_enabled_speaks_plain_text(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FRIDAY_ENABLE_SIRI", "true")
    get_settings.cache_clear()
    with TestClient(create_app()) as client:
        client.app.state.orchestrator = _FakeOrchestrator(
            "**Hi** boss, all systems green."
        )
        resp = client.post("/siri/ask?q=status report")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/plain")
    assert resp.text == "Hi boss, all systems green."


def test_siri_ask_passes_query_to_orchestrator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FRIDAY_ENABLE_SIRI", "true")
    get_settings.cache_clear()
    fake = _FakeOrchestrator("ok")
    with TestClient(create_app()) as client:
        client.app.state.orchestrator = fake
        client.post("/siri/ask", json={"q": "what's the weather"})
    assert fake.seen == ["what's the weather"]


def test_siri_ask_placeholder_speaks_fix_hint_without_orchestrator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mis-wired shortcut sends a literal label ("Dictated Text"); we short-circuit
    to a fix-it hint and never bother the brain (which would just clarify)."""
    monkeypatch.setenv("FRIDAY_ENABLE_SIRI", "true")
    get_settings.cache_clear()
    fake = _FakeOrchestrator("should not be reached")
    with TestClient(create_app()) as client:
        client.app.state.orchestrator = fake
        resp = client.post("/siri/ask?format=json", json={"q": "Dictated Text"})
    assert resp.status_code == 200
    assert resp.json()["mode"] == "hint"
    assert "Dictated Text variable" in resp.json()["speak"]
    assert fake.seen == []  # the orchestrator was never called


def test_siri_ask_json_format_returns_structured_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FRIDAY_ENABLE_SIRI", "true")
    get_settings.cache_clear()
    with TestClient(create_app()) as client:
        client.app.state.orchestrator = _FakeOrchestrator("All set.")
        resp = client.post("/siri/ask?format=json", json={"q": "hi"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["speak"] == "All set."
    assert body["text"] == "All set."
    assert "mode" in body


def test_siri_ask_who_made_you_names_creator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """'Who made you' is answered instantly with the creator's name, no orchestrator."""
    monkeypatch.setenv("FRIDAY_ENABLE_SIRI", "true")
    get_settings.cache_clear()
    fake = _FakeOrchestrator("should not be reached")
    with TestClient(create_app()) as client:
        client.app.state.orchestrator = fake
        resp = client.post("/siri/ask?format=json", json={"q": "who made you?"})
    assert resp.status_code == 200
    assert "Lakshya Badjatya" in resp.json()["speak"]
    assert resp.json()["mode"] == "identity"
    assert fake.seen == []


def test_siri_ask_missing_query_returns_400(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FRIDAY_ENABLE_SIRI", "true")
    get_settings.cache_clear()
    with TestClient(create_app()) as client:
        client.app.state.orchestrator = _FakeOrchestrator("unused")
        resp = client.post("/siri/ask")
    assert resp.status_code == 400


def test_siri_ask_empty_reply_speaks_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FRIDAY_ENABLE_SIRI", "true")
    get_settings.cache_clear()
    with TestClient(create_app()) as client:
        client.app.state.orchestrator = _FakeOrchestrator("")
        resp = client.post("/siri/ask?q=hello")
    assert resp.status_code == 200
    assert resp.text.strip() != ""


def test_siri_ask_requires_bearer_when_auth_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FRIDAY_ENABLE_SIRI", "true")
    monkeypatch.setenv("FRIDAY_REQUIRE_AUTH", "true")
    monkeypatch.setenv("FRIDAY_API_KEYS", "s3cret")
    get_settings.cache_clear()
    with TestClient(create_app()) as client:
        client.app.state.orchestrator = _FakeOrchestrator("hi")
        no_key = client.post("/siri/ask?q=hello")
        with_key = client.post(
            "/siri/ask?q=hello", headers={"Authorization": "Bearer s3cret"}
        )
    assert no_key.status_code == 401
    assert with_key.status_code == 200
    assert with_key.text == "hi"
