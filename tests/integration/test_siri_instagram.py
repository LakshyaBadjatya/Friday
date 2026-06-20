"""Integration tests for Instagram DM intents over the /siri/ask route.

Offline: the lifespan builds the app, then each test wires an ``InstagramService``
over a fake client onto ``app.state.instagram`` and a scripted orchestrator onto
``app.state.orchestrator``. Proves the end-to-end loop (count / read-aloud / bare
"read them aloud" follow-up / reply), that a non-Instagram phrase falls through to
the orchestrator, and that with the flag off the path is skipped entirely.

The env fixture clears the ``get_settings`` cache before AND after (a prior test's
cached auth-on Settings leaks otherwise — a known gotcha in this repo).
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from friday.app import create_app
from friday.config import get_settings
from friday.core.state import GraphState
from friday.instagram.models import IgMessage, IgThread
from friday.instagram.service import InstagramService


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("FRIDAY_LLM_PROVIDER", "fake")
    monkeypatch.setenv("FRIDAY_LLM_FALLBACK_PROVIDER", "none")
    monkeypatch.setenv("FRIDAY_RATE_LIMIT_ENABLED", "false")
    monkeypatch.setenv("FRIDAY_ENABLE_SIRI", "true")
    monkeypatch.setenv("FRIDAY_ENABLE_INSTAGRAM_DMS", "true")
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


class _FakeInstagramClient:
    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []
        self._unread = [
            IgThread(thread_id="1", full_name="Rahul Mehta", unread_count=2),
            IgThread(thread_id="2", username="priya_k", unread_count=1),
        ]
        self._recent = [
            IgThread(thread_id="9", full_name="Rahul Mehta", username="rahul_m"),
        ]
        self._messages = {
            "1": [
                IgMessage(message_id="a", text="hey are you free"),
                IgMessage(message_id="b", text="call me when you can"),
            ],
            "2": [IgMessage(message_id="c", text="dinner tonight?")],
        }

    def unread_threads(self) -> list[IgThread]:
        return self._unread

    def recent_threads(self, limit: int) -> list[IgThread]:
        return self._recent

    def thread_messages(self, thread_id: str, limit: int) -> list[IgMessage]:
        return self._messages.get(thread_id, [])[:limit]

    def send_dm(self, thread_id: str, text: str) -> bool:
        self.sent.append((thread_id, text))
        return True


def _wire(app: FastAPI) -> _FakeInstagramClient:
    client = _FakeInstagramClient()
    app.state.instagram = InstagramService(client, read_aloud_limit=5)
    app.state.orchestrator = _FakeOrchestrator()
    return client


def test_count_over_siri() -> None:
    with TestClient(create_app()) as client:
        _wire(client.app)
        r = client.post("/siri/ask?q=any instagram dms for me")
    assert r.status_code == 200
    assert "3 unread Instagram DMs" in r.text
    assert "Rahul Mehta" in r.text


def test_read_aloud_over_siri() -> None:
    with TestClient(create_app()) as client:
        _wire(client.app)
        r = client.post("/siri/ask?q=read my instagram messages")
    assert r.status_code == 200
    assert "From Rahul Mehta: hey are you free." in r.text
    assert "From Rahul Mehta: call me when you can." in r.text  # both of Rahul's 2 unread
    # for_speech strips the markdown underscore from the @username.
    assert "From @priyak: dinner tonight?." in r.text


def test_bare_read_follows_a_count() -> None:
    # Same TestClient → app.state persists the Instagram marker across requests.
    with TestClient(create_app()) as client:
        _wire(client.app)
        first = client.post("/siri/ask?q=any instagram dms")
        assert first.status_code == 200
        second = client.post("/siri/ask?q=read them aloud")
    assert second.status_code == 200
    assert "From Rahul Mehta: hey are you free." in second.text


def test_bare_read_without_a_prior_count_falls_through() -> None:
    with TestClient(create_app()) as client:
        _wire(client.app)
        fake = client.app.state.orchestrator
        r = client.post("/siri/ask?q=read them aloud")
    assert r.status_code == 200
    assert r.text == "general answer"
    assert fake.calls == ["read them aloud"]


def test_reply_over_siri() -> None:
    with TestClient(create_app()) as client:
        ig = _wire(client.app)
        r = client.post("/siri/ask?q=reply to rahul on instagram saying on my way")
    assert r.status_code == 200
    assert r.text == "Sent to Rahul Mehta on Instagram."
    assert ig.sent == [("9", "on my way")]


def test_non_instagram_query_falls_through_to_orchestrator() -> None:
    with TestClient(create_app()) as client:
        _wire(client.app)
        fake = client.app.state.orchestrator
        r = client.post("/siri/ask?q=tell me a joke")
    assert r.status_code == 200
    assert r.text == "general answer"
    assert fake.calls == ["tell me a joke"]


def test_flag_off_falls_through(monkeypatch: pytest.MonkeyPatch) -> None:
    # Re-disable the flag (the autouse fixture turned it on) so the path is skipped.
    monkeypatch.setenv("FRIDAY_ENABLE_INSTAGRAM_DMS", "false")
    get_settings.cache_clear()
    with TestClient(create_app()) as client:
        fake = _FakeOrchestrator()
        client.app.state.orchestrator = fake
        # No app.state.instagram wired; with the flag off, the builder returns None.
        r = client.post("/siri/ask?q=any instagram dms for me")
    assert r.status_code == 200
    assert r.text == "general answer"
    assert fake.calls == ["any instagram dms for me"]
