"""Integration tests for the ``/tv`` surface (offline, fake LLM)."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from friday.app import create_app
from friday.config import get_settings
from friday.core.state import GraphState
from friday.tv.intents import parse_tv_command


@pytest.fixture(autouse=True)
def _offline_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("FRIDAY_LLM_PROVIDER", "fake")
    monkeypatch.setenv("FRIDAY_LLM_FALLBACK_PROVIDER", "none")
    monkeypatch.setenv("FRIDAY_RATE_LIMIT_ENABLED", "false")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


class _FakeOrchestrator:
    def __init__(self, reply: str) -> None:
        self._reply = reply

    async def handle(self, state: GraphState) -> GraphState:
        state.response = self._reply
        return state


def test_tv_disabled_returns_404(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FRIDAY_ENABLE_TV", "false")
    get_settings.cache_clear()
    with TestClient(create_app()) as client:
        assert client.post("/tv/ask?q=open youtube").status_code == 404


def test_tv_ask_command_returns_action(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FRIDAY_ENABLE_TV", "true")
    get_settings.cache_clear()
    with TestClient(create_app()) as client:
        resp = client.post("/tv/ask?q=open youtube")
    assert resp.status_code == 200
    body = resp.json()
    assert body["action"]["type"] == "open_app"
    assert body["action"]["app"] == "youtube"
    assert body["speak"] == "Opening youtube."


def test_tv_ask_chitchat_uses_orchestrator_and_null_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FRIDAY_ENABLE_TV", "true")
    get_settings.cache_clear()
    with TestClient(create_app()) as client:
        client.app.state.orchestrator = _FakeOrchestrator("It is sunny, boss.")
        resp = client.post("/tv/ask?q=what is the weather")
    body = resp.json()
    assert body["action"] is None
    assert body["speak"] == "It is sunny, boss."


def test_tv_pair_then_command_then_poll(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FRIDAY_ENABLE_TV", "true")
    get_settings.cache_clear()
    with TestClient(create_app()) as client:
        device_id = client.post("/tv/pair", json={"name": "Living Room"}).json()[
            "device_id"
        ]
        queued = client.post(
            "/tv/command", json={"device_id": device_id, "text": "play lofi on youtube"}
        )
        assert queued.json()["queued"] is True
        drained = client.get(f"/tv/poll?device_id={device_id}").json()
    assert len(drained["actions"]) == 1
    assert drained["actions"][0]["type"] == "play"
    assert drained["actions"][0]["query"] == "lofi"


def test_tv_ask_missing_query_returns_400(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FRIDAY_ENABLE_TV", "true")
    get_settings.cache_clear()
    with TestClient(create_app()) as client:
        assert client.post("/tv/ask").status_code == 400


def test_tv_stream_pushes_enqueued_action(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FRIDAY_ENABLE_TV", "true")
    get_settings.cache_clear()
    with TestClient(create_app()) as client:
        device_id = client.post("/tv/pair", json={"name": "TV"}).json()["device_id"]
        # Enqueue before connecting so the action is waiting on the queue.
        action = parse_tv_command("open youtube")
        assert action is not None
        client.app.state.tv_relay.enqueue(device_id, action)
        with client.websocket_connect(f"/tv/stream?device_id={device_id}") as ws:
            msg = ws.receive_json()
    assert msg["type"] == "open_app"
    assert msg["app"] == "youtube"


def test_siri_on_the_tv_routes_to_paired_device(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FRIDAY_ENABLE_TV", "true")
    monkeypatch.setenv("FRIDAY_ENABLE_SIRI", "true")
    get_settings.cache_clear()
    with TestClient(create_app()) as client:
        device_id = client.post("/tv/pair", json={"name": "TV"}).json()["device_id"]
        resp = client.post("/siri/ask?q=play lofi beats on the TV&format=json")
        body = resp.json()
        assert body["action"]["type"] == "play"
        assert body["action"]["query"] == "lofi beats"
        drained = client.get(f"/tv/poll?device_id={device_id}").json()["actions"]
    assert len(drained) == 1
    assert drained[0]["query"] == "lofi beats"
