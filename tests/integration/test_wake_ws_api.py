# © Lakshya Badjatya — Author
"""Integration tests for the /ws/wake socket (transcript -> wake/summon event)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from friday.app import create_app
from friday.config import get_settings


def _client(monkeypatch: pytest.MonkeyPatch, *, enabled: bool) -> TestClient:
    monkeypatch.setenv("FRIDAY_LLM_PROVIDER", "fake")
    monkeypatch.setenv("FRIDAY_MEMORY_DB_PATH", ":memory:")
    monkeypatch.setenv("FRIDAY_OWNER_ADDRESS", "Boss")
    if enabled:
        monkeypatch.setenv("FRIDAY_ENABLE_WAKEWORD", "true")
    else:
        monkeypatch.delenv("FRIDAY_ENABLE_WAKEWORD", raising=False)
    get_settings.cache_clear()
    return TestClient(create_app())


def test_wake_ws_refused_when_off(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, enabled=False) as client:
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/ws/wake") as ws:
                ws.receive_json()  # server accepts then closes (policy violation)


def test_wake_ws_emits_wake_and_summon_events(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, enabled=True) as client:
        with client.websocket_connect("/ws/wake") as ws:
            assert ws.receive_json() == {"type": "ready"}

            ws.send_json({"transcript": "hey friday"})
            wake = ws.receive_json()
            assert wake["type"] == "wake"
            assert wake["operator"] == "FRIDAY"
            assert wake["greeting"] == "I'm up, Boss."

            ws.send_json({"transcript": "friday summon vision"})
            summon = ws.receive_json()
            assert summon["type"] == "summon"
            assert summon["operator"] == "VISION"
            assert summon["greeting"] == "VISION here, Boss."
