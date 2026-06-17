"""/ws/emotion live-stream endpoint (Phase 1)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from friday.api.ws import router
from friday.providers.emotion import FakeEmotion
from friday.voice.emotion_stream import EmotionStreamAnalyzer


def _app(enable: bool, analyzer: EmotionStreamAnalyzer | None = None) -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    app.state.settings = SimpleNamespace(enable_emotion=enable, enable_wakeword=False)
    app.state.emotion_analyzer = analyzer
    return app


def test_ws_emotion_closed_when_disabled() -> None:
    client = TestClient(_app(enable=False))
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/ws/emotion") as ws:
            ws.receive_json()


def test_ws_emotion_ready_then_streams() -> None:
    analyzer = EmotionStreamAnalyzer(
        FakeEmotion(valence=0.2, arousal=0.3, dominance=0.4),
        window_s=0.5, hop_s=0.25,
    )
    client = TestClient(_app(enable=True, analyzer=analyzer))
    with client.websocket_connect("/ws/emotion") as ws:
        assert ws.receive_json() == {"type": "ready"}
        # Drive one hop of audio through the analyzer; the registered listener
        # forwards the reading to the socket (thread-safe across event loops).
        asyncio.run(analyzer.push(b"\x00\x01" * 8000))
        msg = ws.receive_json()
        assert msg["label"] == "sad"
