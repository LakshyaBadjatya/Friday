"""Wiring for the continuous mic -> emotion-analyzer capture loop (graceful)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest import mock

import friday.app as app_module
from friday.errors import ProviderError
from friday.providers.emotion import FakeEmotion
from friday.voice.capture import FakeAudioCapture
from friday.voice.emotion_stream import EmotionStreamAnalyzer


def _ctx(emotion_mic: bool):
    app = SimpleNamespace(
        state=SimpleNamespace(
            emotion_analyzer=EmotionStreamAnalyzer(
                FakeEmotion(), window_s=0.5, hop_s=0.25
            )
        )
    )
    settings = SimpleNamespace(emotion_mic=emotion_mic)
    return app, settings


def test_no_task_when_mic_off() -> None:
    app, settings = _ctx(emotion_mic=False)
    assert app_module._start_emotion_capture(app, settings) is None


def test_no_task_when_backend_missing(monkeypatch) -> None:
    app, settings = _ctx(emotion_mic=True)

    def boom():
        raise ProviderError("no sounddevice")

    monkeypatch.setattr(app_module, "MicCapture", boom)
    assert app_module._start_emotion_capture(app, settings) is None


def test_task_feeds_analyzer_with_fake_capture() -> None:
    app, settings = _ctx(emotion_mic=True)
    seen: list = []
    app.state.emotion_analyzer.on_emotion(seen.append)
    frame = b"\x00\x01" * 8000

    async def run() -> None:
        with mock.patch.object(
            app_module, "MicCapture",
            lambda: FakeAudioCapture([frame, frame, frame]),
        ):
            task = app_module._start_emotion_capture(app, settings)
            assert task is not None
            await task  # finite fake stream -> the loop drains and returns

    asyncio.run(run())
    assert len(seen) >= 1
