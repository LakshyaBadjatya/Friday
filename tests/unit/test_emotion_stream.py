"""Unit tests for the sliding-window emotion analyzer (Phase 1).

Model-free: driven by :class:`FakeEmotion`, so no ONNX model is needed.
"""

from __future__ import annotations

import asyncio

from friday.providers.emotion import Emotion, FakeEmotion
from friday.voice.emotion_stream import EmotionStreamAnalyzer


def test_analyzer_emits_smoothed_emotions_to_listener() -> None:
    # 16 kHz, 16-bit mono -> 2 bytes/sample. 0.5s frames; window 1.0s, hop 0.5s.
    frame = b"\x00\x01" * 8000  # 0.5s of audio
    seen: list[Emotion] = []
    an = EmotionStreamAnalyzer(
        FakeEmotion(valence=0.2, arousal=0.3, dominance=0.4),
        sr=16000, window_s=1.0, hop_s=0.5, alpha=0.5,
    )
    an.on_emotion(seen.append)

    async def drive() -> Emotion | None:
        for _ in range(4):
            await an.push(frame)
        return an.last()

    last = asyncio.run(drive())
    assert len(seen) >= 1  # at least one hop produced a reading
    assert seen[-1].label == "sad"
    assert last is not None and last.label == "sad"


def test_analyzer_no_emotion_before_first_window() -> None:
    an = EmotionStreamAnalyzer(FakeEmotion(), sr=16000, window_s=1.0, hop_s=0.5)
    assert an.last() is None


def test_feed_analyzer_pumps_capture_into_analyzer() -> None:
    from friday.voice.capture import FakeAudioCapture
    from friday.voice.emotion_stream import feed_analyzer

    frame = b"\x00\x01" * 8000  # 0.5s
    seen: list[Emotion] = []
    an = EmotionStreamAnalyzer(
        FakeEmotion(valence=0.2, arousal=0.3, dominance=0.4), window_s=0.5, hop_s=0.25
    )
    an.on_emotion(seen.append)
    cap = FakeAudioCapture([frame, frame, frame])
    asyncio.run(feed_analyzer(cap, an))   # drains the finite fake stream, then returns
    assert len(seen) >= 1 and an.last() is not None
