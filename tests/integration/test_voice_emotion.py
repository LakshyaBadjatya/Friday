"""Phase-1 emotion tee through the voice pipeline (offline, model-free).

Reuses the integration suite's real fakes and orchestrator double; the analyzer
is driven by :class:`FakeEmotion`, so no model or network is touched.
"""

from __future__ import annotations

import asyncio

from friday.providers.emotion import FakeEmotion
from friday.providers.stt import FakeSTT
from friday.providers.tts import FakeTTS
from friday.voice.capture import FakeAudioCapture
from friday.voice.emotion_stream import EmotionStreamAnalyzer
from friday.voice.fixtures import make_silence_frame, make_wake_frame
from friday.voice.pipeline import VoicePipeline
from friday.voice.vad import FakeVAD
from friday.voice.wake_word import FakeWakeWord
from tests.integration.test_voice_pipeline import _ScriptedSTT, _orchestrator


def _capture() -> FakeAudioCapture:
    # wake frame consumed by _await_wake; two utterance frames; VAD ends on silence.
    return FakeAudioCapture(
        [make_wake_frame(), make_wake_frame(), make_silence_frame(), make_silence_frame()]
    )


def test_pipeline_sets_emotion_on_turn_when_analyzer_present() -> None:
    analyzer = EmotionStreamAnalyzer(
        FakeEmotion(valence=0.2, arousal=0.3, dominance=0.4),
        window_s=0.1, hop_s=0.05,  # tiny hop so a single captured frame fires it
    )
    pipe = VoicePipeline(
        wake=FakeWakeWord(), capture=_capture(), vad=FakeVAD([True, True, False]),
        stt=_ScriptedSTT("what's the status"),
        orchestrator=_orchestrator("All systems nominal, Boss."),
        tts=FakeTTS(), analyzer=analyzer,
    )
    turn = asyncio.run(asyncio.wait_for(pipe.run_once("v1"), timeout=2.0))
    assert turn is not None
    assert turn.emotion is not None and turn.emotion.label == "sad"


def test_pipeline_emotion_none_is_noop() -> None:
    pipe = VoicePipeline(
        wake=FakeWakeWord(), capture=_capture(), vad=FakeVAD([True, True, False]),
        stt=_ScriptedSTT("what's the status"),
        orchestrator=_orchestrator("ok"), tts=FakeTTS(),  # no analyzer
    )
    turn = asyncio.run(asyncio.wait_for(pipe.run_once("v0"), timeout=2.0))
    assert turn is not None and turn.emotion is None


def test_pipeline_modulates_tts_speed_when_emotion_tts_on() -> None:
    captured: dict[str, float] = {}

    class _RecordingTTS:
        async def synthesize(self, text: str, voice) -> bytes:  # noqa: ANN001
            captured["speed"] = voice.speed
            return b"audio"

    analyzer = EmotionStreamAnalyzer(
        FakeEmotion(valence=0.2, arousal=0.1, dominance=0.4),  # low arousal -> slower
        window_s=0.1, hop_s=0.05,
    )
    pipe = VoicePipeline(
        wake=FakeWakeWord(), capture=_capture(), vad=FakeVAD([True, True, False]),
        stt=_ScriptedSTT("what's the status"),
        orchestrator=_orchestrator("ok"), tts=_RecordingTTS(),
        analyzer=analyzer, emotion_tts=True,
    )
    asyncio.run(asyncio.wait_for(pipe.run_once("v1"), timeout=2.0))
    assert captured["speed"] < 1.0
