"""End-to-end voice pipeline + ``POST /voice`` integration tests (Phase 3 / Stage B).

Everything here runs offline against fakes and synthetic WAV bytes produced by
:mod:`friday.voice.fixtures` (no microphone, no model, no network):

1. **Pipeline e2e** — a :class:`FakeWakeWord` fires on a wake fixture frame, a
   :class:`FakeAudioCapture` replays the utterance frames, a :class:`FakeVAD`
   scripts the speech/silence boundary, :class:`FakeSTT` yields the transcript,
   the real :class:`Orchestrator` (wired to a scripted :class:`FakeLLM`) produces
   the response, and :class:`FakeTTS` returns non-empty audio bytes. The
   pipeline's ``run_once`` returns the transcript, response text, mode, and audio
   and emits the Idle->Listening->Routing mode transitions.
2. **Barge-in** — ``speak_with_bargein`` streams TTS chunks via a cancellable
   task; setting the barge-in event cancels the playback promptly (inside an
   ``asyncio.wait_for(..., timeout=0.5)`` bound), no chunk is emitted after the
   signal, and the pipeline re-enters Listening.
3. **POST /voice** — returns 404 / ``{"detail": "voice disabled"}`` when
   ``FRIDAY_ENABLE_VOICE`` is off, and transcribes + orchestrates a base64 WAV
   into ``{transcript, text, mode, audio_b64}`` when on.
"""

from __future__ import annotations

import asyncio
import base64
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from friday.app import create_app
from friday.config import Settings
from friday.core.orchestrator import Orchestrator
from friday.core.state import Mode
from friday.memory.short_term import ShortTermMemory
from friday.providers.llm import FakeLLM, LLMResponse, Usage
from friday.providers.stt import FakeSTT, Transcript
from friday.providers.tts import FakeTTS, VoiceConfig
from friday.tools.registry import ToolRegistry
from friday.tools.web_search import WebSearchTool
from friday.voice.capture import FakeAudioCapture
from friday.voice.fixtures import make_silence_frame, make_wake_frame, make_wav
from friday.voice.pipeline import VoicePipeline
from friday.voice.vad import FakeVAD
from friday.voice.wake_word import FakeWakeWord

PERSONA_PATH = (
    Path(__file__).resolve().parents[2] / "src" / "friday" / "persona" / "friday.md"
)


def _orchestrator(text: str) -> Orchestrator:
    """A real orchestrator wired to a scripted :class:`FakeLLM` (zero network)."""
    registry = ToolRegistry()
    registry.register(WebSearchTool())
    llm = FakeLLM(responses=[LLMResponse(text=text, tool_calls=[], usage=Usage())])
    return Orchestrator(
        llm=llm,
        registry=registry,
        memory=ShortTermMemory(),
        persona_path=PERSONA_PATH,
    )


class _ScriptedSTT:
    """A :class:`STTProvider` returning a fixed conversational transcript.

    ``FakeSTT`` returns ``"fake transcript"``, which the deterministic router
    reads as ambiguous (CLARIFY). For an end-to-end turn that exercises persona
    synthesis we want a transcript the router classifies as CONVERSATION, so this
    helper returns a question-word utterance ("what's the status").
    """

    def __init__(self, text: str = "what's the status") -> None:
        self._text = text

    async def transcribe(self, audio: bytes, lang: str | None) -> Transcript:
        return Transcript(text=self._text, lang=lang)


async def test_pipeline_run_once_end_to_end() -> None:
    """Wake -> capture -> STT -> orchestrator -> TTS, returning audio + transcript."""
    # The wake frame fires FakeWakeWord; two utterance frames then a silence
    # frame so the VAD script (speech, speech, silence) ends the utterance.
    wake = make_wake_frame()
    utter_a = make_wake_frame()  # any non-empty body; VAD script drives the cut
    utter_b = make_silence_frame()
    capture = FakeAudioCapture([wake, utter_a, utter_b, make_silence_frame()])

    pipeline = VoicePipeline(
        wake=FakeWakeWord(),
        capture=capture,
        vad=FakeVAD([True, True, False]),
        stt=_ScriptedSTT("what's the status"),
        orchestrator=_orchestrator("All systems nominal, Boss."),
        tts=FakeTTS(),
    )

    seen_modes: list[Mode] = []
    pipeline.on_mode(seen_modes.append)

    result = await asyncio.wait_for(pipeline.run_once(session_id="v1"), timeout=2.0)
    assert result is not None

    assert result.transcript == "what's the status"
    assert result.response_text == "All systems nominal, Boss."
    assert result.mode is Mode.CONVERSATION
    assert isinstance(result.audio, bytes)
    assert result.audio  # non-empty

    # Idle -> Listening -> Routing transitions emitted on state.
    assert Mode.LISTENING in seen_modes
    assert Mode.ROUTING in seen_modes


async def test_pipeline_run_once_returns_none_when_no_wake() -> None:
    """With no wake frame in the capture, ``run_once`` returns ``None`` (no turn)."""
    capture = FakeAudioCapture([make_silence_frame(), make_silence_frame()])
    pipeline = VoicePipeline(
        wake=FakeWakeWord(),
        capture=capture,
        vad=FakeVAD([False]),
        stt=FakeSTT(),
        orchestrator=_orchestrator("unused"),
        tts=FakeTTS(),
    )
    result = await asyncio.wait_for(pipeline.run_once(session_id="v0"), timeout=2.0)
    assert result is None


class _StreamingTTS:
    """A streaming :class:`FakeTTS` that records every chunk it actually emits.

    ``synthesize`` returns joined bytes (so it still satisfies the TTSProvider
    protocol), while ``stream`` yields chunks one at a time with a small await
    between them so a barge-in cancellation can land mid-stream. Every yielded
    chunk is appended to :attr:`emitted`, letting the test assert that nothing is
    emitted after the barge-in signal.
    """

    def __init__(self, chunks: int = 50) -> None:
        self._chunks = chunks
        self.emitted: list[bytes] = []

    async def synthesize(self, text: str, voice: VoiceConfig) -> bytes:
        return b"".join(f"chunk-{i}".encode() for i in range(self._chunks))

    async def stream(self, text: str, voice: VoiceConfig) -> AsyncIterator[bytes]:
        for i in range(self._chunks):
            chunk = f"chunk-{i}".encode()
            self.emitted.append(chunk)
            yield chunk
            # Yield control so the barge-in event can be observed between chunks.
            await asyncio.sleep(0.01)


async def test_speak_with_bargein_cancels_promptly() -> None:
    """Setting the barge-in event cancels playback within the timeout bound."""
    tts = _StreamingTTS(chunks=100)
    pipeline = VoicePipeline(
        wake=FakeWakeWord(),
        capture=FakeAudioCapture([]),
        vad=FakeVAD([]),
        stt=FakeSTT(),
        orchestrator=_orchestrator("unused"),
        tts=tts,
    )

    seen_modes: list[Mode] = []
    pipeline.on_mode(seen_modes.append)

    bargein = asyncio.Event()

    async def _drive() -> None:
        task = asyncio.ensure_future(
            pipeline.speak_with_bargein("a long spoken reply", bargein)
        )
        # Let a couple of chunks stream, then barge in.
        await asyncio.sleep(0.03)
        bargein.set()
        await task

    # The whole speak-then-cancel must complete well within the bound.
    await asyncio.wait_for(_drive(), timeout=0.5)

    emitted_at_signal = len(tts.emitted)
    # Nothing further should be emitted after cancellation; give the loop a beat.
    await asyncio.sleep(0.05)
    assert len(tts.emitted) == emitted_at_signal
    # The stream was interrupted, not fully drained.
    assert emitted_at_signal < 100
    # Barge-in re-enters Listening.
    assert seen_modes[-1] is Mode.LISTENING


def _enable_voice_settings() -> Settings:
    return Settings(_env_file=None, enable_voice=True, llm_provider="fake")


def _disable_voice_settings() -> Settings:
    return Settings(_env_file=None, enable_voice=False, llm_provider="fake")


def test_post_voice_disabled_returns_404(monkeypatch: pytest.MonkeyPatch) -> None:
    """When voice is off, ``POST /voice`` is 404 with the disabled detail."""
    import friday.app as app_module

    monkeypatch.setattr(app_module, "get_settings", _disable_voice_settings)

    app = create_app()
    with TestClient(app) as client:
        app.state.settings = _disable_voice_settings()
        wav_b64 = base64.b64encode(make_wav(seconds=0.1)).decode()
        resp = client.post("/voice", json={"audio_b64": wav_b64})

    assert resp.status_code == 404
    assert resp.json() == {"detail": "voice disabled"}


def test_post_voice_enabled_transcribes_and_orchestrates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When voice is on, ``POST /voice`` returns transcript/text/mode/audio_b64."""
    import friday.app as app_module

    monkeypatch.setattr(app_module, "get_settings", _enable_voice_settings)

    app = create_app()
    with TestClient(app) as client:
        # Re-pin the enabled settings + a scripted orchestrator after lifespan.
        app.state.settings = _enable_voice_settings()
        app.state.orchestrator = _orchestrator("Heard you loud and clear, Boss.")
        app.state.voice_stt = _ScriptedSTT("what's the status")
        app.state.voice_tts = FakeTTS()

        wav_b64 = base64.b64encode(make_wav(seconds=0.1)).decode()
        resp = client.post("/voice", json={"audio_b64": wav_b64})

    assert resp.status_code == 200
    body = resp.json()
    assert body["transcript"] == "what's the status"
    assert body["text"] == "Heard you loud and clear, Boss."
    assert body["mode"] == "CONVERSATION"
    # Audio echoed back as base64 of the FakeTTS bytes.
    assert base64.b64decode(body["audio_b64"]) == b"fake-audio-bytes"
