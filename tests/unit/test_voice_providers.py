"""Unit tests for the STT/TTS provider contracts and their fakes.

These tests pin the contract from Task 0.7 of the implementation plan. Voice is
flagged off this session; only the fakes are exercised. The real adapters are
deferred to Phase 3 and must raise :class:`NotImplementedError`.
"""

from __future__ import annotations

import inspect

import pytest

from friday.providers.stt import (
    FakeSTT,
    STTProvider,
    Transcript,
    WhisperSTT,
)
from friday.providers.tts import (
    ElevenLabsTTS,
    FakeTTS,
    PiperTTS,
    TTSProvider,
    VoiceConfig,
)


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def test_transcript_model_constructs() -> None:
    t = Transcript(text="hello world", lang="en")
    assert t.text == "hello world"
    assert t.lang == "en"


def test_transcript_lang_optional() -> None:
    t = Transcript(text="hi")
    assert t.lang is None


def test_voice_config_defaults() -> None:
    cfg = VoiceConfig()
    assert isinstance(cfg.voice_id, str)
    assert cfg.voice_id
    assert isinstance(cfg.speed, float)
    assert cfg.speed > 0


def test_voice_config_overrides() -> None:
    cfg = VoiceConfig(voice_id="custom", speed=1.5)
    assert cfg.voice_id == "custom"
    assert cfg.speed == 1.5


# --------------------------------------------------------------------------- #
# Protocol membership
# --------------------------------------------------------------------------- #
def test_fakes_are_providers() -> None:
    assert isinstance(FakeSTT(), STTProvider)
    assert isinstance(FakeTTS(), TTSProvider)


# --------------------------------------------------------------------------- #
# FakeSTT
# --------------------------------------------------------------------------- #
async def test_fake_stt_returns_nonempty_transcript() -> None:
    stt = FakeSTT()
    result = await stt.transcribe(b"\x00\x01\x02", None)
    assert isinstance(result, Transcript)
    assert result.text


async def test_fake_stt_propagates_lang() -> None:
    stt = FakeSTT()
    result = await stt.transcribe(b"audio", "fr")
    assert result.lang == "fr"


# --------------------------------------------------------------------------- #
# FakeTTS
# --------------------------------------------------------------------------- #
async def test_fake_tts_returns_nonempty_bytes() -> None:
    tts = FakeTTS()
    out = await tts.synthesize("hello", VoiceConfig())
    assert isinstance(out, bytes)
    assert len(out) > 0


# --------------------------------------------------------------------------- #
# Real adapters deferred to Phase 3
# --------------------------------------------------------------------------- #
async def test_whisper_stt_not_implemented() -> None:
    stt = WhisperSTT()
    with pytest.raises(NotImplementedError) as exc:
        await stt.transcribe(b"audio", None)
    assert "phase 3" in str(exc.value).lower()


async def test_piper_tts_not_implemented() -> None:
    tts = PiperTTS()
    with pytest.raises(NotImplementedError) as exc:
        await tts.synthesize("hi", VoiceConfig())
    assert "phase 3" in str(exc.value).lower()


async def test_elevenlabs_tts_not_implemented() -> None:
    tts = ElevenLabsTTS()
    with pytest.raises(NotImplementedError) as exc:
        await tts.synthesize("hi", VoiceConfig())
    assert "phase 3" in str(exc.value).lower()


def test_real_adapters_are_providers() -> None:
    assert isinstance(WhisperSTT(), STTProvider)
    assert isinstance(PiperTTS(), TTSProvider)
    assert isinstance(ElevenLabsTTS(), TTSProvider)


# --------------------------------------------------------------------------- #
# Contract signatures
# --------------------------------------------------------------------------- #
def test_transcribe_is_coroutine() -> None:
    assert inspect.iscoroutinefunction(FakeSTT().transcribe)


def test_synthesize_is_coroutine() -> None:
    assert inspect.iscoroutinefunction(FakeTTS().synthesize)
