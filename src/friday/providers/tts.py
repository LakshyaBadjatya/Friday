"""Text-to-speech (TTS) provider abstraction, fake, and deferred real adapters.

This module owns the typed TTS boundary for FRIDAY:

* :class:`VoiceConfig` — the pydantic v2 voice-selection model.
* :class:`TTSProvider` — the runtime-checkable async ``synthesize`` protocol.
* :class:`FakeTTS` — a deterministic provider for tests returning non-empty
  audio bytes (no audio synthesis).
* :class:`PiperTTS` / :class:`ElevenLabsTTS` — the real adapters, stubbed; both
  raise :class:`NotImplementedError` until Phase 3 (voice is flagged off this
  session).

No TTS SDK is imported here; the real adapters' dependencies land in Phase 3.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

_PHASE_3_NOTE = (
    "Real TTS ({backend}) is deferred to Phase 3 (voice is flagged off this "
    "session); use FakeTTS for tests."
)


class VoiceConfig(BaseModel):
    """Voice-selection parameters for a synthesis request.

    Attributes:
        voice_id: Provider-specific voice identifier.
        speed: Playback rate multiplier (``1.0`` is natural speed).
    """

    voice_id: str = "default"
    speed: float = Field(default=1.0, gt=0)


@runtime_checkable
class TTSProvider(Protocol):
    """Async contract turning text into audio bytes for a given voice."""

    async def synthesize(self, text: str, voice: VoiceConfig) -> bytes:
        """Synthesize ``text`` into audio bytes.

        Args:
            text: The text to speak.
            voice: The :class:`VoiceConfig` selecting voice and rate.

        Returns:
            Encoded audio bytes (container/encoding is provider-defined).
        """
        ...


class FakeTTS:
    """A deterministic :class:`TTSProvider` for tests.

    Returns a fixed non-empty byte string regardless of input so callers can
    assert audio was produced without invoking a real synthesizer.
    """

    async def synthesize(self, text: str, voice: VoiceConfig) -> bytes:
        return b"fake-audio-bytes"


class PiperTTS:
    """Real Piper :class:`TTSProvider` (local dev default) — deferred to Phase 3."""

    async def synthesize(self, text: str, voice: VoiceConfig) -> bytes:
        raise NotImplementedError(_PHASE_3_NOTE.format(backend="Piper"))


class ElevenLabsTTS:
    """Real ElevenLabs :class:`TTSProvider` (flagged) — deferred to Phase 3."""

    async def synthesize(self, text: str, voice: VoiceConfig) -> bytes:
        raise NotImplementedError(_PHASE_3_NOTE.format(backend="ElevenLabs"))
