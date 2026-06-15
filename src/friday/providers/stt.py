"""Speech-to-text (STT) provider abstraction, fake, and deferred real adapter.

This module owns the typed STT boundary for FRIDAY:

* :class:`Transcript` — the normalized pydantic v2 result model.
* :class:`STTProvider` — the runtime-checkable async ``transcribe`` protocol.
* :class:`FakeSTT` — a deterministic provider for tests (zero models, no audio
  decoding) returning a non-empty :class:`Transcript`.
* :class:`WhisperSTT` — the real Whisper adapter, stubbed; raises
  :class:`NotImplementedError` until Phase 3 (voice is flagged off this
  session).

No STT SDK is imported here; the real adapter's dependency lands in Phase 3.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel

_PHASE_3_NOTE = (
    "Real Whisper STT is deferred to Phase 3 (voice is flagged off this "
    "session); use FakeSTT for tests."
)


class Transcript(BaseModel):
    """A normalized speech-to-text result.

    Attributes:
        text: The transcribed text. Always populated for a successful result.
        lang: Detected or requested BCP-47 language tag, if known.
    """

    text: str
    lang: str | None = None


@runtime_checkable
class STTProvider(Protocol):
    """Async contract turning raw audio bytes into a :class:`Transcript`."""

    async def transcribe(self, audio: bytes, lang: str | None) -> Transcript:
        """Transcribe ``audio`` into a :class:`Transcript`.

        Args:
            audio: Raw audio bytes (container/encoding is provider-defined).
            lang: Optional BCP-47 language hint; ``None`` lets the provider
                auto-detect.

        Returns:
            The normalized :class:`Transcript`.
        """
        ...


class FakeSTT:
    """A deterministic :class:`STTProvider` for tests.

    Ignores the audio payload and returns a fixed non-empty transcript,
    echoing back the requested ``lang`` so callers can assert propagation.
    """

    async def transcribe(self, audio: bytes, lang: str | None) -> Transcript:
        return Transcript(text="fake transcript", lang=lang)


class WhisperSTT:
    """Real Whisper-based :class:`STTProvider` — deferred to Phase 3.

    Present so wiring/typing can reference the concrete adapter now; every call
    raises :class:`NotImplementedError` until the voice phase lands.
    """

    async def transcribe(self, audio: bytes, lang: str | None) -> Transcript:
        raise NotImplementedError(_PHASE_3_NOTE)
