"""Text-to-speech (TTS) provider abstraction, fakes, real adapters, factory.

This module owns the typed TTS boundary for FRIDAY:

* :class:`VoiceConfig` — the pydantic v2 voice-selection model.
* :class:`TTSProvider` — the runtime-checkable async ``synthesize`` protocol.
* :class:`FakeTTS` — a deterministic provider for tests returning non-empty
  audio bytes (no audio synthesis).
* :class:`PiperTTS` / :class:`ElevenLabsTTS` — the original Phase-0 placeholder
  adapters; every call raises :class:`NotImplementedError` (kept for backwards
  compatibility with the Phase-0 wiring/tests).
* :class:`PiperTTSProvider` — the real local Piper adapter: it shells out to the
  ``piper`` binary (lazy, via :mod:`subprocess`) and returns WAV bytes.
* :class:`ElevenLabsTTSProvider` — the real ElevenLabs adapter: a lazy
  :mod:`httpx` POST to the ElevenLabs API; requires ``ELEVENLABS_API_KEY``.
* :func:`make_tts` — a factory selecting ``piper`` | ``elevenlabs`` | ``fake``
  from ``settings.tts_provider`` (env: ``FRIDAY_TTS_PROVIDER``).

No TTS SDK / heavy dependency is imported at module top level. ``piper`` and
``httpx`` are only touched inside the real adapters' methods, so importing this
module never requires the optional voice extras and ``uv sync`` stays green.
"""

from __future__ import annotations

import os
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

from friday.config import Settings
from friday.errors import ProviderError

_PHASE_3_NOTE = (
    "Real TTS ({backend}) is deferred to Phase 3 (voice is flagged off this "
    "session); use FakeTTS for tests."
)

_PIPER_INSTALL_HINT = (
    "The `piper` text-to-speech binary was not found on PATH. Voice extras are "
    "optional and kept out of the uv lock; install them with "
    "`make install-voice` (uv pip install -r requirements-voice.txt) and ensure "
    "a piper voice model is available."
)

_HTTPX_INSTALL_HINT = (
    "httpx is required for the ElevenLabs TTS adapter but is not importable. "
    "Install the voice extras with `make install-voice` "
    "(uv pip install -r requirements-voice.txt)."
)

_ELEVENLABS_KEY_HINT = (
    "ELEVENLABS_API_KEY is not set. Export it in the environment (or pass it to "
    "ElevenLabsTTSProvider) to use the ElevenLabs TTS adapter."
)

_ELEVENLABS_BASE_URL = "https://api.elevenlabs.io/v1/text-to-speech"


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
    """Phase-0 placeholder Piper :class:`TTSProvider`.

    Every call raises :class:`NotImplementedError`. The real implementation
    lives in :class:`PiperTTSProvider` (selected via :func:`make_tts`).
    """

    async def synthesize(self, text: str, voice: VoiceConfig) -> bytes:
        raise NotImplementedError(_PHASE_3_NOTE.format(backend="Piper"))


class ElevenLabsTTS:
    """Phase-0 placeholder ElevenLabs :class:`TTSProvider`.

    Every call raises :class:`NotImplementedError`. The real implementation
    lives in :class:`ElevenLabsTTSProvider` (selected via :func:`make_tts`).
    """

    async def synthesize(self, text: str, voice: VoiceConfig) -> bytes:
        raise NotImplementedError(_PHASE_3_NOTE.format(backend="ElevenLabs"))


class PiperTTSProvider:
    """Real local-Piper :class:`TTSProvider`.

    Shells out to the ``piper`` binary and returns WAV bytes on stdout. Both the
    binary lookup and the :mod:`subprocess` import are performed **lazily** at
    synthesis time, so constructing this adapter (and importing this module)
    never requires the optional voice extras. A missing binary surfaces as a
    :class:`ProviderError` carrying the ``make install-voice`` hint.

    Args:
        model_path: Path to a piper ``.onnx`` voice model. If ``None`` the
            ``piper`` binary's own default-model resolution is used.
        binary: Name/path of the piper executable (default ``"piper"``).
    """

    def __init__(self, model_path: str | None = None, binary: str = "piper") -> None:
        self._model_path = model_path
        self._binary = binary

    async def synthesize(self, text: str, voice: VoiceConfig) -> bytes:
        import asyncio
        import shutil
        import subprocess  # noqa: PLC0415 - lazy by design

        if shutil.which(self._binary) is None:
            raise ProviderError(_PIPER_INSTALL_HINT)

        cmd = [self._binary, "--output_file", "-"]
        if self._model_path is not None:
            cmd += ["--model", self._model_path]
        # Piper exposes a length-scale knob; map our speed multiplier to it
        # (faster speech => shorter samples => smaller length scale).
        if voice.speed and voice.speed > 0:
            cmd += ["--length_scale", str(1.0 / voice.speed)]

        def _run() -> bytes:
            try:
                completed = subprocess.run(  # noqa: S603 - args are controlled
                    cmd,
                    input=text.encode("utf-8"),
                    capture_output=True,
                    check=True,
                )
            except FileNotFoundError as exc:  # pragma: no cover - covered by which()
                raise ProviderError(_PIPER_INSTALL_HINT) from exc
            except subprocess.CalledProcessError as exc:  # pragma: no cover
                raise ProviderError(
                    f"piper synthesis failed (exit {exc.returncode}): "
                    f"{exc.stderr.decode('utf-8', 'replace')}"
                ) from exc
            return completed.stdout

        return await asyncio.get_running_loop().run_in_executor(None, _run)


class ElevenLabsTTSProvider:
    """Real ElevenLabs :class:`TTSProvider`.

    Performs a lazy :mod:`httpx` POST to the ElevenLabs text-to-speech API and
    returns the audio bytes. ``httpx`` is imported lazily inside
    :meth:`synthesize` (keeping module import light); the API key is resolved at
    construction time from the explicit ``api_key`` argument or the
    ``ELEVENLABS_API_KEY`` environment variable, raising a :class:`ProviderError`
    when neither is present.

    Args:
        api_key: ElevenLabs API key. If ``None`` the ``ELEVENLABS_API_KEY``
            environment variable is read.
        timeout: Per-request HTTP timeout in seconds.
    """

    def __init__(self, api_key: str | None = None, timeout: float = 30.0) -> None:
        key = api_key if api_key is not None else os.environ.get("ELEVENLABS_API_KEY")
        if not key:
            raise ProviderError(_ELEVENLABS_KEY_HINT)
        self._api_key = key
        self._timeout = timeout

    async def synthesize(self, text: str, voice: VoiceConfig) -> bytes:
        try:
            import httpx  # noqa: PLC0415 - lazy by design
        except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
            raise ProviderError(_HTTPX_INSTALL_HINT) from exc

        url = f"{_ELEVENLABS_BASE_URL}/{voice.voice_id}"
        headers = {
            "xi-api-key": self._api_key,
            "accept": "audio/mpeg",
            "content-type": "application/json",
        }
        payload = {
            "text": text,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
        }
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
        except httpx.HTTPStatusError as exc:  # pragma: no cover - real-API guard
            raise ProviderError(
                f"ElevenLabs request failed with status "
                f"{exc.response.status_code}: {exc.response.text}"
            ) from exc
        except httpx.HTTPError as exc:  # pragma: no cover - real-API guard
            raise ProviderError(f"ElevenLabs request failed: {exc}") from exc
        return response.content


def make_tts(settings: Settings) -> TTSProvider:
    """Build a :class:`TTSProvider` selected by ``settings.tts_provider``.

    Selection (env: ``FRIDAY_TTS_PROVIDER``):

    * ``"piper"`` -> :class:`PiperTTSProvider` (local default).
    * ``"elevenlabs"`` -> :class:`ElevenLabsTTSProvider` (needs
      ``ELEVENLABS_API_KEY``).
    * ``"fake"`` -> :class:`FakeTTS` (tests / no synthesis).

    The real adapters lazy-load their heavy dependencies, so calling this factory
    is cheap and only fails at synthesis time (or, for ElevenLabs, at
    construction if the API key is missing). An unknown value raises a
    :class:`ProviderError` so misconfiguration fails loudly.
    """
    provider = settings.tts_provider.strip().lower()
    if provider == "fake":
        return FakeTTS()
    if provider == "piper":
        return PiperTTSProvider()
    if provider == "elevenlabs":
        return ElevenLabsTTSProvider()
    raise ProviderError(
        f"unknown FRIDAY_TTS_PROVIDER={settings.tts_provider!r}; "
        "expected one of: piper, elevenlabs, fake"
    )
