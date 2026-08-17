"""Speech-to-text (STT) provider abstraction, fake, and real adapters.

This module owns the typed STT boundary for FRIDAY:

* :class:`Transcript` — the normalized pydantic v2 result model.
* :class:`STTProvider` — the runtime-checkable async ``transcribe`` protocol.
* :class:`FakeSTT` — a deterministic provider for tests (zero models, no audio
  decoding) returning a non-empty :class:`Transcript`.
* :class:`WhisperSTT` — the original Phase-0 placeholder adapter; every call
  raises :class:`NotImplementedError` (kept for backwards compatibility with the
  Phase-0 wiring/tests).
* :class:`FasterWhisperSTT` — the real Whisper adapter built on
  ``faster-whisper``. The heavy dependency is **lazy-imported** inside the
  constructor so importing this module never pulls in ``faster_whisper``; a
  missing library surfaces as a :class:`ProviderError` with a
  ``make install-voice`` hint.

No STT SDK is imported at module top level: ``faster_whisper`` is only touched
inside :class:`FasterWhisperSTT`. This keeps ``uv sync`` / the gate green on a
machine without the optional voice extras installed.
"""

from __future__ import annotations

import os
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel

from friday.config import Settings
from friday.errors import ProviderError

_PHASE_3_NOTE = (
    "Real Whisper STT is deferred to Phase 3 (voice is flagged off this "
    "session); use FakeSTT for tests."
)

_INSTALL_HINT = (
    "faster-whisper is not installed. Voice extras are optional and kept out "
    "of the uv lock; install them with `make install-voice` "
    "(uv pip install -r requirements-voice.txt)."
)

_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

_GEMINI_KEY_HINT = (
    "GEMINI_API_KEY is not set. Export it in the environment (or pass it to "
    "GeminiSTT) to transcribe with the hosted Gemini adapter."
)

_GEMINI_HTTPX_HINT = (
    "httpx is required for the Gemini STT adapter but is not importable; it is "
    "a base dependency, so this means the environment is incomplete."
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
    """Phase-0 placeholder :class:`STTProvider`.

    Present so early wiring/typing could reference a concrete adapter; every
    call raises :class:`NotImplementedError`. The real implementation lives in
    :class:`FasterWhisperSTT`.
    """

    async def transcribe(self, audio: bytes, lang: str | None) -> Transcript:
        raise NotImplementedError(_PHASE_3_NOTE)


class FasterWhisperSTT:
    """Real :class:`STTProvider` backed by ``faster-whisper``.

    The ``faster_whisper`` package is **lazy-imported** inside ``__init__`` so
    that merely importing this module (or constructing :class:`FakeSTT`) never
    requires the heavy optional dependency. If ``faster_whisper`` is missing the
    constructor raises a :class:`ProviderError` carrying the
    ``make install-voice`` hint.

    Args:
        model_size: A ``faster-whisper`` model identifier (e.g. ``"base"``,
            ``"small"``, ``"medium"``). Defaults to ``"base"``.
        device: Compute device passed to ``WhisperModel`` (``"cpu"`` /
            ``"cuda"`` / ``"auto"``).
        compute_type: ``faster-whisper`` quantization/compute type
            (e.g. ``"int8"`` on CPU, ``"float16"`` on GPU).
    """

    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
    ) -> None:
        try:
            from faster_whisper import (  # type: ignore[import-not-found] # noqa: PLC0415
                WhisperModel,
            )
        except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
            raise ProviderError(_INSTALL_HINT) from exc

        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._model: Any = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
        )

    async def transcribe(self, audio: bytes, lang: str | None) -> Transcript:
        """Transcribe raw audio bytes via ``faster-whisper``.

        The audio bytes are written to a temporary WAV file and handed to the
        model. ``faster_whisper.WhisperModel.transcribe`` is synchronous and CPU
        bound; it is run on the default executor so the event loop is not
        blocked. Any failure from the underlying model is wrapped in a
        :class:`ProviderError`.
        """
        import asyncio
        import tempfile

        def _run() -> Transcript:
            with tempfile.NamedTemporaryFile(suffix=".wav") as handle:
                handle.write(audio)
                handle.flush()
                try:
                    segments, info = self._model.transcribe(
                        handle.name,
                        language=lang,
                    )
                    text = "".join(segment.text for segment in segments).strip()
                except Exception as exc:  # pragma: no cover - real-model guard
                    raise ProviderError(
                        f"faster-whisper transcription failed: {exc}"
                    ) from exc
            detected = getattr(info, "language", None)
            return Transcript(text=text, lang=lang or detected)

        return await asyncio.get_running_loop().run_in_executor(None, _run)


class GeminiSTT:
    """Real :class:`STTProvider` that transcribes over HTTP, carrying no model.

    :class:`FasterWhisperSTT` needs the voice extras and a few hundred megabytes
    of weights on local disk, which a small container does not have; this adapter
    needs only the ``GEMINI_API_KEY`` already configured for the LLM. That makes
    it the one that can answer ``POST /voice`` on a hosted deployment.

    The audio is inlined as base64 in the request, so this is for single-shot
    turns rather than long recordings — the phone sends a few seconds of 16 kHz
    mono WAV, which is comfortably inside the inline limit.

    Args:
        api_key: Gemini API key; falls back to the ``GEMINI_API_KEY`` env var.
        model: Gemini model id. The default is a small, fast one — transcription
            is not a reasoning task and a larger model only costs latency.
        base_url: API root, overridable for tests and proxies.
        mime_type: Container of the audio handed to :meth:`transcribe`.
        timeout: Per-request HTTP timeout in seconds.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-2.0-flash",
        base_url: str = _GEMINI_BASE_URL,
        mime_type: str = "audio/wav",
        timeout: float = 60.0,
    ) -> None:
        key = api_key if api_key is not None else os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ProviderError(_GEMINI_KEY_HINT)
        self._api_key = key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._mime_type = mime_type
        self._timeout = timeout

    async def transcribe(self, audio: bytes, lang: str | None) -> Transcript:
        """Transcribe ``audio`` by asking Gemini for the words and nothing else.

        The prompt is explicit that only the spoken words are wanted, because the
        failure mode of a chat model asked to transcribe is a helpful preamble
        ("Sure! Here is the transcription:") that would be spoken back to the
        user as if she had said it. Silence must come back empty rather than as
        an apology, for the same reason.
        """
        import base64  # noqa: PLC0415 - lazy by design
        import json  # noqa: PLC0415

        try:
            import httpx  # noqa: PLC0415 - lazy by design
        except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
            raise ProviderError(_GEMINI_HTTPX_HINT) from exc

        instruction = (
            "Transcribe the speech in this audio. Reply with the spoken words "
            "only: no preamble, no quotation marks, no commentary, no timestamps. "
            "If there is no intelligible speech, reply with nothing at all."
        )
        if lang:
            instruction += f" The speech is in {lang}."

        url = f"{self._base_url}/models/{self._model}:generateContent"
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": instruction},
                        {
                            "inline_data": {
                                "mime_type": self._mime_type,
                                "data": base64.b64encode(audio).decode("ascii"),
                            }
                        },
                    ]
                }
            ],
            # Transcription is a transcription: no sampling creativity wanted.
            "generationConfig": {"temperature": 0.0},
        }
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    url,
                    headers={
                        "x-goog-api-key": self._api_key,
                        "content-type": "application/json",
                    },
                    json=payload,
                )
                response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise ProviderError(
                f"Gemini transcription failed with status "
                f"{exc.response.status_code}: {exc.response.text}"
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f"Gemini transcription failed: {exc}") from exc

        try:
            body = response.json()
        except (json.JSONDecodeError, ValueError) as exc:
            raise ProviderError("Gemini returned a non-JSON response") from exc

        return Transcript(text=_first_text(body).strip(), lang=lang)


def _first_text(body: object) -> str:
    """Pull the reply text out of a Gemini ``generateContent`` body.

    Every level is defensive on purpose: a blocked or empty candidate list is a
    normal response shape, not an error, and it should read as "she said
    nothing" rather than raise a ``KeyError`` from inside the voice route.
    """
    if not isinstance(body, dict):
        return ""
    candidates = body.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return ""
    first = candidates[0]
    if not isinstance(first, dict):
        return ""
    content = first.get("content")
    if not isinstance(content, dict):
        return ""
    parts = content.get("parts")
    if not isinstance(parts, list):
        return ""
    return "".join(
        part["text"]
        for part in parts
        if isinstance(part, dict) and isinstance(part.get("text"), str)
    )


def make_stt(settings: Settings) -> STTProvider:
    """Build an :class:`STTProvider` selected by ``settings.stt_provider``.

    Selection (env: ``FRIDAY_STT_PROVIDER``):

    * ``"faster-whisper"`` -> :class:`FasterWhisperSTT` (local, needs the voice
      extras and downloads model weights).
    * ``"gemini"`` -> :class:`GeminiSTT` (hosted; needs only ``GEMINI_API_KEY``).
    * ``"fake"`` -> :class:`FakeSTT` (tests / no transcription).

    An unknown value raises :class:`ProviderError` so a typo fails loudly at
    startup instead of silently transcribing nothing.
    """
    provider = settings.stt_provider.strip().lower()
    if provider == "fake":
        return FakeSTT()
    if provider in {"faster-whisper", "faster_whisper", "whisper"}:
        return FasterWhisperSTT()
    if provider == "gemini":
        key = settings.gemini_api_key
        return GeminiSTT(
            api_key=key.get_secret_value() if key is not None else None,
            model=settings.gemini_model,
        )
    raise ProviderError(
        f"unknown FRIDAY_STT_PROVIDER={settings.stt_provider!r}; "
        "expected one of: faster-whisper, gemini, fake"
    )
