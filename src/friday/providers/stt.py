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

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel

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
