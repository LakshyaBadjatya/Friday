"""``POST /voice`` — a single-shot spoken turn over HTTP (Phase 3 / Stage B).

Accepts an audio payload (either base64 in a JSON body or a multipart file
upload), transcribes it via the configured :class:`~friday.providers.stt.STTProvider`,
drives the orchestrator, synthesizes the reply with the configured
:class:`~friday.providers.tts.TTSProvider`, and returns
``{transcript, text, mode, audio_b64}``.

The whole route is behind ``FRIDAY_ENABLE_VOICE``: when voice is disabled it
responds ``404`` with ``{"detail": "voice disabled"}`` so the endpoint simply
does not exist for callers until the flag is set. STT/TTS come off ``app.state``
(populated at startup with fakes or the real, config-selected adapters), keeping
the route itself provider-agnostic and import-light.

A raised :class:`~friday.errors.FridayError` (e.g. a missing voice backend) is
mapped to a clean JSON error rather than leaking a 500 — mirroring ``/chat``.
"""

from __future__ import annotations

import base64
import binascii

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from friday.core.orchestrator import Orchestrator
from friday.core.state import GraphState
from friday.errors import FridayError, PermissionError, ProviderError
from friday.logging import get_logger
from friday.providers.stt import FakeSTT, STTProvider
from friday.providers.tts import FakeTTS, TTSProvider, VoiceConfig

logger = get_logger("friday.api.routes_voice")

router = APIRouter()


class VoiceRequest(BaseModel):
    """JSON body for ``POST /voice``: a base64-encoded audio payload."""

    audio_b64: str = Field(min_length=1)
    session_id: str = Field(default="voice", min_length=1, max_length=200)
    lang: str | None = None


class VoiceResponse(BaseModel):
    """Outbound spoken turn: transcript, reply text, mode, and reply audio."""

    transcript: str
    text: str
    mode: str
    audio_b64: str


class ErrorBody(BaseModel):
    """Typed error envelope for a mapped :class:`FridayError`."""

    error: str
    type: str


def _voice_enabled(request: Request) -> bool:
    """Whether voice is enabled, read off the startup settings on ``app.state``."""
    settings = getattr(request.app.state, "settings", None)
    return bool(getattr(settings, "enable_voice", False))


def _status_for(exc: FridayError) -> int:
    """Map a :class:`FridayError` subclass to an HTTP status code."""
    if isinstance(exc, PermissionError):
        return 403
    if isinstance(exc, ProviderError):
        return 502
    return 400


def _get_orchestrator(request: Request) -> Orchestrator:
    """Pull the process-wide orchestrator off app state (built at startup)."""
    orchestrator = getattr(request.app.state, "orchestrator", None)
    if not isinstance(orchestrator, Orchestrator):  # pragma: no cover - startup guard
        raise RuntimeError("orchestrator is not initialized on app.state")
    return orchestrator


def _get_stt(request: Request) -> STTProvider:
    """The STT provider off app state, defaulting to :class:`FakeSTT`."""
    stt = getattr(request.app.state, "voice_stt", None)
    if isinstance(stt, STTProvider):
        return stt
    return FakeSTT()


def _get_tts(request: Request) -> TTSProvider:
    """The TTS provider off app state, defaulting to :class:`FakeTTS`."""
    tts = getattr(request.app.state, "voice_tts", None)
    if isinstance(tts, TTSProvider):
        return tts
    return FakeTTS()


def _disabled() -> JSONResponse:
    """The canonical ``voice disabled`` 404 response."""
    return JSONResponse(status_code=404, content={"detail": "voice disabled"})


async def _read_multipart_audio(request: Request) -> tuple[bytes, str, str | None]:
    """Extract ``(audio, session_id, lang)`` from a multipart form upload.

    Multipart parsing requires the optional ``python-multipart`` library (kept out
    of the lock, like the voice backends). When it is absent a :class:`ValueError`
    surfaces, which the caller maps to a clean ``415`` with an install hint rather
    than a 500.
    """
    form = await request.form()
    field = form.get("file") or form.get("audio")
    if field is None:
        raise ValueError("multipart upload missing a 'file'/'audio' part")
    audio = await field.read() if hasattr(field, "read") else str(field).encode()
    session_field = form.get("session_id")
    session_id = str(session_field) if session_field is not None else "voice"
    lang_field = form.get("lang")
    lang = str(lang_field) if lang_field is not None else None
    return audio, session_id, lang


async def _read_json_audio(request: Request) -> tuple[bytes, str, str | None]:
    """Extract ``(audio, session_id, lang)`` from a base64 JSON body.

    Raises :class:`ValueError` on a malformed body or non-base64 payload; the
    caller maps that to a ``422``.
    """
    try:
        body = await request.json()
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError("expected JSON or multipart audio") from exc
    parsed = VoiceRequest.model_validate(body)
    try:
        audio = base64.b64decode(parsed.audio_b64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("audio_b64 is not valid base64") from exc
    return audio, parsed.session_id, parsed.lang


@router.post("/voice", response_model=None)
async def voice(request: Request) -> JSONResponse:
    """Handle one spoken turn: audio -> STT -> orchestrator -> TTS.

    Accepts the audio as either a multipart ``file`` upload or a base64
    ``audio_b64`` JSON field (selected by ``Content-Type``). Returns
    ``{transcript, text, mode, audio_b64}`` on success; ``404`` when voice is
    disabled; a mapped error body for a raised :class:`FridayError`.
    """
    if not _voice_enabled(request):
        return _disabled()

    content_type = request.headers.get("content-type", "")
    is_multipart = content_type.startswith("multipart/form-data")
    try:
        if is_multipart:
            audio, session_id, lang = await _read_multipart_audio(request)
        else:
            audio, session_id, lang = await _read_json_audio(request)
    except AssertionError:
        # starlette raises AssertionError("...python-multipart...") when the
        # multipart parser library is unavailable; surface a clean 415 + hint.
        return JSONResponse(
            status_code=415,
            content={
                "detail": (
                    "multipart audio requires the optional 'python-multipart' "
                    "library; send base64 JSON ({\"audio_b64\": ...}) instead."
                )
            },
        )
    except ValueError as exc:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    if not audio:
        return JSONResponse(status_code=422, content={"detail": "empty audio payload"})

    stt = _get_stt(request)
    tts = _get_tts(request)
    orchestrator = _get_orchestrator(request)

    try:
        transcript = await stt.transcribe(audio, lang)
        state = GraphState(session_id=session_id, user_input=transcript.text)
        result = await orchestrator.handle(state)
        response_text = result.response or ""
        audio_out = await tts.synthesize(response_text, VoiceConfig())
    except FridayError as exc:
        status = _status_for(exc)
        logger.warning(
            "voice turn raised FridayError",
            extra={"error_type": type(exc).__name__, "status": status},
        )
        return JSONResponse(
            status_code=status,
            content=ErrorBody(error=str(exc), type=type(exc).__name__).model_dump(),
        )

    payload = VoiceResponse(
        transcript=transcript.text,
        text=response_text,
        mode=result.mode.value,
        audio_b64=base64.b64encode(audio_out).decode("ascii"),
    ).model_dump()
    return JSONResponse(status_code=200, content=payload)
