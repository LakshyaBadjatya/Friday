"""``POST /emotion/enroll`` — personalize the V/A/D baseline to the owner's voice.

The owner submits a few short clips of themselves speaking *neutrally*; the route
runs the base emotion provider over them, builds an
:class:`~friday.providers.emotion.EmotionCalibration` (the offset that recentres
the owner's neutral to the population centre), persists it to the configured
``FRIDAY_EMOTION_CALIBRATION`` path, and applies it live. Behind
``FRIDAY_ENABLE_EMOTION``: 404s when emotion sensing is off.

Audio is base64-encoded 16 kHz mono WAV; the route decodes PCM with the stdlib
``wave`` module so no heavy audio dependency is imported here.
"""

from __future__ import annotations

import array
import base64
import io
import wave
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from friday.logging import get_logger
from friday.providers.emotion import CalibratedEmotion, EmotionProvider, enroll_owner

logger = get_logger("friday.api.routes_emotion")

router = APIRouter()

_DEFAULT_CALIBRATION_PATH = "models/emotion/owner_calibration.json"


class EnrollRequest(BaseModel):
    """Owner enrollment: base64 WAV clips of the owner speaking neutrally."""

    clips: list[str] = Field(min_length=1)
    session_id: str = Field(default="owner", min_length=1, max_length=200)


def _emotion_enabled(request: Request) -> bool:
    settings = getattr(request.app.state, "settings", None)
    return bool(getattr(settings, "enable_emotion", False))


def _decode_wav(b64: str) -> bytes:
    """Decode a base64 WAV into raw 16-bit PCM mono bytes (stdlib only)."""
    raw = base64.b64decode(b64)
    with wave.open(io.BytesIO(raw)) as wf:
        nch = wf.getnchannels()
        data = wf.readframes(wf.getnframes())
    if nch > 1:  # collapse to the first channel
        samples = array.array("h", data)
        data = array.array("h", samples[0::nch]).tobytes()
    return data


@router.post("/emotion/enroll", response_model=None)
async def emotion_enroll(req: EnrollRequest, request: Request) -> JSONResponse | dict:
    """Build + persist the owner's emotion calibration from their neutral clips."""
    if not _emotion_enabled(request):
        return JSONResponse(status_code=404, content={"detail": "emotion disabled"})

    base = getattr(request.app.state, "emotion_base_provider", None)
    if not isinstance(base, EmotionProvider):
        return JSONResponse(
            status_code=503, content={"detail": "emotion provider unavailable"}
        )
    try:
        clips = [_decode_wav(c) for c in req.clips]
    except Exception as exc:
        return JSONResponse(status_code=400, content={"detail": f"bad audio: {exc}"})

    calibration = await enroll_owner(base, clips)

    settings = request.app.state.settings
    path = getattr(settings, "emotion_calibration", "") or _DEFAULT_CALIBRATION_PATH
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(calibration.model_dump_json(indent=2), encoding="utf-8")

    # Apply live: re-wrap the running analyzer's provider so the calibration takes
    # effect immediately (best-effort; the persisted file is the source of truth).
    analyzer = getattr(request.app.state, "emotion_analyzer", None)
    if analyzer is not None:
        analyzer._provider = CalibratedEmotion(base, calibration)  # noqa: SLF001
    logger.info("emotion calibration written to %s (%d clips)", path, len(clips))

    return {"clips": len(clips), "path": path, "calibration": calibration.model_dump()}
