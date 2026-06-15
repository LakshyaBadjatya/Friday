"""``/perception`` — the flagged perception REST API (privacy-heavy, off by default).

Five surfaces, all gated behind ``FRIDAY_ENABLE_PERCEPTION`` (read off the startup
settings on ``app.state``); when the flag is off every one is ``404`` so the
feature simply does not exist for callers (mirroring ``/voice`` and ``/study``):

* ``POST /perception/vision`` ``{image_b64}`` -> ``{detections, count}`` — object
  detection over a base64-encoded image.
* ``POST /perception/ocr`` ``{image_b64}`` -> ``{text}`` — OCR over a base64 image.
* ``GET  /perception/clipboard`` -> ``{text}`` — the current clipboard contents.
* ``POST /perception/clipboard`` ``{text}`` -> ``{ok}`` — write the clipboard.
* ``POST /perception/screen`` -> ``{ocr_text, detections}`` — capture the screen
  and describe it (capture -> ocr + detect).

The route reads the shared :class:`~friday.perception.screen.PerceptionService`
off ``app.state.perception`` (``app.py`` builds and stashes it — from fakes by
default — when the flag is on). The service is provider-agnostic, so this route
stays import-light and never pulls in a heavy perception library.
"""

from __future__ import annotations

import base64
import binascii

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError

from friday.errors import FridayError, ProviderError
from friday.logging import get_logger
from friday.perception.screen import PerceptionService

logger = get_logger("friday.api.routes_perception")

router = APIRouter()


class ImageRequest(BaseModel):
    """JSON body carrying a base64-encoded image (vision + OCR)."""

    image_b64: str = Field(min_length=1)


class ClipboardWriteRequest(BaseModel):
    """JSON body for ``POST /perception/clipboard``."""

    text: str = Field(max_length=1_000_000)


def _perception_enabled(request: Request) -> bool:
    """Whether perception is enabled, read off the startup settings on app state."""
    settings = getattr(request.app.state, "settings", None)
    return bool(getattr(settings, "enable_perception", False))


def _disabled() -> JSONResponse:
    """The canonical ``perception disabled`` 404 response."""
    return JSONResponse(status_code=404, content={"detail": "perception disabled"})


def _get_service(request: Request) -> PerceptionService:
    """Pull the process-wide perception service off ``app.state``."""
    service = getattr(request.app.state, "perception", None)
    if not isinstance(service, PerceptionService):  # pragma: no cover - startup guard
        raise RuntimeError("perception service is not initialized on app.state")
    return service


async def _decode_image(
    request: Request,
) -> tuple[bytes | None, JSONResponse | None]:
    """Parse the JSON body and base64-decode its ``image_b64``; return (bytes, err)."""
    try:
        raw = await request.json()
    except (ValueError, UnicodeDecodeError):
        return None, JSONResponse(
            status_code=422, content={"detail": "expected a JSON body"}
        )
    try:
        body = ImageRequest.model_validate(raw)
    except ValidationError as exc:
        return None, JSONResponse(status_code=422, content={"detail": str(exc)})
    try:
        image = base64.b64decode(body.image_b64, validate=True)
    except (binascii.Error, ValueError):
        return None, JSONResponse(
            status_code=422, content={"detail": "image_b64 is not valid base64"}
        )
    if not image:
        return None, JSONResponse(
            status_code=422, content={"detail": "empty image payload"}
        )
    return image, None


def _provider_error(exc: ProviderError) -> JSONResponse:
    """Map a missing-backend :class:`ProviderError` to a clean 502 + hint."""
    logger.warning("perception provider error", extra={"error": str(exc)})
    return JSONResponse(status_code=502, content={"detail": str(exc)})


@router.post("/perception/vision", response_model=None)
async def detect_objects(request: Request) -> JSONResponse:
    """Detect objects in a base64 image; 404 when disabled, 422 on a bad body."""
    if not _perception_enabled(request):
        return _disabled()
    image, error = await _decode_image(request)
    if error is not None:
        return error
    assert image is not None
    service = _get_service(request)
    try:
        detections = await service.vision.detect(image)
    except ProviderError as exc:
        return _provider_error(exc)
    return JSONResponse(
        status_code=200,
        content={
            "detections": [d.model_dump() for d in detections],
            "count": len(detections),
        },
    )


@router.post("/perception/ocr", response_model=None)
async def read_text(request: Request) -> JSONResponse:
    """Read text from a base64 image; 404 when disabled, 422 on a bad body."""
    if not _perception_enabled(request):
        return _disabled()
    image, error = await _decode_image(request)
    if error is not None:
        return error
    assert image is not None
    service = _get_service(request)
    try:
        text = await service.ocr.read(image)
    except ProviderError as exc:
        return _provider_error(exc)
    return JSONResponse(status_code=200, content={"text": text})


@router.get("/perception/clipboard", response_model=None)
async def read_clipboard(request: Request) -> JSONResponse:
    """Return the current clipboard contents; 404 when disabled."""
    if not _perception_enabled(request):
        return _disabled()
    service = _get_service(request)
    try:
        text = service.clipboard.read()
    except ProviderError as exc:
        return _provider_error(exc)
    return JSONResponse(status_code=200, content={"text": text})


@router.post("/perception/clipboard", response_model=None)
async def write_clipboard(request: Request) -> JSONResponse:
    """Write the clipboard from a JSON body; 404 when disabled, 422 on a bad body."""
    if not _perception_enabled(request):
        return _disabled()
    try:
        raw = await request.json()
    except (ValueError, UnicodeDecodeError):
        return JSONResponse(status_code=422, content={"detail": "expected a JSON body"})
    try:
        body = ClipboardWriteRequest.model_validate(raw)
    except ValidationError as exc:
        return JSONResponse(status_code=422, content={"detail": str(exc)})
    service = _get_service(request)
    try:
        service.clipboard.write(body.text)
    except ProviderError as exc:
        return _provider_error(exc)
    return JSONResponse(status_code=200, content={"ok": True})


@router.post("/perception/screen", response_model=None)
async def describe_screen(request: Request) -> JSONResponse:
    """Capture the screen and describe it (ocr + detections); 404 when disabled."""
    if not _perception_enabled(request):
        return _disabled()
    service = _get_service(request)
    try:
        description = await service.describe_screen()
    except ProviderError as exc:
        return _provider_error(exc)
    except FridayError as exc:  # pragma: no cover - defensive: other domain errors
        logger.warning("perception screen error", extra={"error": str(exc)})
        return JSONResponse(status_code=400, content={"detail": str(exc)})
    return JSONResponse(status_code=200, content=description.model_dump())
