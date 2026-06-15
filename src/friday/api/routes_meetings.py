"""``/meetings`` — the flagged meeting-capture REST API (Tier 1).

Four surfaces, all gated behind ``FRIDAY_ENABLE_MEETINGS`` (read off the startup
settings on ``app.state``); when the flag is off every one is ``404`` so the
feature simply does not exist for callers (mirroring ``/rag`` and ``/reminders``):

* ``POST   /meetings/capture`` accepts either a JSON ``{title, audio_b64}`` body
  *or* a ``multipart/form-data`` file upload (the title is the ``title`` form
  field, falling back to the uploaded filename). The audio is transcribed,
  summarized (non-fatal), optionally ingested for retrieval, stored, and the
  resulting :class:`~friday.meetings.capture.MeetingNotes` returned.
* ``GET    /meetings`` -> ``{meetings, count}`` (most-recent first).
* ``GET    /meetings/{id}`` -> the stored notes (404 if no such meeting).
* ``DELETE /meetings/{id}`` -> ``{id, removed}`` (idempotent; ``removed`` is 0
  when the meeting was already gone).

The route reads the shared :class:`~friday.meetings.capture.MeetingCapture`
pipeline off ``app.state.meeting_capture`` and the
:class:`~friday.meetings.store.SQLiteMeetingStore` off ``app.state.meeting_store``
(``app.py`` builds and stashes both when the flag is on).

Multipart parsing reuses the small, dependency-free parser from
:mod:`friday.api.routes_rag` (``python-multipart`` is intentionally not a project
dependency), so the upload body is split on its boundary by hand.
"""

from __future__ import annotations

import base64
import binascii

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError

from friday.api.routes_rag import _parse_multipart, _source_id_from_filename
from friday.logging import get_logger
from friday.meetings.capture import MeetingCapture
from friday.meetings.store import SQLiteMeetingStore

logger = get_logger("friday.api.routes_meetings")

router = APIRouter()

# Upper bound on meetings returned by ``GET /meetings`` — generous but bounded.
_LIST_LIMIT = 200


class CaptureRequest(BaseModel):
    """JSON body for ``POST /meetings/capture`` (the non-upload path)."""

    title: str = Field(min_length=1, max_length=512)
    audio_b64: str = Field(min_length=1)


def _meetings_enabled(request: Request) -> bool:
    """Whether meetings are enabled, read off the startup settings on app state."""
    settings = getattr(request.app.state, "settings", None)
    return bool(getattr(settings, "enable_meetings", False))


def _disabled() -> JSONResponse:
    """The canonical ``meetings disabled`` 404 response."""
    return JSONResponse(status_code=404, content={"detail": "meetings disabled"})


def _get_capture(request: Request) -> MeetingCapture:
    """Pull the process-wide :class:`MeetingCapture` pipeline off ``app.state``."""
    capture = getattr(request.app.state, "meeting_capture", None)
    if not isinstance(capture, MeetingCapture):  # pragma: no cover - startup guard
        raise RuntimeError("meeting capture is not initialized on app.state")
    return capture


def _get_store(request: Request) -> SQLiteMeetingStore:
    """Pull the process-wide meeting store off ``app.state``."""
    store = getattr(request.app.state, "meeting_store", None)
    if not isinstance(store, SQLiteMeetingStore):  # pragma: no cover - startup guard
        raise RuntimeError("meeting store is not initialized on app.state")
    return store


@router.post("/meetings/capture", response_model=None)
async def capture_meeting(request: Request) -> JSONResponse:
    """Capture a meeting (JSON or multipart upload); 404 when disabled.

    JSON path: ``{title, audio_b64}`` -> decode base64 audio -> capture -> store.
    Multipart path: the first uploaded file's bytes are the audio and its
    filename supplies the title. A body that is neither a valid JSON capture
    request nor a multipart file upload (or carries malformed base64) is ``422``.
    The capture pipeline transcribes, summarizes (non-fatal), optionally ingests
    the transcript for retrieval, then persists and returns the stored notes.
    """
    if not _meetings_enabled(request):
        return _disabled()

    capture = _get_capture(request)
    store = _get_store(request)
    content_type = request.headers.get("content-type", "")

    if content_type.startswith("multipart/form-data"):
        body = await request.body()
        parsed = _parse_multipart(body, content_type)
        if parsed is None:
            return JSONResponse(
                status_code=422,
                content={"detail": "expected a multipart file upload"},
            )
        filename, audio = parsed
        # The first uploaded file's filename (dir + extension stripped) supplies
        # the meeting title — the multipart parser only extracts the file part.
        title = _source_id_from_filename(filename)
    else:
        try:
            raw = await request.json()
        except (ValueError, UnicodeDecodeError):
            return JSONResponse(
                status_code=422, content={"detail": "expected a JSON body"}
            )
        try:
            parsed_body = CaptureRequest.model_validate(raw)
        except ValidationError as exc:
            return JSONResponse(status_code=422, content={"detail": str(exc)})
        try:
            audio = base64.b64decode(parsed_body.audio_b64, validate=True)
        except (binascii.Error, ValueError):
            return JSONResponse(
                status_code=422, content={"detail": "audio_b64 is not valid base64"}
            )
        title = parsed_body.title

    notes = await capture.process(title, audio)
    stored = store.add(notes)
    return JSONResponse(status_code=200, content=stored.model_dump())


@router.get("/meetings", response_model=None)
async def list_meetings(request: Request) -> JSONResponse:
    """List stored meetings, most-recent first; 404 when disabled."""
    if not _meetings_enabled(request):
        return _disabled()
    store = _get_store(request)
    meetings = store.list_meetings(limit=_LIST_LIMIT)
    return JSONResponse(
        status_code=200,
        content={
            "meetings": [m.model_dump() for m in meetings],
            "count": len(meetings),
        },
    )


@router.get("/meetings/{meeting_id}", response_model=None)
async def get_meeting(request: Request, meeting_id: int) -> JSONResponse:
    """Return one meeting's notes by id; 404 when disabled or no such meeting."""
    if not _meetings_enabled(request):
        return _disabled()
    store = _get_store(request)
    notes = store.get(meeting_id)
    if notes is None:
        return JSONResponse(
            status_code=404,
            content={"detail": f"no meeting with id {meeting_id}"},
        )
    return JSONResponse(status_code=200, content=notes.model_dump())


@router.delete("/meetings/{meeting_id}", response_model=None)
async def delete_meeting(request: Request, meeting_id: int) -> JSONResponse:
    """Delete a meeting by id; 404 when disabled. Idempotent (``removed`` 0/1)."""
    if not _meetings_enabled(request):
        return _disabled()
    store = _get_store(request)
    removed = store.delete(meeting_id)
    return JSONResponse(
        status_code=200, content={"id": meeting_id, "removed": removed}
    )
