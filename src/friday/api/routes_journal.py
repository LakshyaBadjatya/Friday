"""``/journal`` — the flagged auto-journaling REST API (Tier 2).

Three surfaces, all gated behind ``FRIDAY_ENABLE_JOURNAL`` (read off the startup
settings on ``app.state``); when the flag is off every one is ``404`` so the
feature simply does not exist for callers (mirroring ``/briefing`` and
``/reminders``):

* ``GET  /journal`` -> ``{entries, count}`` (most-recent date first).
* ``GET  /journal/{date}`` -> the stored entry for ``date`` (404 if no such day).
* ``POST /journal/build`` ``{date?}`` -> build (aggregate the day's events) and
  save the :class:`~friday.journal.service.JournalEntry` for the given ``date``
  (``YYYY-MM-DD``), defaulting to UTC today when omitted.

The route reads the shared :class:`~friday.journal.service.JournalService` off
``app.state.journal_service`` and the
:class:`~friday.journal.store.SQLiteJournalStore` off ``app.state.journal_store``
(``app.py`` builds and stashes both when the flag is on), so an on-demand HTTP
build and the proactive scheduler ``"journal"`` action assemble + persist through
the same service + store. ``utcnow`` is read here at request time only — the
tested ``JournalService.build_entry(day)`` unit stays clock-injected; this
endpoint is the "build today now" seam.
"""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError

from friday.journal.service import JournalService
from friday.journal.store import SQLiteJournalStore
from friday.logging import get_logger

logger = get_logger("friday.api.routes_journal")

router = APIRouter()

# Upper bound on entries returned by ``GET /journal`` — generous but bounded.
_LIST_LIMIT = 365


class BuildRequest(BaseModel):
    """JSON body for ``POST /journal/build`` (the ``date`` is optional)."""

    date: str | None = Field(default=None, min_length=10, max_length=10)


def _journal_enabled(request: Request) -> bool:
    """Whether journaling is enabled, read off the startup settings on app state."""
    settings = getattr(request.app.state, "settings", None)
    return bool(getattr(settings, "enable_journal", False))


def _disabled() -> JSONResponse:
    """The canonical ``journal disabled`` 404 response."""
    return JSONResponse(status_code=404, content={"detail": "journal disabled"})


def _get_service(request: Request) -> JournalService:
    """Pull the process-wide :class:`JournalService` off ``app.state`` (startup)."""
    service = getattr(request.app.state, "journal_service", None)
    if not isinstance(service, JournalService):  # pragma: no cover - startup guard
        raise RuntimeError("journal service is not initialized on app.state")
    return service


def _get_store(request: Request) -> SQLiteJournalStore:
    """Pull the process-wide journal store off ``app.state``."""
    store = getattr(request.app.state, "journal_store", None)
    if not isinstance(store, SQLiteJournalStore):  # pragma: no cover - startup guard
        raise RuntimeError("journal store is not initialized on app.state")
    return store


@router.get("/journal", response_model=None)
async def list_journal(request: Request) -> JSONResponse:
    """List journal entries, most-recent date first; 404 when disabled."""
    if not _journal_enabled(request):
        return _disabled()
    store = _get_store(request)
    entries = store.list_entries(limit=_LIST_LIMIT)
    return JSONResponse(
        status_code=200,
        content={
            "entries": [entry.model_dump() for entry in entries],
            "count": len(entries),
        },
    )


@router.post("/journal/build", response_model=None)
async def build_journal(request: Request) -> JSONResponse:
    """Build + save the entry for the given/UTC-today date; 404 when disabled.

    The optional ``{date}`` body (``YYYY-MM-DD``) selects the day to build;
    omitted/empty bodies default to UTC today. A malformed body is ``422``.
    """
    if not _journal_enabled(request):
        return _disabled()

    raw: object = {}
    try:
        raw = await request.json()
    except (ValueError, UnicodeDecodeError):
        # An empty/absent body is fine — default to UTC today.
        raw = {}
    if raw is None:
        raw = {}
    try:
        body = BuildRequest.model_validate(raw)
    except ValidationError as exc:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    day = _resolve_day(body.date)
    if day is None:
        return JSONResponse(
            status_code=422, content={"detail": "date must be YYYY-MM-DD"}
        )

    service = _get_service(request)
    store = _get_store(request)
    entry = await service.build_entry(day)
    store.save(entry)
    return JSONResponse(status_code=200, content=entry.model_dump())


@router.get("/journal/{date}", response_model=None)
async def get_journal(request: Request, date: str) -> JSONResponse:
    """Return the stored entry for ``date``; 404 when disabled or no such day."""
    if not _journal_enabled(request):
        return _disabled()
    store = _get_store(request)
    entry = store.get(date)
    if entry is None:
        return JSONResponse(
            status_code=404, content={"detail": f"no journal entry for {date}"}
        )
    return JSONResponse(status_code=200, content=entry.model_dump())


def _resolve_day(date: str | None) -> datetime | None:
    """Resolve the ``date`` string to a UTC ``datetime``, or UTC-today when ``None``.

    Returns ``None`` for a malformed (non ``YYYY-MM-DD``) date so the caller can
    surface a ``422``. ``utcnow`` is only read on the default path (no date given)
    — the tested ``build_entry(day)`` unit takes ``day`` injected.
    """
    if date is None:
        return datetime.now(UTC)
    try:
        parsed = datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        return None
    return parsed.replace(tzinfo=UTC)
