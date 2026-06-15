"""``/calendar`` — the flagged Google Calendar surface (Tier 3; default off).

Two surfaces, both gated behind ``FRIDAY_ENABLE_CALENDAR`` (read lazily off
:func:`~friday.config.get_settings` so the router works mounted on a bare
``FastAPI()`` app, before ``app.py`` wiring exists); when the flag is off both
are ``404`` so the feature simply does not exist for callers (mirroring
``/maps`` / ``/studio`` / ``/reminders``):

* ``GET  /calendar/events?day=YYYY-MM-DD`` -> ``{events, count}`` — the day's
  events on the user's ``primary`` calendar (defaults to today, UTC). Read-only.
* ``POST /calendar/events`` ``{event: {...}}`` -> ``{event}`` — creates the event
  and returns the created-event JSON. This is the ONLY mutating call.

The Google OAuth bearer token is a :class:`~pydantic.SecretStr` on
:class:`~friday.config.Settings`; it is read via ``get_secret_value()`` ONLY to
build the per-request :class:`~friday.integrations.calendar.GoogleCalendarClient`
header and is never logged or echoed in a response. A missing token (or any
Calendar REST failure) surfaces as a clean ``400`` JSON error rather than a
leaked ``500``.
"""

from __future__ import annotations

from datetime import UTC, datetime

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError

from friday.config import get_settings
from friday.errors import FridayError
from friday.integrations.calendar import GoogleCalendarClient, whats_my_day
from friday.logging import get_logger

logger = get_logger("friday.api.routes_calendar")

router = APIRouter()


class CreateEventRequest(BaseModel):
    """JSON body for ``POST /calendar/events``."""

    event: dict[str, object]


def _calendar_enabled() -> bool:
    """Whether the calendar surface is enabled, read lazily from settings."""
    return bool(getattr(get_settings(), "enable_calendar", False))


def _disabled() -> JSONResponse:
    """The canonical ``calendar disabled`` 404 response."""
    return JSONResponse(status_code=404, content={"detail": "calendar disabled"})


def _build_client() -> GoogleCalendarClient:
    """Build a Calendar client from settings (token read lazily, never logged).

    The bearer token is pulled from the :class:`~pydantic.SecretStr` only to seed
    the client; a missing token does not raise here (the client raises a clear
    :class:`~friday.integrations.calendar.CalendarError` on first use, which the
    route maps to a clean 400).
    """
    secret = getattr(get_settings(), "google_oauth_token", None)
    token = secret.get_secret_value() if secret is not None else None
    # A fresh AsyncClient per request, owned by the route (closed in the handler).
    return GoogleCalendarClient(token, http=httpx.AsyncClient())


@router.get("/calendar/events", response_model=None)
async def list_calendar_events(request: Request, day: str | None = None) -> JSONResponse:
    """List the day's events; 404 when disabled, 400 on a missing token / API error.

    ``day`` defaults to today (UTC) when omitted. The result is
    ``{events, count}`` (the parsed ``items`` list and its length).
    """
    if not _calendar_enabled():
        return _disabled()

    target_day = day or datetime.now(UTC).strftime("%Y-%m-%d")
    client = _build_client()
    try:
        events = await whats_my_day(client, day=target_day)
    except FridayError as exc:
        logger.warning(
            "calendar list failed", extra={"error_type": type(exc).__name__}
        )
        return JSONResponse(status_code=400, content={"detail": str(exc)})
    finally:
        await client.aclose()

    return JSONResponse(
        status_code=200, content={"events": events, "count": len(events)}
    )


@router.post("/calendar/events", response_model=None)
async def create_calendar_event(request: Request) -> JSONResponse:
    """Create an event; 404 when disabled, 422 on bad body, 400 on a token/API error.

    The body is ``{"event": {...}}`` (the Calendar v3 event resource). On success
    the created-event JSON is returned as ``{"event": {...}}``.
    """
    if not _calendar_enabled():
        return _disabled()

    try:
        raw = await request.json()
    except (ValueError, UnicodeDecodeError):
        return JSONResponse(status_code=422, content={"detail": "expected a JSON body"})
    try:
        body = CreateEventRequest.model_validate(raw)
    except ValidationError as exc:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    client = _build_client()
    try:
        created = await client.create_event(body.event)
    except FridayError as exc:
        logger.warning(
            "calendar create failed", extra={"error_type": type(exc).__name__}
        )
        return JSONResponse(status_code=400, content={"detail": str(exc)})
    finally:
        await client.aclose()

    return JSONResponse(status_code=200, content={"event": created})
