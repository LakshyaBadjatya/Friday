"""Unit tests for :class:`friday.integrations.calendar.GoogleCalendarClient` (offline).

Every HTTP call is mocked with ``respx`` against the Google Calendar v3 REST
endpoints — no live network, no real OAuth. The client is a thin ``httpx``
adapter (mirroring :mod:`friday.tools.web_search` / :mod:`friday.n8n.client`):
the OAuth bearer token is held as a plain ``str | None`` (sourced from a
:class:`~pydantic.SecretStr` in config so it never logs) and sent ONLY as the
``Authorization: Bearer`` header.

Covered:
* ``list_events`` issues an authenticated ``GET`` and parses the ``items`` list.
* ``create_event`` (the only mutating call) issues an authenticated ``POST`` and
  returns the created-event JSON; the bearer token rides only in the header.
* A missing token raises a clear :class:`CalendarError` BEFORE any network I/O —
  for both the read and the mutating call.
* ``whats_my_day`` is a thin helper over ``list_events`` for a single day window.
* The token never appears in the client's ``repr``/``str``.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from friday.integrations.calendar import (
    CalendarError,
    GoogleCalendarClient,
    whats_my_day,
)

_PRIMARY = "https://www.googleapis.com/calendar/v3/calendars/primary/events"

_LIST_BODY = {
    "items": [
        {
            "id": "ev1",
            "summary": "Standup",
            "start": {"dateTime": "2026-06-15T09:00:00Z"},
            "end": {"dateTime": "2026-06-15T09:15:00Z"},
        },
        {
            "id": "ev2",
            "summary": "Lunch",
            "start": {"dateTime": "2026-06-15T12:00:00Z"},
            "end": {"dateTime": "2026-06-15T13:00:00Z"},
        },
    ]
}


@respx.mock
async def test_list_events_parses_items() -> None:
    """``list_events`` issues an authed GET and returns the parsed ``items`` list."""
    route = respx.get(_PRIMARY).mock(return_value=httpx.Response(200, json=_LIST_BODY))

    async with httpx.AsyncClient() as http:
        client = GoogleCalendarClient("tok-secret", http=http)
        events = await client.list_events(
            time_min="2026-06-15T00:00:00Z",
            time_max="2026-06-16T00:00:00Z",
        )

    assert [e["id"] for e in events] == ["ev1", "ev2"]
    sent = route.calls.last.request
    # The bearer token rides ONLY in the Authorization header.
    assert sent.headers["Authorization"] == "Bearer tok-secret"
    # The time window is passed as query params.
    assert sent.url.params["timeMin"] == "2026-06-15T00:00:00Z"
    assert sent.url.params["timeMax"] == "2026-06-16T00:00:00Z"


@respx.mock
async def test_list_events_empty_items_returns_empty_list() -> None:
    """A response with no ``items`` key yields an empty list (never raises)."""
    respx.get(_PRIMARY).mock(return_value=httpx.Response(200, json={}))

    async with httpx.AsyncClient() as http:
        client = GoogleCalendarClient("tok-secret", http=http)
        events = await client.list_events(
            time_min="2026-06-15T00:00:00Z",
            time_max="2026-06-16T00:00:00Z",
        )

    assert events == []


@respx.mock
async def test_create_event_posts_and_returns_created() -> None:
    """``create_event`` POSTs the event and returns the created-event JSON."""
    created = {"id": "new1", "summary": "Coffee", "status": "confirmed"}
    route = respx.post(_PRIMARY).mock(return_value=httpx.Response(200, json=created))

    event = {
        "summary": "Coffee",
        "start": {"dateTime": "2026-06-15T15:00:00Z"},
        "end": {"dateTime": "2026-06-15T15:30:00Z"},
    }
    async with httpx.AsyncClient() as http:
        client = GoogleCalendarClient("tok-secret", http=http)
        result = await client.create_event(event)

    assert result == created
    sent = route.calls.last.request
    assert sent.headers["Authorization"] == "Bearer tok-secret"
    # The event body is sent as JSON (the mutating call).
    assert sent.method == "POST"
    assert b"Coffee" in sent.content


@respx.mock
async def test_list_events_non_2xx_raises_calendar_error() -> None:
    """A non-2xx status surfaces as a typed :class:`CalendarError` (no leak)."""
    respx.get(_PRIMARY).mock(return_value=httpx.Response(403, text="forbidden"))

    async with httpx.AsyncClient() as http:
        client = GoogleCalendarClient("tok-secret", http=http)
        with pytest.raises(CalendarError) as exc_info:
            await client.list_events(
                time_min="2026-06-15T00:00:00Z",
                time_max="2026-06-16T00:00:00Z",
            )
    assert "403" in str(exc_info.value)


async def test_missing_token_list_raises_before_network() -> None:
    """A missing token raises a clear error BEFORE any network I/O (read path)."""
    async with httpx.AsyncClient() as http:
        client = GoogleCalendarClient(None, http=http)
        with pytest.raises(CalendarError) as exc_info:
            await client.list_events(
                time_min="2026-06-15T00:00:00Z",
                time_max="2026-06-16T00:00:00Z",
            )
    assert "token" in str(exc_info.value).lower()


async def test_missing_token_create_raises_before_network() -> None:
    """A missing token raises a clear error BEFORE any network I/O (mutating path)."""
    async with httpx.AsyncClient() as http:
        client = GoogleCalendarClient(None, http=http)
        with pytest.raises(CalendarError) as exc_info:
            await client.create_event({"summary": "x"})
    assert "token" in str(exc_info.value).lower()


@respx.mock
async def test_whats_my_day_lists_events_for_the_day() -> None:
    """``whats_my_day`` is a thin helper returning that day's events."""
    route = respx.get(_PRIMARY).mock(return_value=httpx.Response(200, json=_LIST_BODY))

    async with httpx.AsyncClient() as http:
        client = GoogleCalendarClient("tok-secret", http=http)
        events = await whats_my_day(client, day="2026-06-15")

    assert [e["id"] for e in events] == ["ev1", "ev2"]
    params = route.calls.last.request.url.params
    # The day expands to a [00:00, next-00:00) UTC window.
    assert params["timeMin"] == "2026-06-15T00:00:00Z"
    assert params["timeMax"] == "2026-06-16T00:00:00Z"


def test_token_never_in_repr() -> None:
    """The bearer token never leaks into the client's ``repr``/``str``."""
    client = GoogleCalendarClient("super-secret-oauth-token", http=None)  # type: ignore[arg-type]
    assert "super-secret-oauth-token" not in repr(client)
    assert "super-secret-oauth-token" not in str(client)
