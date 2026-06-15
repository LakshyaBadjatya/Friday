"""A thin ``httpx`` adapter over the Google Calendar v3 REST API.

:class:`GoogleCalendarClient` mirrors the keyless-tool style of
:mod:`friday.tools.web_search` and :mod:`friday.n8n.client`: a small async
surface built on an injected ``httpx.AsyncClient``, with typed failures
(:class:`CalendarError`) rather than bare exceptions leaking out. It uses NO
Google SDK — every call is a plain HTTPS request to the Calendar v3 REST API.

Surface (on the user's ``primary`` calendar):

* :meth:`GoogleCalendarClient.list_events` — ``GET .../events?timeMin&timeMax``;
  returns the parsed ``items`` list (read-only, idempotent).
* :meth:`GoogleCalendarClient.create_event` — ``POST .../events`` with the event
  body; returns the created-event JSON. This is the ONLY mutating call.
* :func:`whats_my_day` — a thin helper that expands a ``YYYY-MM-DD`` day into a
  ``[00:00, next-00:00)`` UTC window and lists that day's events.

SECURITY: the OAuth bearer token originates from a :class:`~pydantic.SecretStr`
in config; here it is held as a plain ``str | None`` but is ONLY ever placed in
the ``Authorization: Bearer`` request header and is never logged (no ``logger``
call includes it, and error messages carry the status/endpoint — never the
token). A missing token raises a clear :class:`CalendarError` BEFORE any network
I/O.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx

from friday.errors import FridayError

logger = logging.getLogger("friday.integrations.calendar")

#: Base of the Calendar v3 REST API events collection for the ``primary`` calendar.
_EVENTS_URL = "https://www.googleapis.com/calendar/v3/calendars/primary/events"
#: Default per-request wall-clock budget (seconds) for every Calendar REST call.
_DEFAULT_TIMEOUT = 15.0


class CalendarError(FridayError):
    """A Google Calendar REST call failed (missing token, transport, or non-2xx)."""


class GoogleCalendarClient:
    """Async ``httpx`` client for the subset of Google Calendar v3 FRIDAY uses.

    Args:
        token: The Google OAuth bearer token, or ``None`` when unset. Sent as
            ``Authorization: Bearer`` on every call; both :meth:`list_events` and
            :meth:`create_event` raise a clear :class:`CalendarError` when it is
            ``None`` (before any network I/O).
        http: An injected ``httpx.AsyncClient`` the caller owns (so the client is
            trivially testable with ``respx`` and shares connection pooling).
        timeout: Per-request wall-clock budget in seconds.
    """

    def __init__(
        self,
        token: str | None,
        *,
        http: httpx.AsyncClient,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        self._token = token
        self._http = http
        self._timeout = timeout

    @property
    def has_token(self) -> bool:
        """Whether a bearer token is configured (without exposing it)."""
        return bool(self._token)

    def __repr__(self) -> str:
        """A token-free repr (the secret never leaks into repr/str/logs)."""
        return f"GoogleCalendarClient(has_token={self.has_token})"

    async def aclose(self) -> None:
        """Close the underlying ``httpx.AsyncClient`` (when the client owns it)."""
        await self._http.aclose()

    def _auth_headers(self) -> dict[str, str]:
        """The authenticated-request headers; raises when no token is set.

        The token is placed ONLY here, in the ``Authorization`` header — never in
        a log line or an error message.
        """
        if not self._token:
            raise CalendarError(
                "google oauth token not set — configure FRIDAY_GOOGLE_OAUTH_TOKEN "
                "to use the calendar"
            )
        return {"Authorization": f"Bearer {self._token}"}

    async def list_events(
        self, time_min: str, time_max: str
    ) -> list[dict[str, Any]]:
        """``GET .../events``; return the parsed ``items`` list (read-only).

        Raises :class:`CalendarError` when no token is configured (before any
        network I/O), on a transport error, or on a non-2xx response. A response
        with no ``items`` yields an empty list (never raises).
        """
        headers = self._auth_headers()
        params = {
            "timeMin": time_min,
            "timeMax": time_max,
            "singleEvents": "true",
            "orderBy": "startTime",
        }
        try:
            response = await self._http.get(
                _EVENTS_URL, params=params, headers=headers, timeout=self._timeout
            )
        except httpx.HTTPError as exc:
            logger.warning("calendar list_events transport error: %s", exc)
            raise CalendarError(f"calendar list request failed: {exc}") from exc

        if not response.is_success:
            logger.warning(
                "calendar list_events returned HTTP %d", response.status_code
            )
            raise CalendarError(
                f"calendar list returned HTTP {response.status_code}: "
                f"{_safe_body(response)}"
            )

        try:
            body = response.json()
        except ValueError as exc:
            raise CalendarError(
                f"calendar list returned a non-JSON body: {exc}"
            ) from exc
        if not isinstance(body, dict):
            raise CalendarError("calendar list returned an unexpected (non-object) body")
        items = body.get("items", [])
        return [item for item in items if isinstance(item, dict)]

    async def create_event(self, event: dict[str, Any]) -> dict[str, Any]:
        """``POST .../events``; return the created-event JSON (the ONLY mutation).

        Raises :class:`CalendarError` when no token is configured (before any
        network I/O), on a transport error, or on a non-2xx response.
        """
        headers = self._auth_headers()
        try:
            response = await self._http.post(
                _EVENTS_URL, json=event, headers=headers, timeout=self._timeout
            )
        except httpx.HTTPError as exc:
            logger.warning("calendar create_event transport error: %s", exc)
            raise CalendarError(f"calendar create request failed: {exc}") from exc

        if not response.is_success:
            logger.warning(
                "calendar create_event returned HTTP %d", response.status_code
            )
            raise CalendarError(
                f"calendar create returned HTTP {response.status_code}: "
                f"{_safe_body(response)}"
            )

        try:
            body = response.json()
        except ValueError as exc:
            raise CalendarError(
                f"calendar create returned a non-JSON body: {exc}"
            ) from exc
        if not isinstance(body, dict):
            raise CalendarError(
                "calendar create returned an unexpected (non-object) body"
            )
        return body


def _day_window(day: str) -> tuple[str, str]:
    """Expand a ``YYYY-MM-DD`` day into a ``[00:00, next-00:00)`` UTC RFC3339 pair.

    Raises :class:`CalendarError` on a malformed day so the caller maps it to a
    clean 4xx rather than a leaked 500.
    """
    try:
        start = datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=UTC)
    except ValueError as exc:
        raise CalendarError(
            f"invalid day {day!r}: expected YYYY-MM-DD"
        ) from exc
    end = start + timedelta(days=1)
    return (_rfc3339(start), _rfc3339(end))


def _rfc3339(moment: datetime) -> str:
    """Format a UTC ``datetime`` as ``YYYY-MM-DDTHH:MM:SSZ`` (Calendar's timeMin/Max)."""
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


async def whats_my_day(
    client: GoogleCalendarClient, *, day: str
) -> list[dict[str, Any]]:
    """List a single day's events — the "what's my day" helper.

    Expands ``day`` (``YYYY-MM-DD``) into a ``[00:00, next-00:00)`` UTC window and
    defers to :meth:`GoogleCalendarClient.list_events`. Raises
    :class:`CalendarError` on a malformed day or any list failure.
    """
    time_min, time_max = _day_window(day)
    return await client.list_events(time_min=time_min, time_max=time_max)


def _safe_body(response: httpx.Response) -> str:
    """A short, token-free snippet of a response body for an error message.

    Truncated so a large/HTML error page does not bloat the raised error. Carries
    no FRIDAY secret (the bearer token is only ever in the request header).
    """
    text = response.text or ""
    snippet = text.strip().replace("\n", " ")
    return snippet[:200]
