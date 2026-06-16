# © Lakshya Badjatya — Author
"""Keyless current-weather / forecast tool backed by the wttr.in service.

:class:`WeatherTool` fetches the current conditions for a city or place over
:mod:`httpx` and parses wttr.in's ``j1`` JSON shape into a compact, structured
payload. It is read-only (``side_effecting=False``, ``idempotent=True``) and
needs no API key — wttr.in is keyless.

The reliability contract mirrors the other keyless readers in the codebase
(``web_search`` / ``infofeed`` / ``agent_reach``):

* one bounded retry on a transient network error, then a retriable
  ``ToolResult(ok=False, error=ToolError(code="weather_failed"))``;
* a non-OK HTTP status is a failure, ``retriable`` reflecting whether the status
  is in the transient 5xx/429/202 range;
* a body that cannot be parsed as the expected ``j1`` JSON yields a non-retriable
  ``weather_parse_failed``.

The host is fixed (``wttr.in``), so there is no SSRF resolver to run — the
attacker-influencable location is simply URL-encoded into the path. On any
failure the tool never fabricates: the result carries the error payload only,
never invented weather keys.
"""

from __future__ import annotations

import logging
from typing import Any
from urllib.parse import quote

import httpx
from pydantic import BaseModel, Field, field_validator

from friday.tools.base import ToolError, ToolResult

logger = logging.getLogger("friday.tools.weather")

#: Base of the keyless wttr.in endpoint; the location is URL-encoded into the path.
_WTTR_BASE = "https://wttr.in/"

#: Default per-request timeout (seconds) for the weather fetch.
DEFAULT_TIMEOUT = 15.0

# A browser-like UA is REQUIRED by wttr.in to serve JSON rather than its plain
# ASCII art; it carries no secrets and is safe to hardcode (mirrors web_search /
# infofeed).
_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)

# httpx exceptions we treat as transient and therefore worth exactly one retry
# (the same set ``web_search`` retries on).
_RETRIABLE_EXCEPTIONS = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.ReadError,
    httpx.ReadTimeout,
    httpx.WriteError,
    httpx.WriteTimeout,
    httpx.PoolTimeout,
    httpx.RemoteProtocolError,
)
# HTTP statuses we treat as transient (worth surfacing as retriable=True). 202 is
# included: wttr.in occasionally returns 202 while warming a result, so a retry
# can succeed.
_TRANSIENT_STATUSES = frozenset({202, 429, 500, 502, 503, 504})


class WeatherArgs(BaseModel):
    """Arguments for :class:`WeatherTool`."""

    location: str = Field(min_length=1)

    @field_validator("location")
    @classmethod
    def _reject_control_chars(cls, value: str) -> str:
        """Reject control characters / newlines in the location.

        The location is interpolated into a URL path; a newline or other control
        character is never a legitimate place name and could be used to smuggle
        request structure, so it is rejected outright (the value is still
        URL-encoded before use as a second line of defence).
        """
        if any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in value):
            raise ValueError("location must not contain control characters")
        return value


def _as_int(value: object) -> int:
    """Coerce a wttr.in numeric field (which arrives as a string) to ``int``.

    Raises :class:`ValueError`/:class:`TypeError` on a missing or non-numeric
    value so the caller surfaces a ``weather_parse_failed`` rather than guessing.
    """
    return int(str(value).strip())


def _resolved_area(data: dict[str, Any], fallback: str) -> str:
    """Pull the resolved area name from a ``j1`` payload, else ``fallback``.

    wttr.in echoes the place it resolved the query to under
    ``nearest_area[0].areaName[0].value``; using it makes the summary name the
    actual matched place. Any gap in that nested shape falls back to the input.
    """
    try:
        area = data["nearest_area"][0]["areaName"][0]["value"]
    except (KeyError, IndexError, TypeError):
        return fallback
    area_text = str(area).strip()
    return area_text or fallback


def _parse_j1(data: dict[str, Any], location: str) -> dict[str, Any]:
    """Parse a wttr.in ``j1`` payload into the compact result payload.

    Raises :class:`KeyError`/:class:`IndexError`/:class:`TypeError`/
    :class:`ValueError` when the expected ``current_condition`` shape is missing
    or non-numeric; the caller maps that to ``weather_parse_failed``.
    """
    current = data["current_condition"][0]
    temp_c = _as_int(current["temp_C"])
    feels_like_c = _as_int(current["FeelsLikeC"])
    humidity_pct = _as_int(current["humidity"])
    wind_kph = _as_int(current["windspeedKmph"])
    description = str(current["weatherDesc"][0]["value"]).strip()
    area = _resolved_area(data, location)
    summary = (
        f"{area}: {description}, {temp_c}°C "
        f"(feels {feels_like_c}°C), humidity {humidity_pct}%"
    )
    return {
        "location": area,
        "temp_c": temp_c,
        "feels_like_c": feels_like_c,
        "description": description,
        "humidity_pct": humidity_pct,
        "wind_kph": wind_kph,
        "summary": summary,
    }


class WeatherTool:
    """Fetch the current weather for a city/place via the keyless wttr.in service.

    Args:
        timeout: Per-request wall-clock budget (seconds) for the weather fetch.
    """

    name = "weather"
    description = (
        "Get the CURRENT weather/forecast for a city or place. Use for any "
        "weather/temperature question."
    )
    args_model = WeatherArgs
    required_permission = "web"
    idempotent = True
    side_effecting = False

    def __init__(self, *, timeout: float = DEFAULT_TIMEOUT) -> None:
        self._timeout = timeout

    async def __call__(self, args: Any) -> ToolResult:
        """Fetch the weather (one bounded retry) and parse it into the payload."""
        # ``args`` arrives validated from the registry; coerce defensively so the
        # tool is also safe to call directly.
        if not isinstance(args, WeatherArgs):
            args = WeatherArgs.model_validate(args)

        last_exc: Exception | None = None
        # Two attempts total: the initial call plus one bounded retry.
        for attempt in range(2):
            try:
                response = await self._fetch(args.location)
            except _RETRIABLE_EXCEPTIONS as exc:
                last_exc = exc
                logger.warning(
                    "weather network error (attempt %d/2): %s", attempt + 1, exc
                )
                continue

            if response.status_code != httpx.codes.OK:
                retriable = response.status_code in _TRANSIENT_STATUSES
                logger.warning(
                    "weather non-OK status %d for %r",
                    response.status_code,
                    args.location,
                )
                return ToolResult(
                    ok=False,
                    data={},
                    error=ToolError(
                        code="weather_failed",
                        message=f"weather service returned HTTP {response.status_code}",
                        retriable=retriable,
                    ),
                )

            try:
                payload = _parse_j1(response.json(), args.location)
            except (
                ValueError,
                KeyError,
                IndexError,
                TypeError,
            ) as exc:
                logger.warning("weather parse error for %r: %s", args.location, exc)
                return ToolResult(
                    ok=False,
                    data={},
                    error=ToolError(
                        code="weather_parse_failed",
                        message=f"could not parse weather response: {exc}",
                        retriable=False,
                    ),
                )

            return ToolResult(ok=True, data=payload, error=None)

        # Both attempts hit a retriable network error.
        return ToolResult(
            ok=False,
            data={},
            error=ToolError(
                code="weather_failed",
                message=f"weather request failed after retry: {last_exc}",
                retriable=True,
            ),
        )

    async def _fetch(self, location: str) -> httpx.Response:
        """Issue the wttr.in ``j1`` GET for ``location`` (may raise).

        The fixed host (``wttr.in``) means no SSRF resolver is needed; the
        attacker-influencable location is URL-encoded into the path. A browser
        User-Agent is sent so wttr.in serves the ``j1`` JSON rather than ASCII art.
        """
        url = f"{_WTTR_BASE}{quote(location)}"
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            return await client.get(
                url,
                params={"format": "j1"},
                headers={"User-Agent": _USER_AGENT},
            )
