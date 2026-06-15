"""Gateway-hardening ASGI middleware: bearer auth + fixed-window rate limiting.

Two pure-ASGI middlewares wrap the FRIDAY app (Phase 6, build-spec §11/§13):

* :class:`AuthMiddleware` — when ``settings.require_auth`` is set, every request
  except ``GET /health`` must carry an ``Authorization: Bearer <key>`` header
  whose ``<key>`` is in ``settings.api_keys``; a missing or malformed header or
  an unknown key short-circuits with ``401 {"detail": "unauthorized"}``. With
  ``require_auth`` off (the default), it is a transparent pass-through.
* :class:`RateLimitMiddleware` — a fixed-window per-client limiter
  (``rate_limit_requests`` per ``rate_limit_window_seconds``). The client key is
  the bearer token when present, else the peer IP; over the limit returns
  ``429 {"detail": "rate limit exceeded"}`` with a ``Retry-After`` header.
  ``GET /health`` is exempt. The clock is injectable (read from
  ``app.state.rate_limit_clock`` at request time, default :func:`time.monotonic`)
  so tests advance "now" deterministically without touching the wall clock.

Both are written as raw ASGI (not ``BaseHTTPMiddleware``) so the rejection
responses never run the heavy request-body machinery and the clock can be
resolved per request off ``app.state``.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable

from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp, Receive, Scope, Send

from friday.config import Settings

#: Paths exempt from both auth and rate limiting (liveness must always answer).
_HEALTH_PATH = "/health"

#: A clock returns a monotonically increasing seconds value.
Clock = Callable[[], float]


def _bearer_key(request: Request) -> str | None:
    """Extract the bearer token from the ``Authorization`` header, or ``None``.

    Returns the token only for a well-formed ``Bearer <token>`` header (scheme
    matched case-insensitively, exactly one space-separated token); anything
    else (missing header, wrong scheme, empty token) yields ``None``.
    """
    header = request.headers.get("authorization")
    if header is None:
        return None
    parts = header.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    token = parts[1].strip()
    return token or None


class AuthMiddleware:
    """Require a valid bearer key on every non-``/health`` route when enabled."""

    def __init__(self, app: ASGIApp, *, settings: Settings) -> None:
        self._app = app
        self._settings = settings
        self._keys = frozenset(settings.api_keys)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not self._settings.require_auth:
            await self._app(scope, receive, send)
            return

        request = Request(scope, receive=receive)
        if request.url.path == _HEALTH_PATH:
            await self._app(scope, receive, send)
            return

        key = _bearer_key(request)
        if key is None or key not in self._keys:
            response: Response = JSONResponse(
                status_code=401, content={"detail": "unauthorized"}
            )
            await response(scope, receive, send)
            return

        await self._app(scope, receive, send)


class _Window:
    """A single client's fixed-window state: the window start and its count."""

    __slots__ = ("start", "count")

    def __init__(self, start: float) -> None:
        self.start = start
        self.count = 0


class RateLimitMiddleware:
    """Fixed-window per-client rate limiter with an injectable clock.

    The limit is ``settings.rate_limit_requests`` requests per
    ``settings.rate_limit_window_seconds`` per client. The clock used to time
    windows is resolved per request: ``app.state.rate_limit_clock`` if present
    (tests inject a fake), else the ``default_clock`` passed at construction
    (``time.monotonic``). When ``rate_limit_enabled`` is false, or for
    ``GET /health``, the middleware is a transparent pass-through.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        settings: Settings,
        default_clock: Clock = time.monotonic,
    ) -> None:
        self._app = app
        self._settings = settings
        self._default_clock = default_clock
        self._windows: dict[str, _Window] = {}

    def _clock(self, scope: Scope) -> Clock:
        """Resolve the active clock, preferring an injected one on ``app.state``."""
        app = scope.get("app")
        injected = getattr(getattr(app, "state", None), "rate_limit_clock", None)
        if callable(injected):
            return injected  # type: ignore[no-any-return]
        return self._default_clock

    def _client_key(self, request: Request) -> str:
        """Identify the caller: bearer key if present, else peer IP."""
        key = _bearer_key(request)
        if key is not None:
            return f"key:{key}"
        host = request.client.host if request.client is not None else "unknown"
        return f"ip:{host}"

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not self._settings.rate_limit_enabled:
            await self._app(scope, receive, send)
            return

        request = Request(scope, receive=receive)
        if request.url.path == _HEALTH_PATH:
            await self._app(scope, receive, send)
            return

        now = self._clock(scope)()
        window_len = self._settings.rate_limit_window_seconds
        client = self._client_key(request)

        window = self._windows.get(client)
        if window is None or (now - window.start) >= window_len:
            window = _Window(start=now)
            self._windows[client] = window

        if window.count >= self._settings.rate_limit_requests:
            retry_after = max(0, int(window.start + window_len - now))
            response: Response = JSONResponse(
                status_code=429,
                content={"detail": "rate limit exceeded"},
                headers={"Retry-After": str(retry_after)},
            )
            await response(scope, receive, send)
            return

        window.count += 1
        await self._app(scope, receive, send)


# Re-export the awaitable app type for callers that annotate the wrapped app.
ASGIAppCallable = Callable[[Scope, Receive, Send], Awaitable[None]]
