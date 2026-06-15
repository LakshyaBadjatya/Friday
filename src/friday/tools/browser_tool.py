"""Keyless page reader: fetch a page's readable text via the Jina Reader.

:class:`BrowserTool` is a thin, read-only "open this page and give me the text"
tool. It mirrors the ``read_url`` channel of
:class:`~friday.tools.agent_reach.AgentReachTool`: it issues a single
``GET https://r.jina.ai/<url>`` over :mod:`httpx` and returns the clean readable
text Jina produces. It is keyless (no API token), so it works out of the box.

The tool is read-only (``side_effecting=False``, ``idempotent=True``) and NEVER
fabricates: on any failure it returns the error payload only, never an invented
page. Its reliability contract matches the other keyless readers:

* one bounded retry on a transient network error, then a retriable
  ``ToolResult(ok=False, error=ToolError(code="fetch_failed"))``;
* a non-OK HTTP status is a failure, ``retriable`` reflecting the 5xx/429 range.

Where :class:`AgentReachTool` is the Research/Knowledge agents' richer reach (it
also adds CLI transcription), :class:`BrowserTool` is the minimal "browse to a
URL" verb other callers can use directly. The Jina base and timeout are injected
(dependency injection); nothing here imports application config.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx
from pydantic import BaseModel, Field

from friday.tools.base import ToolError, ToolResult

logger = logging.getLogger("friday.tools.browser_tool")

#: Default keyless Jina Reader base; ``GET {base}{url}`` returns readable text.
DEFAULT_JINA_BASE = "https://r.jina.ai/"
#: Default per-request timeout (seconds) for the page fetch.
DEFAULT_TIMEOUT = 60.0

# A browser-like UA reduces the chance the upstream serves a challenge page; it
# carries no secrets and is safe to hardcode (mirrors agent_reach / web_search).
_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)

# httpx exceptions we treat as transient and therefore worth exactly one retry.
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
# HTTP statuses we treat as transient (worth surfacing as retriable=True).
_TRANSIENT_STATUSES = frozenset({429, 500, 502, 503, 504})


class BrowserArgs(BaseModel):
    """Arguments for :class:`BrowserTool`."""

    url: str = Field(min_length=1)


class BrowserTool:
    """Fetch a page's readable text (keyless) via the Jina Reader.

    Args:
        jina_base: Base of the keyless Jina Reader endpoint; the tool issues
            ``GET {jina_base}{url}`` and returns the readable text back.
        timeout: Per-request wall-clock budget (seconds) for the fetch.
    """

    name = "browser"
    description = (
        "Open a web page and return its readable text (keyless, via Jina Reader)."
    )
    args_model = BrowserArgs
    required_permission = "web"
    idempotent = True
    side_effecting = False

    def __init__(
        self,
        *,
        jina_base: str = DEFAULT_JINA_BASE,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self._jina_base = jina_base
        self._timeout = timeout

    async def __call__(self, args: Any) -> ToolResult:
        """Fetch ``url`` as readable text with one bounded network retry."""
        # ``args`` arrives validated from the registry; coerce defensively.
        if not isinstance(args, BrowserArgs):
            args = BrowserArgs.model_validate(args)

        last_exc: Exception | None = None
        # Two attempts total: the initial call plus one bounded retry.
        for attempt in range(2):
            try:
                response = await self._fetch(args.url)
            except _RETRIABLE_EXCEPTIONS as exc:
                last_exc = exc
                logger.warning(
                    "browser fetch network error (attempt %d/2): %s",
                    attempt + 1,
                    exc,
                )
                continue

            if response.status_code != httpx.codes.OK:
                retriable = response.status_code in _TRANSIENT_STATUSES
                logger.warning(
                    "browser fetch non-OK status %d for %r",
                    response.status_code,
                    args.url,
                )
                return ToolResult(
                    ok=False,
                    data={},
                    error=ToolError(
                        code="fetch_failed",
                        message=f"jina reader returned HTTP {response.status_code}",
                        retriable=retriable,
                    ),
                )

            return ToolResult(
                ok=True,
                data={
                    "content": response.text,
                    "source": "jina-reader",
                    "url": args.url,
                },
                error=None,
            )

        # Both attempts hit a retriable network error.
        return ToolResult(
            ok=False,
            data={},
            error=ToolError(
                code="fetch_failed",
                message=f"jina reader request failed after retry: {last_exc}",
                retriable=True,
            ),
        )

    async def _fetch(self, url: str) -> httpx.Response:
        """Issue the Jina Reader GET and return the raw response (may raise)."""
        target = f"{self._jina_base}{url}"
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            return await client.get(target, headers={"User-Agent": _USER_AGENT})
