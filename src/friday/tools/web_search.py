"""Keyless web search tool backed by the DuckDuckGo HTML endpoint.

``WebSearchTool`` performs a POST against DuckDuckGo's keyless HTML search
endpoint (or a configurable SearXNG base) and parses the returned markup into
structured ``{title, url, snippet}`` results. It is read-only
(``side_effecting=False``) and ``idempotent=True``.

Reliability contract:

* On a *retriable* network error (connect/read/timeout/transport) the request is
  retried exactly once. If the retry also fails, the tool returns
  ``ToolResult(ok=False, error=ToolError(code="search_failed", retriable=True))``.
* A non-2xx HTTP status is treated as a failure: ``code="search_failed"`` with
  ``retriable`` reflecting whether the status is in the transient 5xx/429 range.
* Results are *only* ever derived from the live response. On any failure (or an
  empty page) the tool returns an empty result list — it never fabricates data.

HTML parsing uses the standard library :class:`html.parser.HTMLParser` so the
tool adds no third-party parsing dependency.
"""

from __future__ import annotations

import logging
from html.parser import HTMLParser
from typing import Any

import httpx
from pydantic import BaseModel, Field

from friday.tools.base import ToolError, ToolResult

logger = logging.getLogger("friday.tools.web_search")

DEFAULT_DDG_URL = "https://html.duckduckgo.com/html/"
_REQUEST_TIMEOUT = 10.0
# A browser-like UA reduces the chance of the HTML endpoint serving a challenge
# page; it carries no secrets and is safe to hardcode.
_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)

# httpx exceptions we consider transient and therefore worth one retry.
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
# included because the search backend currently answers some queries with a
# "still processing" 202 that a retry can resolve — so it must surface as a
# retriable failure, not a hard one.
_TRANSIENT_STATUSES = frozenset({202, 429, 500, 502, 503, 504})


class WebSearchArgs(BaseModel):
    """Arguments for :class:`WebSearchTool`."""

    query: str
    max_results: int = Field(default=5, ge=1, le=25)


class _DDGResultParser(HTMLParser):
    """Extract ``{title, url, snippet}`` rows from DuckDuckGo HTML markup.

    DuckDuckGo's HTML endpoint marks each result title/link with
    ``class="result__a"`` and its description with ``class="result__snippet"``.
    We accumulate text between the relevant tag boundaries and pair titles with
    the following snippet.
    """

    def __init__(self) -> None:
        super().__init__()
        self.results: list[dict[str, str]] = []
        self._in_title = False
        self._in_snippet = False
        self._current_title: list[str] = []
        self._current_url: str | None = None
        self._current_snippet: list[str] = []

    @staticmethod
    def _class_of(attrs: list[tuple[str, str | None]]) -> str:
        for key, value in attrs:
            if key == "class" and value is not None:
                return value
        return ""

    @staticmethod
    def _href_of(attrs: list[tuple[str, str | None]]) -> str | None:
        for key, value in attrs:
            if key == "href":
                return value
        return None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        classes = self._class_of(attrs).split()
        if "result__a" in classes:
            self._in_title = True
            self._current_title = []
            self._current_url = self._href_of(attrs)
        elif "result__snippet" in classes:
            self._in_snippet = True
            self._current_snippet = []

    def handle_endtag(self, tag: str) -> None:
        if tag != "a":
            return
        if self._in_title:
            self._in_title = False
        elif self._in_snippet:
            self._in_snippet = False
            self._flush_result()

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self._current_title.append(data)
        elif self._in_snippet:
            self._current_snippet.append(data)

    def _flush_result(self) -> None:
        title = " ".join("".join(self._current_title).split()).strip()
        snippet = " ".join("".join(self._current_snippet).split()).strip()
        url = (self._current_url or "").strip()
        if title and url:
            self.results.append({"title": title, "url": url, "snippet": snippet})
        self._current_title = []
        self._current_url = None
        self._current_snippet = []


def _parse_html(html_text: str, max_results: int) -> list[dict[str, str]]:
    """Parse DuckDuckGo HTML into at most ``max_results`` result rows."""
    parser = _DDGResultParser()
    parser.feed(html_text)
    return parser.results[:max_results]


class WebSearchTool:
    """Keyless web search over the DuckDuckGo HTML endpoint."""

    name = "web_search"
    description = "Search the public web and return titles, URLs, and snippets."
    args_model = WebSearchArgs
    required_permission = "web_search"
    idempotent = True
    side_effecting = False

    def __init__(self, base_url: str = DEFAULT_DDG_URL) -> None:
        self._base_url = base_url

    async def _fetch(self, args: WebSearchArgs) -> httpx.Response:
        """Issue the search POST and return the raw response (may raise)."""
        async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT) as client:
            return await client.post(
                self._base_url,
                data={"q": args.query},
                headers={"User-Agent": _USER_AGENT},
            )

    async def __call__(self, args: Any) -> ToolResult:
        """Run the search with one bounded retry on transient network failure."""
        # ``args`` arrives validated from the registry, but coerce defensively so
        # the tool is also safe to call directly.
        if not isinstance(args, WebSearchArgs):
            args = WebSearchArgs.model_validate(args)

        last_exc: Exception | None = None
        last_status: int | None = None
        # Two attempts total: initial call + one bounded retry on a retriable
        # error or a transient HTTP status (e.g. a "still processing" 202).
        for attempt in range(2):
            try:
                response = await self._fetch(args)
            except _RETRIABLE_EXCEPTIONS as exc:
                last_exc = exc
                logger.warning(
                    "web_search network error (attempt %d/2): %s", attempt + 1, exc
                )
                continue

            if response.status_code != httpx.codes.OK:
                retriable = response.status_code in _TRANSIENT_STATUSES
                logger.warning(
                    "web_search non-OK status %d for query %r",
                    response.status_code,
                    args.query,
                )
                # A transient status (5xx / 429 / a still-processing 202) is worth
                # one bounded retry; a hard status (e.g. 404) is returned at once.
                if retriable and attempt == 0:
                    last_status = response.status_code
                    continue
                return ToolResult(
                    ok=False,
                    data={"results": []},
                    error=ToolError(
                        code="search_failed",
                        message=(
                            f"search backend returned HTTP {response.status_code}"
                        ),
                        retriable=retriable,
                    ),
                )

            results = _parse_html(response.text, args.max_results)
            return ToolResult(
                ok=True,
                data={"query": args.query, "results": results},
                error=None,
            )

        # Both attempts hit a retriable network error or transient status.
        message = (
            f"search backend returned HTTP {last_status} after retry"
            if last_status is not None
            else f"search request failed after retry: {last_exc}"
        )
        return ToolResult(
            ok=False,
            data={"results": []},
            error=ToolError(
                code="search_failed",
                message=message,
                retriable=True,
            ),
        )
