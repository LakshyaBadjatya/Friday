"""Keyless RSS / Atom feed reader.

:class:`InfofeedTool` fetches a single feed URL over :mod:`httpx` and parses it
into a flat list of ``{title, link}`` items. It handles both the RSS 2.0 shape
(``<rss><channel><item>``) and the Atom shape (``<feed><entry>``), so most public
feeds work without configuration. Parsing uses the standard-library
:mod:`xml.etree.ElementTree`, so the tool adds no third-party dependency.

The tool is read-only (``side_effecting=False``, ``idempotent=True``) and never
fabricates: on any failure it returns the error payload only, never invented
items. Its reliability contract mirrors the other keyless readers in the
codebase (``web_search`` / ``agent_reach``):

* one bounded retry on a transient network error, then a retriable
  ``ToolResult(ok=False, error=ToolError(code="feed_failed"))``;
* a non-OK HTTP status is a failure, ``retriable`` reflecting the 5xx/429 range;
* a body that cannot be parsed as XML yields a non-retriable
  ``feed_parse_failed``.

**XML safety.** A fetched feed is untrusted input, so the body is parsed with
external-entity and DTD processing disabled to block XXE and billion-laughs
(entity-expansion) attacks. When the optional :mod:`defusedxml` package is
installed it is used (lazy-imported, no hard dependency); otherwise the parse
falls back to a hardened stdlib :class:`xml.etree.ElementTree.XMLParser` whose
underlying expat parser rejects any ``<!DOCTYPE ...>`` declaration — both XXE and
billion-laughs require a DTD/entity definitions, so refusing DOCTYPE neutralizes
them. A feed carrying a DOCTYPE is therefore surfaced as ``feed_parse_failed``
rather than parsed.
"""

from __future__ import annotations

import logging
from typing import Any
from xml.etree import ElementTree as ET

import httpx
from pydantic import BaseModel, Field

from friday.tools.base import ToolError, ToolResult

logger = logging.getLogger("friday.tools.infofeed")

#: Default per-request timeout (seconds) for the feed fetch.
DEFAULT_TIMEOUT = 30.0

# A browser-like UA reduces the chance the upstream serves a challenge page; it
# carries no secrets and is safe to hardcode (mirrors web_search / agent_reach).
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

# Atom uses a namespace; tags arrive as ``{http://www.w3.org/2005/Atom}entry``.
_ATOM_NS = "http://www.w3.org/2005/Atom"


class FeedArgs(BaseModel):
    """Arguments for :class:`InfofeedTool`."""

    url: str = Field(min_length=1)


class FeedItem(BaseModel):
    """One parsed feed entry: its title and canonical link (either may be empty)."""

    title: str
    link: str


def _local(tag: str) -> str:
    """Strip an XML namespace, returning the bare local tag name."""
    return tag.rsplit("}", 1)[-1]


def _text(element: ET.Element | None) -> str:
    """Return stripped text of ``element`` (empty string when absent)."""
    if element is None or element.text is None:
        return ""
    return element.text.strip()


def _atom_link(entry: ET.Element) -> str:
    """Pick the best ``href`` from an Atom entry's ``<link>`` elements.

    Prefers ``rel="alternate"`` (or a link with no ``rel``, which Atom defines as
    ``alternate``); falls back to the first link's ``href``.
    """
    fallback = ""
    for child in entry:
        if _local(child.tag) != "link":
            continue
        href = child.get("href", "")
        if not href:
            continue
        rel = child.get("rel")
        if rel in (None, "alternate"):
            return href
        if not fallback:
            fallback = href
    return fallback


def _hardened_fromstring(body: bytes) -> ET.Element:
    """Parse ``body`` with a hand-built expat parser that refuses any DOCTYPE.

    Both XXE (external entities) and the billion-laughs entity-expansion attack
    require a ``<!DOCTYPE ...>`` to define entities, so refusing DOCTYPE outright
    neutralizes both without any third-party dependency. The expat parser drives
    a :class:`~xml.etree.ElementTree.TreeBuilder`; ``StartDoctypeDeclHandler`` is
    set to raise so a DOCTYPE-bearing document fails as a parse error.
    """
    from xml.parsers import expat

    builder = ET.TreeBuilder()

    def _forbid_doctype(*_args: object, **_kwargs: object) -> None:
        raise ET.ParseError("DOCTYPE declarations are not permitted")

    parser = expat.ParserCreate()
    parser.StartDoctypeDeclHandler = _forbid_doctype
    parser.StartElementHandler = lambda tag, attrs: builder.start(tag, attrs)
    parser.EndElementHandler = builder.end
    parser.CharacterDataHandler = builder.data
    try:
        parser.Parse(body, True)
    except expat.ExpatError as exc:
        raise ET.ParseError(str(exc)) from exc
    return builder.close()


def _safe_fromstring(body: bytes) -> ET.Element:
    """Parse ``body`` into an element tree with XXE/billion-laughs defused.

    Prefers the optional :mod:`defusedxml` package (lazy-imported so it is not a
    hard dependency). When it is absent, falls back to :func:`_hardened_fromstring`,
    a stdlib expat parser that rejects any DOCTYPE declaration — closing the door
    on external entities and the billion-laughs entity-expansion attack, both of
    which require a DTD.

    Raises :class:`xml.etree.ElementTree.ParseError` when ``body`` is not
    well-formed XML, or when it declares a DOCTYPE (treated as a parse failure).
    """
    try:
        from defusedxml.ElementTree import (  # type: ignore[import-untyped]
            fromstring as _defused_fromstring,
        )
    except ImportError:
        return _hardened_fromstring(body)

    result = _defused_fromstring(body)
    # defusedxml returns an ElementTree.Element; narrow for the type checker.
    assert isinstance(result, ET.Element)
    return result


def _parse_feed(body: bytes) -> list[FeedItem]:
    """Parse RSS or Atom ``body`` bytes into a list of :class:`FeedItem`.

    Raises :class:`xml.etree.ElementTree.ParseError` when ``body`` is not XML
    (including when it declares a DOCTYPE; see :func:`_safe_fromstring`).
    """
    root = _safe_fromstring(body)
    items: list[FeedItem] = []

    # RSS 2.0: <rss><channel><item><title/><link/></item></channel></rss>.
    for item in root.iter():
        if _local(item.tag) != "item":
            continue
        title = ""
        link = ""
        for child in item:
            local = _local(child.tag)
            if local == "title":
                title = _text(child)
            elif local == "link":
                link = _text(child)
        items.append(FeedItem(title=title, link=link))

    if items:
        return items

    # Atom: <feed><entry><title/><link href=.../></entry></feed>.
    for entry in root.iter(f"{{{_ATOM_NS}}}entry"):
        title = _text(entry.find(f"{{{_ATOM_NS}}}title"))
        items.append(FeedItem(title=title, link=_atom_link(entry)))

    # Fall back to namespace-agnostic ``entry`` scan (feeds with a default
    # namespace declared oddly) if the namespaced lookup found nothing.
    if not items:
        for entry in root.iter():
            if _local(entry.tag) != "entry":
                continue
            title = ""
            for child in entry:
                if _local(child.tag) == "title":
                    title = _text(child)
                    break
            items.append(FeedItem(title=title, link=_atom_link(entry)))

    return items


class InfofeedTool:
    """Fetch and parse an RSS/Atom feed into a list of ``{title, link}`` items.

    Args:
        timeout: Per-request wall-clock budget (seconds) for the feed fetch.
    """

    name = "infofeed"
    description = (
        "Fetch an RSS or Atom feed by URL and return its items (title + link)."
    )
    args_model = FeedArgs
    required_permission = "web"
    idempotent = True
    side_effecting = False

    def __init__(self, *, timeout: float = DEFAULT_TIMEOUT) -> None:
        self._timeout = timeout

    async def __call__(self, args: Any) -> ToolResult:
        """Fetch ``url`` (one bounded retry) and parse it into feed items."""
        # ``args`` arrives validated from the registry; coerce defensively.
        if not isinstance(args, FeedArgs):
            args = FeedArgs.model_validate(args)

        last_exc: Exception | None = None
        # Two attempts total: the initial call plus one bounded retry.
        for attempt in range(2):
            try:
                response = await self._fetch(args.url)
            except _RETRIABLE_EXCEPTIONS as exc:
                last_exc = exc
                logger.warning(
                    "infofeed network error (attempt %d/2): %s", attempt + 1, exc
                )
                continue

            if response.status_code != httpx.codes.OK:
                retriable = response.status_code in _TRANSIENT_STATUSES
                logger.warning(
                    "infofeed non-OK status %d for %r",
                    response.status_code,
                    args.url,
                )
                return ToolResult(
                    ok=False,
                    data={},
                    error=ToolError(
                        code="feed_failed",
                        message=f"feed returned HTTP {response.status_code}",
                        retriable=retriable,
                    ),
                )

            try:
                items = _parse_feed(response.content)
            except ET.ParseError as exc:
                logger.warning("infofeed parse error for %r: %s", args.url, exc)
                return ToolResult(
                    ok=False,
                    data={},
                    error=ToolError(
                        code="feed_parse_failed",
                        message=f"could not parse feed XML: {exc}",
                        retriable=False,
                    ),
                )

            return ToolResult(
                ok=True,
                data={
                    "url": args.url,
                    "items": [item.model_dump() for item in items],
                    "count": len(items),
                },
                error=None,
            )

        # Both attempts hit a retriable network error.
        return ToolResult(
            ok=False,
            data={},
            error=ToolError(
                code="feed_failed",
                message=f"feed request failed after retry: {last_exc}",
                retriable=True,
            ),
        )

    async def _fetch(self, url: str) -> httpx.Response:
        """Issue the feed GET and return the raw response (may raise)."""
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            return await client.get(url, headers={"User-Agent": _USER_AGENT})
