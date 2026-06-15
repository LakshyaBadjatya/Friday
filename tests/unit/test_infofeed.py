"""Unit tests for :class:`friday.tools.infofeed.InfofeedTool`.

Fully offline: HTTP is mocked with ``respx`` (no live network). The tool is
read-only and must never fabricate items on failure. XML parsing is hardened
against XXE / billion-laughs, so a feed declaring a DOCTYPE is refused.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from friday.tools.base import ToolResult
from friday.tools.infofeed import FeedArgs, InfofeedTool

FEED_URL = "https://example.com/feed.xml"

RSS_BODY = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Example Channel</title>
    <item>
      <title>First Post</title>
      <link>https://example.com/first</link>
    </item>
    <item>
      <title>Second Post</title>
      <link>https://example.com/second</link>
    </item>
  </channel>
</rss>
"""

ATOM_BODY = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>Example Atom Feed</title>
  <entry>
    <title>Atom Entry One</title>
    <link rel="alternate" href="https://example.com/atom-1"/>
    <link rel="edit" href="https://example.com/atom-1/edit"/>
  </entry>
  <entry>
    <title>Atom Entry Two</title>
    <link href="https://example.com/atom-2"/>
  </entry>
</feed>
"""

# A malicious feed declaring a DOCTYPE (the billion-laughs / XXE vector).
DOCTYPE_BODY = (
    '<?xml version="1.0"?>'
    '<!DOCTYPE rss [<!ENTITY lol "lol">]>'
    "<rss><channel><item><title>&lol;</title></item></channel></rss>"
)


# -- attributes / args --------------------------------------------------- #


def test_infofeed_tool_attrs() -> None:
    tool = InfofeedTool()
    assert tool.name == "infofeed"
    assert tool.side_effecting is False
    assert tool.idempotent is True
    assert tool.required_permission == "web"
    assert tool.args_model is FeedArgs


def test_feed_args_rejects_empty_url() -> None:
    with pytest.raises(ValueError):
        FeedArgs(url="")


# -- RSS parsing --------------------------------------------------------- #


@respx.mock
async def test_rss_feed_parsed_into_items() -> None:
    respx.get(FEED_URL).mock(
        return_value=httpx.Response(200, content=RSS_BODY.encode("utf-8"))
    )
    tool = InfofeedTool()
    result = await tool(FeedArgs(url=FEED_URL))

    assert isinstance(result, ToolResult)
    assert result.ok is True
    assert result.error is None
    assert result.data["count"] == 2
    items = result.data["items"]
    assert items[0] == {"title": "First Post", "link": "https://example.com/first"}
    assert items[1] == {
        "title": "Second Post",
        "link": "https://example.com/second",
    }


# -- Atom parsing -------------------------------------------------------- #


@respx.mock
async def test_atom_feed_parsed_into_items() -> None:
    respx.get(FEED_URL).mock(
        return_value=httpx.Response(200, content=ATOM_BODY.encode("utf-8"))
    )
    tool = InfofeedTool()
    result = await tool(FeedArgs(url=FEED_URL))

    assert result.ok is True
    items = result.data["items"]
    assert result.data["count"] == 2
    assert items[0]["title"] == "Atom Entry One"
    # The alternate link is preferred over the edit link.
    assert items[0]["link"] == "https://example.com/atom-1"
    assert items[1]["link"] == "https://example.com/atom-2"


# -- failure modes (never fabricate) ------------------------------------- #


@respx.mock
async def test_non_ok_status_returns_failure() -> None:
    respx.get(FEED_URL).mock(return_value=httpx.Response(503, text="down"))
    tool = InfofeedTool()
    result = await tool(FeedArgs(url=FEED_URL))
    assert result.ok is False
    assert result.error is not None
    assert result.error.code == "feed_failed"
    assert result.error.retriable is True
    assert "items" not in result.data


@respx.mock
async def test_404_is_non_retriable_failure() -> None:
    respx.get(FEED_URL).mock(return_value=httpx.Response(404, text="gone"))
    tool = InfofeedTool()
    result = await tool(FeedArgs(url=FEED_URL))
    assert result.ok is False
    assert result.error is not None
    assert result.error.code == "feed_failed"
    assert result.error.retriable is False


@respx.mock
async def test_retries_once_then_fails() -> None:
    route = respx.get(FEED_URL).mock(
        side_effect=httpx.ConnectError("no route")
    )
    tool = InfofeedTool()
    result = await tool(FeedArgs(url=FEED_URL))
    assert route.call_count == 2
    assert result.ok is False
    assert result.error is not None
    assert result.error.code == "feed_failed"
    assert result.error.retriable is True


@respx.mock
async def test_succeeds_on_retry() -> None:
    route = respx.get(FEED_URL).mock(
        side_effect=[
            httpx.ConnectError("blip"),
            httpx.Response(200, content=RSS_BODY.encode("utf-8")),
        ]
    )
    tool = InfofeedTool()
    result = await tool(FeedArgs(url=FEED_URL))
    assert route.call_count == 2
    assert result.ok is True
    assert result.data["count"] == 2


@respx.mock
async def test_non_xml_body_returns_parse_failure() -> None:
    respx.get(FEED_URL).mock(
        return_value=httpx.Response(200, content=b"this is not xml <<<")
    )
    tool = InfofeedTool()
    result = await tool(FeedArgs(url=FEED_URL))
    assert result.ok is False
    assert result.error is not None
    assert result.error.code == "feed_parse_failed"
    assert result.error.retriable is False
    assert "items" not in result.data


@respx.mock
async def test_doctype_feed_is_rejected_not_expanded() -> None:
    # XXE / billion-laughs guard: a feed declaring a DOCTYPE must be refused at
    # parse time rather than have its entities expanded.
    respx.get(FEED_URL).mock(
        return_value=httpx.Response(200, content=DOCTYPE_BODY.encode("utf-8"))
    )
    tool = InfofeedTool()
    result = await tool(FeedArgs(url=FEED_URL))
    assert result.ok is False
    assert result.error is not None
    assert result.error.code == "feed_parse_failed"
    assert "items" not in result.data
