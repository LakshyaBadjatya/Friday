"""Unit tests for :class:`friday.tools.browser_tool.BrowserTool`.

Fully offline: HTTP is mocked with ``respx`` (no live network), mirroring the
agent_reach read_url tests. The tool is read-only and must never fabricate page
content on failure.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from friday.tools.base import ToolResult
from friday.tools.browser_tool import BrowserArgs, BrowserTool

JINA_BASE = "https://r.jina.ai/"
PAGE_URL = "https://example.com/article"
JINA_URL = f"{JINA_BASE}{PAGE_URL}"

SAMPLE_TEXT = "Example Article\n\nReadable body extracted by Jina Reader.\n"


# -- attributes / args --------------------------------------------------- #


def test_browser_tool_attrs() -> None:
    tool = BrowserTool()
    assert tool.name == "browser"
    assert tool.side_effecting is False
    assert tool.idempotent is True
    assert tool.required_permission == "web"
    assert tool.args_model is BrowserArgs


def test_browser_args_rejects_empty_url() -> None:
    with pytest.raises(ValueError):
        BrowserArgs(url="")


# -- happy path ---------------------------------------------------------- #


@respx.mock
async def test_returns_readable_text() -> None:
    respx.get(JINA_URL).mock(return_value=httpx.Response(200, text=SAMPLE_TEXT))
    tool = BrowserTool(jina_base=JINA_BASE)
    result = await tool(BrowserArgs(url=PAGE_URL))

    assert isinstance(result, ToolResult)
    assert result.ok is True
    assert result.error is None
    assert result.data["content"] == SAMPLE_TEXT
    assert result.data["source"] == "jina-reader"
    assert result.data["url"] == PAGE_URL


@respx.mock
async def test_targets_jina_reader_with_url_appended() -> None:
    route = respx.get(JINA_URL).mock(
        return_value=httpx.Response(200, text=SAMPLE_TEXT)
    )
    tool = BrowserTool(jina_base=JINA_BASE)
    await tool(BrowserArgs(url=PAGE_URL))
    assert route.called
    assert str(route.calls.last.request.url) == JINA_URL


# -- failure modes (never fabricate) ------------------------------------- #


@respx.mock
async def test_retries_once_then_fails() -> None:
    route = respx.get(JINA_URL).mock(
        side_effect=httpx.ConnectError("no route to host")
    )
    tool = BrowserTool(jina_base=JINA_BASE)
    result = await tool(BrowserArgs(url=PAGE_URL))

    assert route.call_count == 2
    assert result.ok is False
    assert result.error is not None
    assert result.error.code == "fetch_failed"
    assert result.error.retriable is True
    assert "content" not in result.data


@respx.mock
async def test_succeeds_on_retry() -> None:
    route = respx.get(JINA_URL).mock(
        side_effect=[
            httpx.ConnectError("transient blip"),
            httpx.Response(200, text=SAMPLE_TEXT),
        ]
    )
    tool = BrowserTool(jina_base=JINA_BASE)
    result = await tool(BrowserArgs(url=PAGE_URL))

    assert route.call_count == 2
    assert result.ok is True
    assert result.data["content"] == SAMPLE_TEXT


@respx.mock
async def test_transient_http_error_is_retriable_failure() -> None:
    respx.get(JINA_URL).mock(
        return_value=httpx.Response(503, text="service unavailable")
    )
    tool = BrowserTool(jina_base=JINA_BASE)
    result = await tool(BrowserArgs(url=PAGE_URL))

    assert result.ok is False
    assert result.error is not None
    assert result.error.code == "fetch_failed"
    assert result.error.retriable is True
    assert "content" not in result.data


@respx.mock
async def test_client_http_error_is_non_retriable_failure() -> None:
    respx.get(JINA_URL).mock(return_value=httpx.Response(404, text="not found"))
    tool = BrowserTool(jina_base=JINA_BASE)
    result = await tool(BrowserArgs(url=PAGE_URL))

    assert result.ok is False
    assert result.error is not None
    assert result.error.code == "fetch_failed"
    assert result.error.retriable is False
