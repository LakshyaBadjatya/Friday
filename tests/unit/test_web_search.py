"""Unit tests for :class:`friday.tools.web_search.WebSearchTool`.

All HTTP is mocked with ``respx`` — no live network. Covers parsing of the
DuckDuckGo HTML response into structured results, the bounded single retry on a
retriable network error, and the failure payload returned when both attempts
fail. The tool must never fabricate results.
"""

from __future__ import annotations

import httpx
import respx

from friday.tools.base import ToolResult
from friday.tools.web_search import WebSearchArgs, WebSearchTool, _parse_html

DDG_ENDPOINT = "https://html.duckduckgo.com/html/"

# A trimmed but structurally faithful DuckDuckGo HTML result block: two results,
# each with a ``result__a`` title/link and a ``result__snippet`` description.
SAMPLE_HTML = """
<html><body>
<div class="result results_links results_links_deep web-result">
  <div class="result__body">
    <h2 class="result__title">
      <a rel="nofollow" class="result__a" href="https://example.com/vector-db">
        Best Vector Database 2026
      </a>
    </h2>
    <a class="result__snippet" href="https://example.com/vector-db">
      A thorough comparison of vector databases for production RAG.
    </a>
  </div>
</div>
<div class="result results_links results_links_deep web-result">
  <div class="result__body">
    <h2 class="result__title">
      <a rel="nofollow" class="result__a" href="https://qdrant.tech/">
        Qdrant - Vector Search Engine
      </a>
    </h2>
    <a class="result__snippet" href="https://qdrant.tech/">
      Open-source vector similarity search engine with extended filtering.
    </a>
  </div>
</div>
</body></html>
"""


def test_web_search_tool_attrs() -> None:
    tool = WebSearchTool()
    assert tool.name == "web_search"
    assert tool.side_effecting is False
    assert tool.idempotent is True
    assert tool.args_model is WebSearchArgs


def test_web_search_args_defaults() -> None:
    args = WebSearchArgs(query="hello")
    assert args.query == "hello"
    assert args.max_results == 5


@respx.mock
async def test_web_search_parses_results() -> None:
    respx.post(DDG_ENDPOINT).mock(
        return_value=httpx.Response(200, text=SAMPLE_HTML)
    )
    tool = WebSearchTool()
    result = await tool(WebSearchArgs(query="best vector db", max_results=5))

    assert isinstance(result, ToolResult)
    assert result.ok is True
    assert result.error is None
    results = result.data["results"]
    assert len(results) == 2

    first = results[0]
    assert first["title"] == "Best Vector Database 2026"
    assert first["url"] == "https://example.com/vector-db"
    assert "comparison of vector databases" in first["snippet"]

    second = results[1]
    assert second["url"] == "https://qdrant.tech/"


def test_parse_html_keeps_snippetless_results() -> None:
    # A DDG result that has a title/link but no result__snippet must still be
    # returned (with snippet=""), including when it is the LAST result — the bug
    # silently dropped such results.
    html = """
    <div class="result"><a class="result__a" href="https://a.example/">Alpha</a></div>
    <div class="result">
      <a class="result__a" href="https://b.example/">Beta</a>
      <a class="result__snippet" href="https://b.example/">Beta has a snippet.</a>
    </div>
    <div class="result"><a class="result__a" href="https://c.example/">Gamma</a></div>
    """
    results = _parse_html(html, 10)
    by_url = {r["url"]: r for r in results}
    assert set(by_url) == {"https://a.example/", "https://b.example/", "https://c.example/"}
    assert by_url["https://a.example/"]["title"] == "Alpha"
    assert by_url["https://a.example/"]["snippet"] == ""  # snippet-less, not dropped
    assert by_url["https://c.example/"]["snippet"] == ""  # trailing snippet-less, not dropped
    assert "snippet" in by_url["https://b.example/"]["snippet"]


@respx.mock
async def test_web_search_respects_max_results() -> None:
    respx.post(DDG_ENDPOINT).mock(
        return_value=httpx.Response(200, text=SAMPLE_HTML)
    )
    tool = WebSearchTool()
    result = await tool(WebSearchArgs(query="anything", max_results=1))
    assert result.ok is True
    assert len(result.data["results"]) == 1


@respx.mock
async def test_web_search_retries_once_then_fails() -> None:
    route = respx.post(DDG_ENDPOINT).mock(
        side_effect=httpx.ConnectError("no route to host")
    )
    tool = WebSearchTool()
    result = await tool(WebSearchArgs(query="will fail"))

    # Exactly two attempts: the initial call plus one bounded retry.
    assert route.call_count == 2
    assert result.ok is False
    assert result.error is not None
    assert result.error.code == "search_failed"
    # No fabricated data on failure.
    assert result.data.get("results", []) == []


@respx.mock
async def test_web_search_succeeds_on_retry() -> None:
    route = respx.post(DDG_ENDPOINT).mock(
        side_effect=[
            httpx.ConnectError("transient blip"),
            httpx.Response(200, text=SAMPLE_HTML),
        ]
    )
    tool = WebSearchTool()
    result = await tool(WebSearchArgs(query="recoverable"))

    assert route.call_count == 2
    assert result.ok is True
    assert len(result.data["results"]) == 2


@respx.mock
async def test_web_search_http_error_returns_failure() -> None:
    respx.post(DDG_ENDPOINT).mock(
        return_value=httpx.Response(503, text="service unavailable")
    )
    tool = WebSearchTool()
    result = await tool(WebSearchArgs(query="server down"))
    assert result.ok is False
    assert result.error is not None
    assert result.error.code == "search_failed"


@respx.mock
async def test_web_search_empty_results_does_not_fabricate() -> None:
    respx.post(DDG_ENDPOINT).mock(
        return_value=httpx.Response(200, text="<html><body>no results</body></html>")
    )
    tool = WebSearchTool()
    result = await tool(WebSearchArgs(query="nothing here"))
    assert result.ok is True
    assert result.data["results"] == []


@respx.mock
async def test_web_search_202_is_retried_and_retriable() -> None:
    # The backend currently answers some queries with a "still processing" 202;
    # it must be retried once and surfaced as a RETRIABLE failure (not a hard one).
    route = respx.post(DDG_ENDPOINT).mock(
        return_value=httpx.Response(202, text="processing")
    )
    tool = WebSearchTool()
    result = await tool(WebSearchArgs(query="slow backend"))

    # Exactly two attempts: the initial 202 plus one bounded retry.
    assert route.call_count == 2
    assert result.ok is False
    assert result.error is not None
    assert result.error.code == "search_failed"
    assert result.error.retriable is True
    # Never fabricate results on failure.
    assert result.data.get("results", []) == []
