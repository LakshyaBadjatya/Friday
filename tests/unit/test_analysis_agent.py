"""Unit tests for :class:`friday.agents.analysis.AnalysisAgent` (Phase 2, Stage 3).

The analysis agent searches the web through the injected :class:`ToolRegistry`
(``allowed_tools={"web_search"}``) and then synthesizes an answer that:

* carries an explicit ``[confidence: low|medium|high]`` tag, and
* NEVER emits a bare numeric probability/percentage that is not backed by a
  cited ``source_id`` drawn from the retrieved evidence.

All LLM calls run on :class:`~friday.providers.llm.FakeLLM` (zero network) and
every web call is ``respx``-mocked, so the suite is fully offline and
deterministic. The load-bearing test is *adversarial*: a "give me the exact
probability" prompt must not yield a fabricated, unsourced percentage.
"""

from __future__ import annotations

import re

import httpx
import respx

from friday.agents.analysis import AnalysisAgent
from friday.agents.base import Agent, AgentResult
from friday.core.state import GraphState, Mode
from friday.providers.llm import FakeLLM, LLMResponse, Usage
from friday.tools.registry import ToolRegistry
from friday.tools.web_search import WebSearchTool

# A bare percentage like "73%" or "12.5 %" or "a 40 percent chance" with no
# adjacent bracketed source citation. Used to assert no *unsourced* figure
# leaks. ``[S1]``-style citations are stripped before this is applied.
_BARE_PERCENT_RE = re.compile(r"\d+(?:\.\d+)?\s*(?:%|percent)", re.IGNORECASE)
_CONFIDENCE_TAG_RE = re.compile(r"\[confidence:\s*(low|medium|high)\]", re.IGNORECASE)
# Citation markers we treat as a legitimate source reference, e.g. ``[S1]`` or
# ``(source: S2)``. Stripped from the output before the bare-percent scan so a
# *sourced* figure is not flagged.
_CITATION_RE = re.compile(r"\[s\d+\]|\(source:\s*s\d+\)", re.IGNORECASE)


def _resp(text: str) -> LLMResponse:
    return LLMResponse(text=text, tool_calls=[], usage=Usage())


def _make_agent(
    llm: FakeLLM, *, search_base: str | None = None
) -> AnalysisAgent:
    registry = ToolRegistry()
    if search_base is not None:
        registry.register(WebSearchTool(base_url=search_base))
    else:
        registry.register(WebSearchTool())
    return AnalysisAgent(registry=registry, llm=llm)


_SAMPLE_HTML = (
    '<html><body><div class="result">'
    '<a class="result__a" href="https://example.com/abc">Acme Corp Outlook</a>'
    '<a class="result__snippet" href="https://example.com/abc">'
    "Analysts note steady revenue growth and a strong balance sheet."
    "</a></div>"
    '<div class="result">'
    '<a class="result__a" href="https://example.com/def">Sector Report</a>'
    '<a class="result__snippet" href="https://example.com/def">'
    "Macro headwinds remain but the sector is broadly stable."
    "</a></div></body></html>"
)


def _strip_citations(text: str) -> str:
    return _CITATION_RE.sub("", text)


# --------------------------------------------------------------------------- #
# Contract
# --------------------------------------------------------------------------- #
def test_agent_satisfies_protocol_and_metadata() -> None:
    agent = _make_agent(FakeLLM(responses=[]))
    assert isinstance(agent, Agent)
    assert agent.name == "analysis"
    assert agent.allowed_tools == frozenset({"web_search"})


@respx.mock
async def test_run_searches_then_returns_agent_result_with_confidence_tag() -> None:
    search_base = "https://analysis-search.test/html/"
    route = respx.post(search_base).mock(
        return_value=httpx.Response(200, text=_SAMPLE_HTML)
    )
    # A well-formed, sourced synthesis with an explicit qualitative confidence.
    llm = FakeLLM(
        responses=[
            _resp(
                "Acme's fundamentals look solid [S1] and the wider sector is "
                "broadly stable [S2]. On balance the outlook leans positive. "
                "[confidence: medium]"
            )
        ]
    )
    agent = _make_agent(llm, search_base=search_base)
    state = GraphState(
        session_id="a1",
        mode=Mode.RESEARCH,
        user_input="analyze the outlook for Acme Corp stock",
    )

    result = await agent.run(state)

    assert isinstance(result, AgentResult)
    assert route.called, "the agent must actually hit web_search"
    # The web_search call is recorded for audit.
    assert any(tc.name == "web_search" for tc in result.tool_calls_made)
    # Explicit qualitative confidence tag is present.
    assert _CONFIDENCE_TAG_RE.search(result.output) is not None


@respx.mock
async def test_adversarial_exact_probability_yields_no_unsourced_percentage() -> None:
    """The crux: an "exact probability" demand must not fabricate a bare %%.

    Even if the underlying model is *scripted* to emit a tempting unsourced
    percentage, the agent must scrub/guard so the surfaced output contains no
    bare numeric probability that lacks a source citation — it states the
    evidence and a qualitative confidence instead.
    """
    search_base = "https://analysis-adv.test/html/"
    respx.post(search_base).mock(return_value=httpx.Response(200, text=_SAMPLE_HTML))
    # Adversarial: the model tries to hand back a fabricated, unsourced number.
    llm = FakeLLM(
        responses=[
            _resp(
                "There is a 73% probability the stock rises and a 27 percent "
                "chance it falls. It will definitely go up."
            )
        ]
    )
    agent = _make_agent(llm, search_base=search_base)
    state = GraphState(
        session_id="a2",
        mode=Mode.RESEARCH,
        user_input="what's the exact probability this stock rises? give me a number.",
    )

    result = await agent.run(state)

    # No bare, unsourced percentage survives in the surfaced output.
    scrubbed = _strip_citations(result.output)
    assert _BARE_PERCENT_RE.search(scrubbed) is None, (
        f"an unsourced percentage leaked into the output: {result.output!r}"
    )
    # Instead, a qualitative confidence tag is present.
    assert _CONFIDENCE_TAG_RE.search(result.output) is not None
    # And it does not overclaim certainty about a forecast.
    assert "definitely" not in result.output.lower()


@respx.mock
async def test_sourced_percentage_is_permitted() -> None:
    """A percentage that *cites a source* is legitimate and must pass through."""
    search_base = "https://analysis-sourced.test/html/"
    respx.post(search_base).mock(return_value=httpx.Response(200, text=_SAMPLE_HTML))
    llm = FakeLLM(
        responses=[
            _resp(
                "Revenue grew 12% last year [S1], a healthy clip. "
                "[confidence: high]"
            )
        ]
    )
    agent = _make_agent(llm, search_base=search_base)
    state = GraphState(
        session_id="a3",
        mode=Mode.RESEARCH,
        user_input="how fast is Acme growing?",
    )

    result = await agent.run(state)

    # The sourced "12% [S1]" must survive (it is evidence-backed, not fabricated).
    assert "12%" in result.output
    assert _CONFIDENCE_TAG_RE.search(result.output) is not None


@respx.mock
async def test_search_failure_is_reported_honestly_not_fabricated() -> None:
    """A failed search yields an honest, low-confidence answer — no invention."""
    search_base = "https://analysis-fail.test/html/"
    respx.post(search_base).mock(return_value=httpx.Response(503))
    # If the agent (wrongly) calls the LLM it would emit this; the test asserts
    # the surfaced output is honest about the lack of evidence regardless.
    llm = FakeLLM(
        responses=[
            _resp("The stock will rise 50%. [confidence: high]")
        ]
    )
    agent = _make_agent(llm, search_base=search_base)
    state = GraphState(
        session_id="a4",
        mode=Mode.RESEARCH,
        user_input="analyze whether the stock rises",
    )

    result = await agent.run(state)

    scrubbed = _strip_citations(result.output)
    assert _BARE_PERCENT_RE.search(scrubbed) is None
    # With no retrieved evidence, confidence must be low and it must say so.
    tag = _CONFIDENCE_TAG_RE.search(result.output)
    assert tag is not None and tag.group(1).lower() == "low"
    assert result.confidence <= 0.5
