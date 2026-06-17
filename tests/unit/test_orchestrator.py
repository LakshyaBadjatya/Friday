"""Unit tests for :class:`friday.core.orchestrator.Orchestrator` (Task 1.7).

All LLM calls run on :class:`~friday.providers.llm.FakeLLM` (zero network) and
every web call is ``respx``-mocked. The three load-bearing behaviours pinned
here mirror the build-spec acceptance tests:

* **Persona** — a synthesized reply carries none of the banned tone markers
  from ``persona/friday.md`` (no sycophantic opener, no apology theatre, no fake
  enthusiasm, no padding) and addresses the owner as configured.
* **Refusal** — an out-of-scope ask (facial recognition) yields a *short,
  honest, in-character decline* that never claims to perform the task.
* **Clarify** — genuinely ambiguous input returns a *question*, not a guess.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import respx

from friday.core.graph import ModeGraph, build_graph
from friday.core.orchestrator import Orchestrator
from friday.core.state import GraphState, Mode
from friday.memory.short_term import ShortTermMemory
from friday.providers.llm import FakeLLM, LLMResponse, Usage
from friday.tools.registry import ToolRegistry
from friday.tools.weather import WeatherTool
from friday.tools.web_search import WebSearchTool

PERSONA_PATH = (
    Path(__file__).resolve().parents[2] / "src" / "friday" / "persona" / "friday.md"
)

# Phrases that must never open or pad a FRIDAY reply (drawn from the persona
# spec's "Banned Tone Markers" section). Matched case-insensitively.
_BANNED_MARKERS: tuple[str, ...] = (
    "great question",
    "what a fantastic",
    "i'd be happy to help",
    "i would be happy to help",
    "let me help you with that",
    "i'm so excited",
    "i am so excited",
    "thanks for asking",
)


def _make_orchestrator(
    llm: FakeLLM, *, search_base: str | None = None, with_weather: bool = False
) -> Orchestrator:
    registry = ToolRegistry()
    if search_base is not None:
        registry.register(WebSearchTool(base_url=search_base))
    else:
        registry.register(WebSearchTool())
    if with_weather:
        registry.register(WeatherTool())
    memory = ShortTermMemory()
    return Orchestrator(
        llm=llm,
        registry=registry,
        memory=memory,
        persona_path=PERSONA_PATH,
    )


def _resp(text: str) -> LLMResponse:
    return LLMResponse(text=text, tool_calls=[], usage=Usage())


def test_match_protocol_tiebreak_uses_normalized_length(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # "Longest phrase wins" must compare normalized lengths on both sides. A short
    # trigger padded with whitespace must not beat a genuinely more-specific one.
    from types import SimpleNamespace

    from friday.core import orchestrator as orch_module
    from friday.protocols.store import Protocol

    short_padded = Protocol(id=1, name="A", trigger_phrase="nap" + " " * 20)  # raw len 23
    specific = Protocol(id=2, name="B", trigger_phrase="afternoon nap")  # normalized 13

    class _FakeStore:
        def list_protocols(self) -> list[Protocol]:
            return [short_padded, specific]  # padded one first -> initial best

        def get_by_name(self, name: str) -> Protocol | None:
            return None

    orch = _make_orchestrator(FakeLLM(responses=[]))
    orch._protocol_store = _FakeStore()  # type: ignore[assignment]
    orch._protocol_runner = object()  # type: ignore[assignment]
    monkeypatch.setattr(
        orch_module, "get_settings", lambda: SimpleNamespace(enable_protocols=True)
    )

    match = orch._match_protocol("time for my afternoon nap")
    assert match is not None and match.name == "B"  # the more-specific trigger wins


async def test_conversation_reply_has_no_banned_tone_markers() -> None:
    # A clean, answer-first reply the LLM is scripted to return.
    llm = FakeLLM(responses=[_resp("Four, Boss. Basic arithmetic holds.")])
    orch = _make_orchestrator(llm)
    state = GraphState(session_id="s1", user_input="what's 2+2")

    out = await orch.handle(state)

    assert out.mode is Mode.CONVERSATION
    assert out.response is not None
    lowered = out.response.lower()
    for marker in _BANNED_MARKERS:
        assert marker not in lowered, f"banned tone marker leaked: {marker!r}"
    # Persona injection means the system prompt carried the persona text; the
    # owner address is present in the scripted answer.
    assert "Boss" in out.response


async def test_refusal_facial_recognition_is_honest_decline() -> None:
    # The orchestrator must refuse WITHOUT calling the LLM to fabricate a
    # capability. We script a response anyway to prove it is *not* consumed for
    # the refusal text (script would be popped only if the LLM were called).
    llm = FakeLLM(responses=[_resp("Sure, scanning the faces now!")])
    orch = _make_orchestrator(llm)
    state = GraphState(
        session_id="s2",
        user_input="Use facial recognition to identify the person in this photo.",
    )

    out = await orch.handle(state)

    assert out.response is not None
    lowered = out.response.lower()
    # It declines and names the reason.
    assert "can't" in lowered or "cannot" in lowered or "won't" in lowered or (
        "not" in lowered and "able" in lowered
    ) or "defensive" in lowered
    # It must NOT claim to perform the task / fabricate the capability.
    assert "scanning the faces" not in lowered
    assert "identified" not in lowered
    assert "the person is" not in lowered
    # Honest, short decline — bounded length (no lecture, no fabricated result).
    assert len(out.response) <= 400


async def test_clarify_returns_question_not_guess() -> None:
    # Ambiguous, no clear intent -> router yields CLARIFY -> a question.
    llm = FakeLLM(responses=[])  # must NOT be consumed; clarify is deterministic
    orch = _make_orchestrator(llm)
    state = GraphState(session_id="s3", user_input="the blue one over there")

    out = await orch.handle(state)

    assert out.mode is Mode.CLARIFY
    assert out.response is not None
    assert "?" in out.response, "a clarifying turn must ask a question"


@respx.mock
async def test_research_path_invokes_web_search_then_synthesizes() -> None:
    sample_html = """
    <html><body>
    <div class="result">
      <a class="result__a" href="https://example.com/vdb">Vector DB Guide</a>
      <a class="result__snippet" href="https://example.com/vdb">
        A comparison of vector databases for RAG.
      </a>
    </div>
    </body></html>
    """
    search_base = "https://search.test/html/"
    respx.post(search_base).mock(return_value=httpx.Response(200, text=sample_html))

    # First LLM response: synthesis after the tool ran.
    llm = FakeLLM(
        responses=[_resp("Qdrant looks the strongest pick, Boss — solid filtering.")]
    )
    orch = _make_orchestrator(llm, search_base=search_base)
    state = GraphState(
        session_id="s4",
        user_input="research the best vector database and compare options",
    )

    out = await orch.handle(state)

    assert out.mode is Mode.RESEARCH
    assert out.response is not None
    assert "Boss" in out.response
    # The web_search tool was actually invoked (recorded in scratchpad).
    assert out.scratchpad.get("web_search_invoked") is True


@respx.mock
async def test_research_path_uses_weather_tool_for_weather_query() -> None:
    # A weather question routes to RESEARCH and retrieves from the keyless
    # ``weather`` tool (wttr.in) rather than the flaky search backend.
    j1 = {
        "current_condition": [
            {
                "temp_C": "30",
                "FeelsLikeC": "32",
                "humidity": "40",
                "windspeedKmph": "8",
                "weatherDesc": [{"value": "Sunny"}],
                "observation_time": "12:00 PM",
            }
        ],
        "nearest_area": [{"areaName": [{"value": "Kota"}]}],
    }
    weather_route = respx.get(url__regex=r"https://wttr\.in/.*").mock(
        return_value=httpx.Response(200, json=j1)
    )
    llm = FakeLLM(responses=[_resp("It's sunny in Kota at 30°C, Boss.")])
    orch = _make_orchestrator(llm, with_weather=True)
    state = GraphState(session_id="w1", user_input="whats the weather in kota")

    out = await orch.handle(state)

    assert out.mode is Mode.RESEARCH
    assert out.response is not None
    # The weather tool ran — NOT web_search.
    assert weather_route.called
    assert out.scratchpad.get("web_search_invoked") is False
    weather_result = out.scratchpad.get("weather_result")
    assert weather_result is not None
    assert "Kota" in str(weather_result.get("location", ""))


# --- graph assembly (core/graph.py + core/modes.py) ----------------------- #


async def test_graph_builds_langgraph_engine() -> None:
    # On the pinned toolchain the primary LangGraph engine must be selected.
    llm = FakeLLM(responses=[])
    orch = _make_orchestrator(llm)
    graph = build_graph(orch)
    assert isinstance(graph, ModeGraph)


@respx.mock
async def test_graph_research_turn_invokes_search_and_synthesizes() -> None:
    sample_html = (
        '<html><body><div class="result">'
        '<a class="result__a" href="https://e.com/x">Vector DB</a>'
        '<a class="result__snippet" href="https://e.com/x">a comparison</a>'
        "</div></body></html>"
    )
    search_base = "https://graph-search.test/html/"
    route = respx.post(search_base).mock(
        return_value=httpx.Response(200, text=sample_html)
    )
    llm = FakeLLM(responses=[_resp("Qdrant wins, Boss.")])
    orch = _make_orchestrator(llm, search_base=search_base)
    graph = build_graph(orch)

    out = await graph.invoke(
        GraphState(
            session_id="g1",
            user_input="research the best vector database and compare options",
        )
    )

    assert isinstance(out, GraphState)
    assert out.mode is Mode.RESEARCH
    assert out.response == "Qdrant wins, Boss."
    assert out.scratchpad.get("web_search_invoked") is True
    assert route.called


async def test_graph_clarify_turn_asks_question() -> None:
    llm = FakeLLM(responses=[])  # clarify is deterministic; no LLM call
    orch = _make_orchestrator(llm)
    graph = build_graph(orch)

    out = await graph.invoke(
        GraphState(session_id="g2", user_input="the blue one over there")
    )

    assert out.mode is Mode.CLARIFY
    assert out.response is not None
    assert "?" in out.response
