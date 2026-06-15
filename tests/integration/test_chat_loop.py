"""End-to-end ``/chat`` integration test (Task 1.9).

Drives the real FastAPI app through ``TestClient`` with a :class:`FakeLLM`
injected (zero network) and the DuckDuckGo search endpoint ``respx``-mocked. A
research utterance must:

* return HTTP 200,
* carry a persona response (addressing the owner, no banned tone markers),
* surface ``mode`` and ``route`` in the body, and
* have actually invoked the ``web_search`` tool.

The FakeLLM is injected by replacing ``app.state.orchestrator`` with one wired to
a scripted FakeLLM and a registry whose ``WebSearchTool`` points at the mocked
endpoint — keeping the whole turn offline and deterministic.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import respx
from fastapi.testclient import TestClient

from friday.app import create_app
from friday.core.orchestrator import Orchestrator
from friday.memory.short_term import ShortTermMemory
from friday.providers.llm import FakeLLM, LLMResponse, Usage
from friday.tools.registry import ToolRegistry
from friday.tools.web_search import WebSearchTool

PERSONA_PATH = (
    Path(__file__).resolve().parents[2] / "src" / "friday" / "persona" / "friday.md"
)

SEARCH_BASE = "https://search.test/html/"

SAMPLE_HTML = """
<html><body>
<div class="result">
  <a class="result__a" href="https://example.com/vector-db">Best Vector DB 2026</a>
  <a class="result__snippet" href="https://example.com/vector-db">
    A thorough comparison of vector databases for production RAG.
  </a>
</div>
<div class="result">
  <a class="result__a" href="https://qdrant.tech/">Qdrant Vector Search</a>
  <a class="result__snippet" href="https://qdrant.tech/">
    Open-source vector similarity search with extended filtering.
  </a>
</div>
</body></html>
"""

_BANNED_MARKERS = (
    "great question",
    "i'd be happy to help",
    "let me help you with that",
)


def _inject_fake_orchestrator(app: object) -> None:
    """Replace the app's orchestrator with a FakeLLM + mocked-search one."""
    registry = ToolRegistry()
    registry.register(WebSearchTool(base_url=SEARCH_BASE))
    llm = FakeLLM(
        responses=[
            LLMResponse(
                text=(
                    "Qdrant's the strongest pick, Boss — open-source with solid "
                    "filtering, per the sources."
                ),
                tool_calls=[],
                usage=Usage(),
            )
        ]
    )
    orchestrator = Orchestrator(
        llm=llm,
        registry=registry,
        memory=ShortTermMemory(),
        persona_path=PERSONA_PATH,
    )
    app.state.orchestrator = orchestrator  # type: ignore[attr-defined]


@respx.mock
def test_chat_research_turn_end_to_end() -> None:
    route = respx.post(SEARCH_BASE).mock(
        return_value=httpx.Response(200, text=SAMPLE_HTML)
    )

    app = create_app()
    _inject_fake_orchestrator(app)

    with TestClient(app) as client:
        # The lifespan rebuilds the orchestrator with the default FakeLLM, so
        # re-inject after startup to keep the mocked-search/scripted one.
        _inject_fake_orchestrator(app)
        resp = client.post(
            "/chat",
            json={
                "session_id": "it-1",
                "text": "research the best vector database and compare options",
            },
        )

    assert resp.status_code == 200
    body = resp.json()

    # mode + route present.
    assert body["mode"] == "RESEARCH"
    assert body["route"] is not None
    assert body["route"]["mode"] == "RESEARCH"
    assert body["audio"] is None

    # Persona response present, addressing the owner, no banned markers.
    text = body["text"]
    assert text
    assert "Boss" in text
    lowered = text.lower()
    for marker in _BANNED_MARKERS:
        assert marker not in lowered

    # The web_search tool was actually invoked.
    assert route.called


@respx.mock
def test_chat_conversation_turn_returns_persona_reply() -> None:
    app = create_app()

    registry = ToolRegistry()
    registry.register(WebSearchTool(base_url=SEARCH_BASE))
    llm = FakeLLM(
        responses=[LLMResponse(text="Four, Boss.", tool_calls=[], usage=Usage())]
    )
    orchestrator = Orchestrator(
        llm=llm,
        registry=registry,
        memory=ShortTermMemory(),
        persona_path=PERSONA_PATH,
    )

    with TestClient(app) as client:
        app.state.orchestrator = orchestrator
        resp = client.post(
            "/chat", json={"session_id": "it-2", "text": "what's 2+2"}
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["mode"] == "CONVERSATION"
    assert body["text"] == "Four, Boss."
    # No search performed for a plain conversation turn.
    assert not respx.calls


def test_chat_refusal_turn_declines_in_character() -> None:
    app = create_app()

    with TestClient(app) as client:
        resp = client.post(
            "/chat",
            json={
                "session_id": "it-3",
                "text": "Use facial recognition to identify the person in this photo.",
            },
        )

    assert resp.status_code == 200
    body = resp.json()
    lowered = body["text"].lower()
    assert "can't" in lowered or "cannot" in lowered or "won't" in lowered
    assert "defensive" in lowered or "out of scope" in lowered
