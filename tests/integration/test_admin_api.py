"""Integration tests for the admin API (Phase 5, Stage 2A).

Drives the real FastAPI app through ``TestClient`` with a scripted
:class:`FakeLLM` injected (zero network) and asserts that the observability layer
wired in ``app.py`` surfaces through the ``/admin`` routes:

* **Every request emits a trace.** After one ``POST /chat`` the
  ``GET /admin/traces`` view returns a trace for that turn carrying the
  route -> dispatch -> synth spans — the load-bearing "every request is traced"
  guarantee (build-spec §11).
* ``GET /admin/metrics`` shows ``requests >= 1`` after a turn.
* ``GET /admin/audit`` shows a tool-call row when a tool actually ran (the
  research path through the mocked web search).
* ``GET /admin/flags`` returns the current feature flags and ``POST /admin/flags``
  toggles one in a runtime override holder, with the change reflected back.
* ``GET /admin/state`` returns live numbers (active sessions, their modes,
  short-term sizes, memory stats).

Everything is offline: the LLM is a :class:`FakeLLM` and the DuckDuckGo endpoint
is ``respx``-mocked, so no network is touched.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import respx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from friday.app import create_app
from friday.core.orchestrator import Orchestrator
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
</body></html>
"""


def _conversation_orchestrator(app: FastAPI) -> None:
    """Wire a FakeLLM conversation orchestrator that shares app.state observability.

    Reuses the tracer/audit/metrics + registry the lifespan placed on
    ``app.state`` so the admin views read the same stores the orchestrator emits
    into — exactly the production wiring, only with a scripted LLM.
    """
    registry = app.state.registry
    llm = FakeLLM(
        responses=[LLMResponse(text="Four, Boss.", tool_calls=[], usage=Usage())]
    )
    app.state.orchestrator = Orchestrator(
        llm=llm,
        registry=registry,
        memory=app.state.short_term,
        persona_path=PERSONA_PATH,
        tracer=app.state.tracer,
        metrics=app.state.metrics,
    )


def _research_orchestrator(app: FastAPI) -> None:
    """Wire a FakeLLM research orchestrator over the mocked web-search tool.

    A research turn dispatches ``web_search`` through the registry, so the
    tool-call audit lands a row — letting the audit assertion prove a real tool
    execution was recorded.
    """
    registry = ToolRegistry(
        audit=app.state.audit,
        metrics=app.state.metrics,
        correlation_id="admin-test",
    )
    registry.register(WebSearchTool(base_url=SEARCH_BASE))
    app.state.registry = registry
    llm = FakeLLM(
        responses=[
            LLMResponse(
                text="Qdrant's the strongest pick, Boss.",
                tool_calls=[],
                usage=Usage(),
            )
        ]
    )
    app.state.orchestrator = Orchestrator(
        llm=llm,
        registry=registry,
        memory=app.state.short_term,
        persona_path=PERSONA_PATH,
        tracer=app.state.tracer,
        metrics=app.state.metrics,
    )


def test_admin_traces_shows_a_trace_for_every_chat_turn() -> None:
    # After a single /chat turn, /admin/traces must return a trace whose spans
    # cover route -> dispatch -> synth: the "every request emits a trace" gate.
    app = create_app()
    with TestClient(app) as client:
        _conversation_orchestrator(app)
        chat = client.post("/chat", json={"session_id": "admin-1", "text": "what's 2+2"})
        assert chat.status_code == 200

        traces = client.get("/admin/traces")
        assert traces.status_code == 200
        body = traces.json()
        rows = body["traces"]
        assert len(rows) >= 1
        latest = rows[-1]
        assert "correlation_id" in latest
        span_names = [s["name"] for s in latest["spans"]]
        assert span_names == ["route", "dispatch", "synth"]
        # Each span carries a name and timings.
        for span in latest["spans"]:
            assert span["start"] is not None
            assert span["end"] is not None


def test_admin_metrics_reports_requests_after_a_turn() -> None:
    app = create_app()
    with TestClient(app) as client:
        _conversation_orchestrator(app)
        client.post("/chat", json={"session_id": "admin-2", "text": "hi"})

        metrics = client.get("/admin/metrics")
        assert metrics.status_code == 200
        snap = metrics.json()
        assert snap["requests"] >= 1
        assert "by_mode" in snap
        assert "tool_calls" in snap
        assert "errors" in snap


@respx.mock
def test_admin_audit_shows_tool_call_when_one_ran() -> None:
    respx.post(SEARCH_BASE).mock(return_value=httpx.Response(200, text=SAMPLE_HTML))
    app = create_app()
    with TestClient(app) as client:
        _research_orchestrator(app)
        chat = client.post(
            "/chat",
            json={
                "session_id": "admin-3",
                "text": "research the best vector database and compare options",
            },
        )
        assert chat.status_code == 200
        assert chat.json()["mode"] == "RESEARCH"

        audit = client.get("/admin/audit")
        assert audit.status_code == 200
        body = audit.json()
        tool_calls = body["tool_calls"]
        assert any(row["tool"] == "web_search" for row in tool_calls)


def test_admin_flags_returns_and_toggles_a_flag() -> None:
    app = create_app()
    with TestClient(app) as client:
        _conversation_orchestrator(app)
        flags = client.get("/admin/flags")
        assert flags.status_code == 200
        current = flags.json()["flags"]
        assert "enable_voice" in current
        # The settings default is voice OFF.
        assert current["enable_voice"] is False

        toggled = client.post(
            "/admin/flags", json={"name": "enable_voice", "value": True}
        )
        assert toggled.status_code == 200
        effective = toggled.json()["flags"]
        assert effective["enable_voice"] is True

        # The override persists across a subsequent GET.
        again = client.get("/admin/flags")
        assert again.json()["flags"]["enable_voice"] is True


def test_admin_flags_rejects_unknown_flag() -> None:
    app = create_app()
    with TestClient(app) as client:
        _conversation_orchestrator(app)
        resp = client.post(
            "/admin/flags", json={"name": "not_a_real_flag", "value": True}
        )
        assert resp.status_code == 400


def test_admin_state_returns_live_numbers() -> None:
    app = create_app()
    with TestClient(app) as client:
        _conversation_orchestrator(app)
        client.post("/chat", json={"session_id": "admin-4", "text": "hello there"})

        state = client.get("/admin/state")
        assert state.status_code == 200
        body = state.json()
        sessions = body["sessions"]
        ids = {s["session_id"] for s in sessions}
        assert "admin-4" in ids
        row = next(s for s in sessions if s["session_id"] == "admin-4")
        # The turn recorded a user + assistant message, so short-term size >= 2.
        assert row["short_term_size"] >= 2
        assert row["mode"] == "CONVERSATION"
        assert "memory" in body
        assert "facts" in body["memory"]


def test_admin_routes_are_registered_on_a_fresh_app() -> None:
    # The admin router is wired in app.py (not only after a manual re-inject).
    app = create_app()
    with TestClient(app) as client:
        resp = client.get("/admin/metrics")
        assert resp.status_code == 200
