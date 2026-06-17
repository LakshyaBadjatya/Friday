# © Lakshya Badjatya — Author
"""Integration tests for GET /admin/usage (the cost-dashboard data).

Offline: a scripted :class:`FakeLLM` (zero network) carrying a non-zero
:class:`Usage`, wired into an orchestrator that shares the ``app.state`` usage
ledger the lifespan built — exactly the production wiring. Proves the endpoint is
empty-but-valid before any turn, and reflects real token spend after one.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from friday.app import create_app
from friday.config import get_settings
from friday.core.orchestrator import Orchestrator
from friday.providers.llm import FakeLLM, LLMResponse, Usage

PERSONA_PATH = (
    Path(__file__).resolve().parents[2] / "src" / "friday" / "persona" / "friday.md"
)


def _app(monkeypatch: pytest.MonkeyPatch) -> FastAPI:
    """Build an app over pinned, pristine offline settings (no cache leakage)."""
    monkeypatch.setenv("FRIDAY_LLM_PROVIDER", "fake")
    monkeypatch.setenv("FRIDAY_MEMORY_DB_PATH", ":memory:")
    get_settings.cache_clear()
    return create_app()


def _wire_usage_orchestrator(app: FastAPI) -> None:
    """Wire a FakeLLM orchestrator sharing app.state's usage ledger.

    The scripted completion carries a non-zero token Usage so the ledger has
    real numbers to surface after the turn.
    """
    llm = FakeLLM(
        responses=[
            LLMResponse(
                text="Four, Boss.",
                tool_calls=[],
                usage=Usage(prompt_tokens=12, completion_tokens=8),
            )
        ]
    )
    app.state.orchestrator = Orchestrator(
        llm=llm,
        registry=app.state.registry,
        memory=app.state.short_term,
        persona_path=PERSONA_PATH,
        tracer=app.state.tracer,
        metrics=app.state.metrics,
        usage_ledger=app.state.usage,
    )


def test_usage_is_empty_but_valid_before_any_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    with TestClient(_app(monkeypatch)) as client:
        resp = client.get("/admin/usage")
        assert resp.status_code == 200
        snap = resp.json()
        assert snap == {
            "completions": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "tokens": 0,
            "usd": 0.0,
            "by_model": {},
        }
    get_settings.cache_clear()


def test_usage_reflects_spend_after_a_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    app = _app(monkeypatch)
    with TestClient(app) as client:
        _wire_usage_orchestrator(app)
        chat = client.post("/chat", json={"session_id": "usage-1", "text": "what's 2+2"})
        assert chat.status_code == 200

        snap = client.get("/admin/usage").json()
        assert snap["completions"] >= 1
        # At least our scripted completion's 12 + 8 tokens are accounted for.
        assert snap["tokens"] >= 20
        assert snap["prompt_tokens"] >= 12
        assert snap["completion_tokens"] >= 8
        assert snap["usd"] == 0.0  # free models
        # The completion lands under a per-model row.
        assert snap["by_model"]
        assert sum(m["tokens"] for m in snap["by_model"].values()) == snap["tokens"]
    get_settings.cache_clear()
