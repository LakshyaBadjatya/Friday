"""Integration tests for the orchestrator's self-critique hook (Tier 2).

End-to-end through :meth:`Orchestrator.handle` on offline fakes — the synthesis
LLM and the critic LLM are *separate* providers so each call is unambiguous. The
pinned behaviours mirror the plan:

* **Flag ON + a failing verdict with a revision** -> the turn returns the revised
  text (one bounded pass).
* **Flag OFF** -> the critic's LLM is NEVER called (asserted via a recording spy
  whose ``calls`` stays at 0), and the original reply is returned verbatim.
* **Flag ON but the critic LLM errors** -> NON-FATAL: the original response is
  unchanged.
* **Flag ON + a clean reply the verdict passes** -> unchanged, and the outcome is
  recorded on the scratchpad.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import friday.config as config
import friday.core.orchestrator as orch_mod
from friday.core.critic import SelfCritic
from friday.core.orchestrator import Orchestrator
from friday.core.state import GraphState, Mode
from friday.memory.short_term import ShortTermMemory
from friday.providers.llm import (
    FakeLLM,
    LLMResponse,
    Message,
    ToolSpec,
    Usage,
)
from friday.tools.registry import ToolRegistry
from friday.tools.web_search import WebSearchTool

PERSONA_PATH = (
    Path(__file__).resolve().parents[2] / "src" / "friday" / "persona" / "friday.md"
)


def _resp(text: str) -> LLMResponse:
    return LLMResponse(text=text, tool_calls=[], usage=Usage())


def _verdict_resp(
    *, ok: bool, issues: list[str] | None = None, revised: str | None = None
) -> LLMResponse:
    payload = {"ok": ok, "issues": issues or [], "revised": revised}
    return LLMResponse(text=json.dumps(payload), tool_calls=[], usage=Usage())


class _RecordingLLM:
    """A spy LLM that counts ``complete`` calls; calling it is a hard failure.

    The flag-off test injects this as the critic's provider and asserts it is
    never reached — both via ``calls == 0`` and by the call itself raising.
    """

    def __init__(self) -> None:
        self.calls = 0

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
    ) -> LLMResponse:
        self.calls += 1
        raise AssertionError("critic LLM must not be called when the flag is off")


class _RaisingLLM:
    """An LLM stub whose ``complete`` always raises (non-fatal-path check)."""

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
    ) -> LLMResponse:
        raise RuntimeError("critic llm exploded")


def _set_flag(monkeypatch: pytest.MonkeyPatch, *, enabled: bool) -> None:
    """Force ``enable_self_critique`` for both the config + orchestrator modules."""
    settings = config.Settings(_env_file=None, enable_self_critique=enabled)
    monkeypatch.setattr(config, "get_settings", lambda: settings)
    monkeypatch.setattr(orch_mod, "get_settings", lambda: settings)


def _orchestrator(synth_llm: FakeLLM, critic: SelfCritic | None) -> Orchestrator:
    registry = ToolRegistry()
    registry.register(WebSearchTool())
    return Orchestrator(
        llm=synth_llm,
        registry=registry,
        memory=ShortTermMemory(),
        persona_path=PERSONA_PATH,
        critic=critic,
    )


async def test_flag_on_failing_verdict_returns_revised_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_flag(monkeypatch, enabled=True)
    # Synthesis produces a weak draft; the critic flags it and offers a revision.
    synth_llm = FakeLLM(responses=[_resp("Let me think about that, Boss...")])
    critic_llm = FakeLLM(
        responses=[
            _verdict_resp(
                ok=False,
                issues=["did not answer the question"],
                revised="Four, Boss. Basic arithmetic holds.",
            )
        ]
    )
    orch = _orchestrator(synth_llm, SelfCritic(critic_llm))
    state = GraphState(session_id="c1", user_input="what's 2+2")

    out = await orch.handle(state)

    assert out.mode is Mode.CONVERSATION
    assert out.response == "Four, Boss. Basic arithmetic holds."
    recorded = out.scratchpad.get("self_critique")
    assert recorded is not None
    assert recorded["ok"] is False
    assert recorded["revised"] is True


async def test_flag_off_never_calls_critic_llm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_flag(monkeypatch, enabled=False)
    synth_llm = FakeLLM(responses=[_resp("Four, Boss. Basic arithmetic holds.")])
    spy = _RecordingLLM()
    orch = _orchestrator(synth_llm, SelfCritic(spy))  # type: ignore[arg-type]
    state = GraphState(session_id="c2", user_input="what's 2+2")

    out = await orch.handle(state)

    # The critic LLM was never reached, and the original reply is returned as-is.
    assert spy.calls == 0
    assert out.response == "Four, Boss. Basic arithmetic holds."
    assert "self_critique" not in out.scratchpad


async def test_flag_on_critic_llm_error_keeps_original(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_flag(monkeypatch, enabled=True)
    synth_llm = FakeLLM(responses=[_resp("Four, Boss. Basic arithmetic holds.")])
    orch = _orchestrator(synth_llm, SelfCritic(_RaisingLLM()))  # type: ignore[arg-type]
    state = GraphState(session_id="c3", user_input="what's 2+2")

    out = await orch.handle(state)  # must not raise

    # Non-fatal: the critic error leaves the synthesized reply untouched, and the
    # recorded outcome is a passing (no-op) critique with no revision.
    assert out.response == "Four, Boss. Basic arithmetic holds."
    recorded = out.scratchpad.get("self_critique")
    assert recorded == {"ok": True, "issues": [], "revised": False}


async def test_flag_on_clean_draft_passes_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_flag(monkeypatch, enabled=True)
    clean = "Four, Boss. Basic arithmetic holds."
    synth_llm = FakeLLM(responses=[_resp(clean)])
    critic_llm = FakeLLM(responses=[_verdict_resp(ok=True)])
    orch = _orchestrator(synth_llm, SelfCritic(critic_llm))
    state = GraphState(session_id="c4", user_input="what's 2+2")

    out = await orch.handle(state)

    assert out.response == clean
    recorded = out.scratchpad.get("self_critique")
    assert recorded == {"ok": True, "issues": [], "revised": False}


async def test_no_critic_wired_is_inert_even_with_flag_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Flag on but no critic injected: the turn behaves exactly as before.
    _set_flag(monkeypatch, enabled=True)
    synth_llm = FakeLLM(responses=[_resp("Four, Boss.")])
    orch = _orchestrator(synth_llm, None)
    state = GraphState(session_id="c5", user_input="what's 2+2")

    out = await orch.handle(state)

    assert out.response == "Four, Boss."
    assert "self_critique" not in out.scratchpad


async def test_clarify_turn_is_not_critiqued(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A deterministic CLARIFY question is never sent to the critic (no LLM call).
    _set_flag(monkeypatch, enabled=True)
    synth_llm = FakeLLM(responses=[])  # clarify is deterministic; no synth call
    spy = _RecordingLLM()
    orch = _orchestrator(synth_llm, SelfCritic(spy))  # type: ignore[arg-type]
    state = GraphState(session_id="c6", user_input="the blue one over there")

    out = await orch.handle(state)

    assert out.mode is Mode.CLARIFY
    assert spy.calls == 0
    assert "self_critique" not in out.scratchpad
