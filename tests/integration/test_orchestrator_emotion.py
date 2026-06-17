"""Phase-2: the orchestrator injects an emotion tone-hint when adaptation is on."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from friday.config import get_settings
from friday.core.orchestrator import Orchestrator
from friday.core.state import GraphState
from friday.memory.short_term import ShortTermMemory
from friday.providers.emotion import Emotion
from friday.providers.llm import LLMResponse, Message, Usage
from friday.tools.registry import ToolRegistry

PERSONA = (
    Path(__file__).resolve().parents[2] / "src" / "friday" / "persona" / "friday.md"
)


class _CapturingLLM:
    """Records the messages of the last completion; returns a fixed reply."""

    def __init__(self) -> None:
        self.last: list[Message] = []

    async def complete(self, messages, tools=None) -> LLMResponse:
        self.last = list(messages)
        return LLMResponse(text="Right here, Boss.", tool_calls=[], usage=Usage())


def _orch(llm: _CapturingLLM) -> Orchestrator:
    return Orchestrator(
        llm=llm, registry=ToolRegistry(), memory=ShortTermMemory(),
        persona_path=PERSONA,
    )


def _sad_state() -> GraphState:
    state = GraphState(session_id="s", user_input="what's the status")
    state.emotion = Emotion(
        valence=0.2, arousal=0.3, dominance=0.4, label="sad",
        intensity=0.7, confidence=0.6, ts=0.0,
    )
    return state


def _system_text(llm: _CapturingLLM) -> str:
    return next(m.content for m in llm.last if m.role == "system")


def test_emotion_hint_injected_when_adapt_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FRIDAY_EMOTION_ADAPT", "true")
    monkeypatch.setenv("FRIDAY_LLM_PROVIDER", "fake")
    get_settings.cache_clear()
    llm = _CapturingLLM()
    asyncio.run(_orch(llm).handle(_sad_state()))
    text = _system_text(llm).lower()
    assert "sad" in text and "voice cue" in text


def test_no_emotion_hint_when_adapt_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FRIDAY_EMOTION_ADAPT", raising=False)
    monkeypatch.setenv("FRIDAY_LLM_PROVIDER", "fake")
    get_settings.cache_clear()
    llm = _CapturingLLM()
    asyncio.run(_orch(llm).handle(_sad_state()))
    assert "voice cue" not in _system_text(llm).lower()
