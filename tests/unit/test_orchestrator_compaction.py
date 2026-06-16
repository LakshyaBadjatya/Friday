# © Lakshya Badjatya — Author
"""Unit test for the orchestrator's context-compaction hook (_maybe_compact)."""

from __future__ import annotations

import pytest

from friday.config import get_settings
from friday.core.orchestrator import Orchestrator
from friday.core.state import GraphState
from friday.memory.compaction import Compactor
from friday.memory.short_term import ShortTermMemory
from friday.providers.llm import FakeLLM, LLMResponse, Message
from friday.tools.registry import ToolRegistry


def _orchestrator(memory: ShortTermMemory, compaction: Compactor | None) -> Orchestrator:
    return Orchestrator(
        llm=FakeLLM(responses=[]),
        registry=ToolRegistry(),
        memory=memory,
        persona_path="persona.md",
        compaction=compaction,
    )


def _filled_memory(n: int = 12) -> ShortTermMemory:
    memory = ShortTermMemory()
    for i in range(n):
        memory.append("s1", Message(role="user", content=f"m{i}"))
    return memory


def _compactor() -> Compactor:
    return Compactor(FakeLLM([LLMResponse(text="SUMMARY")]), keep_recent=3, trigger_at=8)


async def test_compaction_replaces_buffer_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FRIDAY_ENABLE_COMPACTION", "true")
    get_settings.cache_clear()
    try:
        memory = _filled_memory(12)
        orch = _orchestrator(memory, _compactor())
        await orch._maybe_compact(GraphState(session_id="s1", user_input="x"))
        hist = memory.history("s1")
        # Buffer is now a summary message followed by the 3 most recent turns.
        assert hist[0].role == "system" and "SUMMARY" in (hist[0].content or "")
        assert [m.content for m in hist[1:]] == ["m9", "m10", "m11"]
    finally:
        get_settings.cache_clear()


async def test_compaction_inert_when_flag_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("FRIDAY_ENABLE_COMPACTION", raising=False)
    monkeypatch.setenv("FRIDAY_LLM_PROVIDER", "fake")
    get_settings.cache_clear()
    try:
        memory = _filled_memory(12)
        orch = _orchestrator(memory, _compactor())
        await orch._maybe_compact(GraphState(session_id="s1", user_input="x"))
        assert len(memory.history("s1")) == 12  # untouched
    finally:
        get_settings.cache_clear()


async def test_compaction_inert_when_unwired(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FRIDAY_ENABLE_COMPACTION", "true")
    get_settings.cache_clear()
    try:
        memory = _filled_memory(12)
        orch = _orchestrator(memory, None)  # no Compactor wired
        await orch._maybe_compact(GraphState(session_id="s1", user_input="x"))
        assert len(memory.history("s1")) == 12  # untouched
    finally:
        get_settings.cache_clear()
