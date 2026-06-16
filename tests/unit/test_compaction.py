# © Lakshya Badjatya — Author
"""Unit tests for context compaction (over a scripted FakeLLM)."""

from __future__ import annotations

import pytest

from friday.memory.compaction import CompactionResult, Compactor
from friday.providers.llm import FakeLLM, LLMResponse, Message


def _history(n: int) -> list[Message]:
    return [
        Message(role="user" if i % 2 == 0 else "assistant", content=f"m{i}")
        for i in range(n)
    ]


async def test_below_threshold_returns_none() -> None:
    llm = FakeLLM(responses=[])  # never called
    out = await Compactor(llm, keep_recent=2, trigger_at=8).maybe_compact(_history(8))
    assert out is None


async def test_compacts_older_and_keeps_recent_tail() -> None:
    llm = FakeLLM(responses=[LLMResponse(text="condensed note")])
    hist = _history(10)
    out = await Compactor(llm, keep_recent=3, trigger_at=8).maybe_compact(hist)
    assert isinstance(out, CompactionResult)
    assert out.summary == "condensed note"
    assert out.compacted_count == 7  # 10 - 3 kept
    assert [m.content for m in out.kept] == ["m7", "m8", "m9"]


async def test_provider_error_returns_none() -> None:
    llm = FakeLLM(responses=[])  # exhausted -> ProviderError -> None
    out = await Compactor(llm, keep_recent=2, trigger_at=4).maybe_compact(_history(10))
    assert out is None


async def test_empty_summary_returns_none() -> None:
    llm = FakeLLM(responses=[LLMResponse(text="   ")])
    out = await Compactor(llm, keep_recent=2, trigger_at=4).maybe_compact(_history(10))
    assert out is None


def test_invalid_thresholds_rejected() -> None:
    llm = FakeLLM(responses=[])
    with pytest.raises(ValueError):
        Compactor(llm, keep_recent=-1)
    with pytest.raises(ValueError):
        Compactor(llm, keep_recent=10, trigger_at=4)
