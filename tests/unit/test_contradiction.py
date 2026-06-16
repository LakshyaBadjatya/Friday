# © Lakshya Badjatya — Author
"""Unit tests for contradiction detection (over a scripted FakeLLM)."""

from __future__ import annotations

import json

from friday.memory.citations import Source
from friday.memory.contradiction import ContradictionDetector
from friday.providers.llm import FakeLLM, LLMResponse

_EXISTING = [Source(source_id="f1", text="the meeting is on Monday")]


async def test_no_existing_facts_returns_no_contradiction() -> None:
    out = await ContradictionDetector(FakeLLM([])).check("anything", [])
    assert out.contradicts is False


async def test_detects_contradiction_with_source() -> None:
    verdict = json.dumps({"contradicts": True, "source_id": "f1", "why": "Monday vs Tuesday"})
    llm = FakeLLM([LLMResponse(text=verdict)])
    out = await ContradictionDetector(llm).check("the meeting is on Tuesday", _EXISTING)
    assert out.contradicts is True
    assert out.conflicting_source_id == "f1"
    assert out.explanation == "Monday vs Tuesday"


async def test_no_contradiction_verdict() -> None:
    llm = FakeLLM([LLMResponse(text='{"contradicts": false, "source_id": null, "why": ""}')])
    out = await ContradictionDetector(llm).check("the sky is blue", _EXISTING)
    assert out.contradicts is False


async def test_unknown_source_id_is_dropped() -> None:
    verdict = json.dumps({"contradicts": True, "source_id": "ghost", "why": "x"})
    llm = FakeLLM([LLMResponse(text=verdict)])
    out = await ContradictionDetector(llm).check("new", _EXISTING)
    assert out.contradicts is True
    assert out.conflicting_source_id is None  # 'ghost' not among existing ids


async def test_provider_error_is_conservative() -> None:
    out = await ContradictionDetector(FakeLLM([])).check("new", _EXISTING)
    assert out.contradicts is False
    assert "unavailable" in out.explanation


async def test_unparseable_is_conservative() -> None:
    llm = FakeLLM([LLMResponse(text="no json here")])
    out = await ContradictionDetector(llm).check("new", _EXISTING)
    assert out.contradicts is False
