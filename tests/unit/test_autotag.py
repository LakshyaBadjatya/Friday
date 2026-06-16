# © Lakshya Badjatya — Author
"""Unit tests for auto-tagging (over a scripted FakeLLM)."""

from __future__ import annotations

import pytest

from friday.memory.autotag import AutoTagger
from friday.providers.llm import FakeLLM, LLMResponse


async def test_parses_and_normalizes_tags() -> None:
    llm = FakeLLM([LLMResponse(text='["AI", " Notes ", "ai"]')])  # dup + casing + space
    tags = await AutoTagger(llm).tag("a note about AI")
    assert tags == ["ai", "notes"]  # lowercased, trimmed, de-duped, order kept


async def test_allowed_vocabulary_filters() -> None:
    llm = FakeLLM([LLMResponse(text='["work", "personal", "spam"]')])
    tagger = AutoTagger(llm, allowed_tags=["work", "personal"])
    assert await tagger.tag("x") == ["work", "personal"]  # 'spam' dropped


async def test_max_tags_caps_output() -> None:
    llm = FakeLLM([LLMResponse(text='["a","b","c","d","e","f"]')])
    assert await AutoTagger(llm, max_tags=3).tag("x") == ["a", "b", "c"]


async def test_provider_error_returns_empty() -> None:
    assert await AutoTagger(FakeLLM([])).tag("x") == []


async def test_unparseable_returns_empty() -> None:
    assert await AutoTagger(FakeLLM([LLMResponse(text="nope")])).tag("x") == []


def test_invalid_max_tags_rejected() -> None:
    with pytest.raises(ValueError):
        AutoTagger(FakeLLM([]), max_tags=0)
