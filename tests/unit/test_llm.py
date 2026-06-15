"""Unit tests for the LLM provider contract, FakeLLM, and FallbackLLM.

These tests pin the contract from Task 0.5 of the implementation plan.
"""

from __future__ import annotations

import logging

import pytest

from friday.errors import ProviderError
from friday.providers.llm import (
    FakeLLM,
    FallbackLLM,
    LLMProvider,
    LLMResponse,
    Message,
    ToolCall,
    ToolSpec,
    Usage,
)


def test_models_construct_and_default() -> None:
    msg = Message(role="user", content="yo")
    assert msg.role == "user"
    assert msg.tool_call_id is None
    assert msg.name is None

    spec = ToolSpec(name="web_search", description="search", parameters={"type": "object"})
    assert spec.name == "web_search"

    call = ToolCall(id="c1", name="web_search", arguments={"query": "x"})
    assert call.arguments == {"query": "x"}

    usage = Usage()
    assert usage.prompt_tokens == 0
    assert usage.completion_tokens == 0

    resp = LLMResponse(text="hi", tool_calls=[], usage=Usage())
    assert resp.text == "hi"
    assert resp.tool_calls == []


async def test_fake_scripted() -> None:
    fake = FakeLLM(responses=[LLMResponse(text="hi", tool_calls=[], usage=Usage())])
    r = await fake.complete([Message(role="user", content="yo")], tools=None)
    assert r.text == "hi"


async def test_fake_pops_in_order() -> None:
    fake = FakeLLM(
        responses=[
            LLMResponse(text="first", tool_calls=[], usage=Usage()),
            LLMResponse(text="second", tool_calls=[], usage=Usage()),
        ]
    )
    r1 = await fake.complete([Message(role="user", content="a")])
    r2 = await fake.complete([Message(role="user", content="b")])
    assert r1.text == "first"
    assert r2.text == "second"


async def test_fake_exhausted_raises() -> None:
    fake = FakeLLM(responses=[])
    with pytest.raises(ProviderError):
        await fake.complete([Message(role="user", content="a")])


async def test_fallback_calls_secondary_once(caplog: pytest.LogCaptureFixture) -> None:
    class Boom(LLMProvider):
        calls = 0

        async def complete(
            self,
            messages: list[Message],
            tools: list[ToolSpec] | None = None,
        ) -> LLMResponse:
            type(self).calls += 1
            raise ProviderError("primary down")

    secondary = FakeLLM(
        responses=[LLMResponse(text="from-secondary", tool_calls=[], usage=Usage())]
    )
    fb = FallbackLLM(primary=Boom(), secondary=secondary)
    with caplog.at_level(logging.WARNING):
        r = await fb.complete([Message(role="user", content="hi")], tools=None)
    assert r.text == "from-secondary"
    assert Boom.calls == 1
    # The switch must be logged.
    assert any("fallback" in rec.getMessage().lower() for rec in caplog.records)


async def test_fallback_both_fail_raises() -> None:
    class Boom(LLMProvider):
        async def complete(
            self,
            messages: list[Message],
            tools: list[ToolSpec] | None = None,
        ) -> LLMResponse:
            raise ProviderError("down")

    fb = FallbackLLM(primary=Boom(), secondary=Boom())
    with pytest.raises(ProviderError):
        await fb.complete([Message(role="user", content="hi")], tools=None)


async def test_fallback_primary_success_skips_secondary() -> None:
    class CountingFake(LLMProvider):
        calls = 0

        async def complete(
            self,
            messages: list[Message],
            tools: list[ToolSpec] | None = None,
        ) -> LLMResponse:
            type(self).calls += 1
            return LLMResponse(text="secondary", tool_calls=[], usage=Usage())

    primary = FakeLLM(responses=[LLMResponse(text="primary", tool_calls=[], usage=Usage())])
    secondary = CountingFake()
    fb = FallbackLLM(primary=primary, secondary=secondary)
    r = await fb.complete([Message(role="user", content="hi")])
    assert r.text == "primary"
    assert CountingFake.calls == 0


async def test_fallback_on_timeout() -> None:
    class Slow(LLMProvider):
        async def complete(
            self,
            messages: list[Message],
            tools: list[ToolSpec] | None = None,
        ) -> LLMResponse:
            raise TimeoutError("primary timed out")

    secondary = FakeLLM(
        responses=[LLMResponse(text="recovered", tool_calls=[], usage=Usage())]
    )
    fb = FallbackLLM(primary=Slow(), secondary=secondary)
    r = await fb.complete([Message(role="user", content="hi")])
    assert r.text == "recovered"
