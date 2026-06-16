# © Lakshya Badjatya — Author
"""Unit tests for the OpenRouter / OpenCode adapters and the per-call model
override on the OpenAI-compatible providers (respx, no real network).

Both OpenRouter and OpenCode expose an OpenAI-compatible ``/chat/completions``
surface, so the adapters are thin :class:`_OpenAICompatProvider` subclasses that
inherit all mapping/parsing/error-wrapping and only relabel their errors. The
per-call ``model`` keyword lets a single constructed provider service many
catalog models without rebuilding a client per model.
"""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from friday.errors import ProviderError
from friday.providers.llm import (
    FakeLLM,
    FallbackLLM,
    LLMResponse,
    Message,
    OpenCodeProvider,
    OpenRouterProvider,
    ToolSpec,
    Usage,
)

OPENROUTER_BASE = "https://openrouter.ai/api/v1"
OPENROUTER_ENDPOINT = f"{OPENROUTER_BASE}/chat/completions"
OPENCODE_BASE = "https://opencode.ai/zen/v1"
OPENCODE_ENDPOINT = f"{OPENCODE_BASE}/chat/completions"


def _openrouter(model: str = "google/gemma-4-31b-it:free") -> OpenRouterProvider:
    return OpenRouterProvider(
        api_key="sk-or-test",
        base_url=OPENROUTER_BASE,
        model=model,
    )


def _opencode(model: str = "mimo-v2.5-free") -> OpenCodeProvider:
    return OpenCodeProvider(
        api_key="oc-test",
        base_url=OPENCODE_BASE,
        model=model,
    )


def _ok_response(text: str = "hello") -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": text, "tool_calls": None}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2},
        },
    )


# --------------------------------------------------------------------------- #
# OpenRouter
# --------------------------------------------------------------------------- #
@respx.mock
async def test_openrouter_parses_response() -> None:
    respx.post(OPENROUTER_ENDPOINT).mock(return_value=_ok_response("hi from router"))
    r = await _openrouter().complete([Message(role="user", content="hi")], tools=None)
    assert r.text == "hi from router"
    assert r.usage.prompt_tokens == 3
    assert r.usage.completion_tokens == 2
    assert r.tool_calls == []


@respx.mock
async def test_openrouter_sends_model() -> None:
    route = respx.post(OPENROUTER_ENDPOINT).mock(return_value=_ok_response("ok"))
    await _openrouter().complete([Message(role="user", content="hi")])
    body = json.loads(route.calls.last.request.content)
    assert body["model"] == "google/gemma-4-31b-it:free"


@respx.mock
async def test_openrouter_maps_429_to_provider_error() -> None:
    respx.post(OPENROUTER_ENDPOINT).mock(
        return_value=httpx.Response(429, json={"error": "rate limited"})
    )
    with pytest.raises(ProviderError):
        await _openrouter().complete([Message(role="user", content="hi")])


# --------------------------------------------------------------------------- #
# OpenCode
# --------------------------------------------------------------------------- #
@respx.mock
async def test_opencode_parses_response() -> None:
    respx.post(OPENCODE_ENDPOINT).mock(return_value=_ok_response("hi from zen"))
    r = await _opencode().complete([Message(role="user", content="hi")], tools=None)
    assert r.text == "hi from zen"
    assert r.tool_calls == []


@respx.mock
async def test_opencode_sends_model() -> None:
    route = respx.post(OPENCODE_ENDPOINT).mock(return_value=_ok_response("ok"))
    await _opencode().complete([Message(role="user", content="hi")])
    body = json.loads(route.calls.last.request.content)
    assert body["model"] == "mimo-v2.5-free"


@respx.mock
async def test_opencode_maps_429_to_provider_error() -> None:
    respx.post(OPENCODE_ENDPOINT).mock(
        return_value=httpx.Response(429, json={"error": "rate limited"})
    )
    with pytest.raises(ProviderError):
        await _opencode().complete([Message(role="user", content="hi")])


# --------------------------------------------------------------------------- #
# Per-call model override
# --------------------------------------------------------------------------- #
@respx.mock
async def test_per_call_model_override_changes_request_body() -> None:
    route = respx.post(OPENROUTER_ENDPOINT).mock(return_value=_ok_response("ok"))
    p = _openrouter(model="google/gemma-4-31b-it:free")
    await p.complete(
        [Message(role="user", content="hi")],
        model="openai/gpt-oss-120b:free",
    )
    body = json.loads(route.calls.last.request.content)
    assert body["model"] == "openai/gpt-oss-120b:free"


@respx.mock
async def test_per_call_model_none_uses_construction_model() -> None:
    route = respx.post(OPENROUTER_ENDPOINT).mock(return_value=_ok_response("ok"))
    p = _openrouter(model="qwen/qwen3-coder:free")
    await p.complete([Message(role="user", content="hi")], model=None)
    body = json.loads(route.calls.last.request.content)
    assert body["model"] == "qwen/qwen3-coder:free"


@respx.mock
async def test_per_call_model_does_not_mutate_construction_model() -> None:
    respx.post(OPENROUTER_ENDPOINT).mock(return_value=_ok_response("ok"))
    p = _openrouter(model="base-model")
    await p.complete([Message(role="user", content="hi")], model="override-model")
    route = respx.post(OPENROUTER_ENDPOINT).mock(return_value=_ok_response("ok2"))
    await p.complete([Message(role="user", content="hi")])
    body = json.loads(route.calls.last.request.content)
    assert body["model"] == "base-model"


# --------------------------------------------------------------------------- #
# Backward compatibility / forwarding
# --------------------------------------------------------------------------- #
async def test_fake_ignores_model_keyword() -> None:
    fake = FakeLLM(responses=[LLMResponse(text="hi", tool_calls=[], usage=Usage())])
    r = await fake.complete([Message(role="user", content="yo")], model="whatever")
    assert r.text == "hi"


async def test_fallback_forwards_model_to_secondary() -> None:
    seen: dict[str, str | None] = {}

    class Boom(FakeLLM):
        async def complete(
            self,
            messages: list[Message],
            tools: list[ToolSpec] | None = None,
            *,
            model: str | None = None,
        ) -> LLMResponse:
            raise ProviderError("primary down")

    class Capturing(FakeLLM):
        async def complete(
            self,
            messages: list[Message],
            tools: list[ToolSpec] | None = None,
            *,
            model: str | None = None,
        ) -> LLMResponse:
            seen["model"] = model
            return LLMResponse(text="ok", tool_calls=[], usage=Usage())

    fb = FallbackLLM(primary=Boom([]), secondary=Capturing([]))
    r = await fb.complete([Message(role="user", content="hi")], model="m-x")
    assert r.text == "ok"
    assert seen["model"] == "m-x"


async def test_fallback_forwards_model_to_primary() -> None:
    seen: dict[str, str | None] = {}

    class Capturing(FakeLLM):
        async def complete(
            self,
            messages: list[Message],
            tools: list[ToolSpec] | None = None,
            *,
            model: str | None = None,
        ) -> LLMResponse:
            seen["model"] = model
            return LLMResponse(text="primary", tool_calls=[], usage=Usage())

    fb = FallbackLLM(primary=Capturing([]), secondary=FakeLLM([]))
    r = await fb.complete([Message(role="user", content="hi")], model="m-y")
    assert r.text == "primary"
    assert seen["model"] == "m-y"
