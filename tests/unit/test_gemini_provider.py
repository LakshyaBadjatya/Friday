"""Unit tests for the Gemini adapter using respx (no real network).

Gemini is exposed through its OpenAI-compatible endpoint, so the adapter is
structurally identical to :class:`NvidiaNIMProvider`: it maps FRIDAY's typed
models to/from the OpenAI wire format, parses text + usage, and wraps every
transport/HTTP error in a typed :class:`ProviderError`.
"""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from friday.errors import ProviderError
from friday.providers.llm import GeminiProvider, Message, ToolSpec

BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"


def _provider() -> GeminiProvider:
    return GeminiProvider(
        api_key="gemini-test",
        base_url=BASE_URL,
        model="gemini-2.0-flash",
    )


@respx.mock
async def test_gemini_parses_response() -> None:
    respx.post(ENDPOINT).mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": "hello from gemini", "tool_calls": None}}
                ],
                "usage": {"prompt_tokens": 7, "completion_tokens": 4},
            },
        )
    )
    p = _provider()
    r = await p.complete([Message(role="user", content="hi")], tools=None)
    assert r.text == "hello from gemini"
    assert r.usage.prompt_tokens == 7
    assert r.usage.completion_tokens == 4
    assert r.tool_calls == []


@respx.mock
async def test_gemini_sends_messages_and_model() -> None:
    route = respx.post(ENDPOINT).mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "ok", "tool_calls": None}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )
    )
    p = _provider()
    await p.complete(
        [
            Message(role="system", content="be terse"),
            Message(role="user", content="hi"),
        ]
    )
    assert route.called
    body = json.loads(route.calls.last.request.content)
    assert body["model"] == "gemini-2.0-flash"
    assert body["messages"][0] == {"role": "system", "content": "be terse"}
    assert body["messages"][1] == {"role": "user", "content": "hi"}


@respx.mock
async def test_gemini_maps_tools() -> None:
    route = respx.post(ENDPOINT).mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "ok", "tool_calls": None}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )
    )
    p = _provider()
    tools = [
        ToolSpec(
            name="web_search",
            description="search the web",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        )
    ]
    await p.complete([Message(role="user", content="search cats")], tools=tools)
    body = json.loads(route.calls.last.request.content)
    fn = body["tools"][0]["function"]
    assert body["tools"][0]["type"] == "function"
    assert fn["name"] == "web_search"
    assert fn["description"] == "search the web"
    assert fn["parameters"]["properties"]["query"]["type"] == "string"


@respx.mock
async def test_gemini_parses_tool_calls() -> None:
    respx.post(ENDPOINT).mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "web_search",
                                        "arguments": '{"query": "cats"}',
                                    },
                                }
                            ],
                        }
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 0},
            },
        )
    )
    p = _provider()
    r = await p.complete([Message(role="user", content="search cats")])
    assert r.text is None
    assert len(r.tool_calls) == 1
    tc = r.tool_calls[0]
    assert tc.id == "call_1"
    assert tc.name == "web_search"
    assert tc.arguments == {"query": "cats"}


@respx.mock
async def test_gemini_wraps_http_429_in_provider_error() -> None:
    respx.post(ENDPOINT).mock(
        return_value=httpx.Response(429, json={"error": "rate limited"})
    )
    p = _provider()
    with pytest.raises(ProviderError):
        await p.complete([Message(role="user", content="hi")])


@respx.mock
async def test_gemini_wraps_network_error_in_provider_error() -> None:
    respx.post(ENDPOINT).mock(side_effect=httpx.ConnectError("no route"))
    p = _provider()
    with pytest.raises(ProviderError):
        await p.complete([Message(role="user", content="hi")])
