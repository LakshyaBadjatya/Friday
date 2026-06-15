"""Unit tests for the NVIDIA NIM adapter using respx (no real network)."""

from __future__ import annotations

import httpx
import pytest
import respx

from friday.errors import ProviderError
from friday.providers.llm import Message, NvidiaNIMProvider, ToolSpec

BASE_URL = "https://integrate.api.nvidia.com/v1"
ENDPOINT = f"{BASE_URL}/chat/completions"


def _provider() -> NvidiaNIMProvider:
    return NvidiaNIMProvider(
        api_key="nvapi-test",
        base_url=BASE_URL,
        model="meta/llama-3.3-70b-instruct",
    )


@respx.mock
async def test_nvidia_parses_response() -> None:
    respx.post(ENDPOINT).mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "hello from nim", "tool_calls": None}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2},
            },
        )
    )
    p = _provider()
    r = await p.complete([Message(role="user", content="hi")], tools=None)
    assert r.text == "hello from nim"
    assert r.usage.completion_tokens == 2
    assert r.usage.prompt_tokens == 3
    assert r.tool_calls == []


@respx.mock
async def test_nvidia_sends_messages_and_model() -> None:
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
    sent = route.calls.last.request
    import json

    body = json.loads(sent.content)
    assert body["model"] == "meta/llama-3.3-70b-instruct"
    assert body["messages"][0] == {"role": "system", "content": "be terse"}
    assert body["messages"][1] == {"role": "user", "content": "hi"}


@respx.mock
async def test_nvidia_maps_tools() -> None:
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
    import json

    body = json.loads(route.calls.last.request.content)
    assert body["tools"][0]["type"] == "function"
    fn = body["tools"][0]["function"]
    assert fn["name"] == "web_search"
    assert fn["description"] == "search the web"
    assert fn["parameters"]["properties"]["query"]["type"] == "string"


@respx.mock
async def test_nvidia_parses_tool_calls() -> None:
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
async def test_nvidia_wraps_http_error_in_provider_error() -> None:
    respx.post(ENDPOINT).mock(
        return_value=httpx.Response(500, json={"error": "boom"})
    )
    p = _provider()
    with pytest.raises(ProviderError):
        await p.complete([Message(role="user", content="hi")])


@respx.mock
async def test_nvidia_wraps_network_error_in_provider_error() -> None:
    respx.post(ENDPOINT).mock(side_effect=httpx.ConnectError("no route"))
    p = _provider()
    with pytest.raises(ProviderError):
        await p.complete([Message(role="user", content="hi")])
