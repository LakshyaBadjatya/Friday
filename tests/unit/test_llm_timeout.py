"""Timeout handling for the NVIDIA NIM adapter (no real network).

A slow endpoint must surface as a typed :class:`ProviderError`, never a raw
``httpx``/``openai`` exception and never a hang. ``respx`` mocks the
OpenAI-compatible endpoint and raises a read timeout as its side effect; the
adapter is constructed with a tiny ``timeout`` and ``max_retries=0`` so the SDK
does not silently retry and multiply latency (retry policy is owned by
:class:`FallbackLLM`).
"""

from __future__ import annotations

import httpx
import pytest
import respx

from friday.errors import ProviderError
from friday.providers.llm import Message, NvidiaNIMProvider

BASE_URL = "https://integrate.api.nvidia.com/v1"
ENDPOINT = f"{BASE_URL}/chat/completions"


def _provider() -> NvidiaNIMProvider:
    return NvidiaNIMProvider(
        api_key="nvapi-test",
        base_url=BASE_URL,
        model="meta/llama-3.3-70b-instruct",
        timeout=0.01,
    )


@respx.mock
async def test_nvidia_read_timeout_raises_provider_error() -> None:
    respx.post(ENDPOINT).mock(side_effect=httpx.ReadTimeout("slow"))
    p = _provider()
    with pytest.raises(ProviderError):
        await p.complete([Message(role="user", content="hi")])


@respx.mock
async def test_nvidia_connect_timeout_raises_provider_error() -> None:
    respx.post(ENDPOINT).mock(side_effect=httpx.ConnectTimeout("slow connect"))
    p = _provider()
    with pytest.raises(ProviderError):
        await p.complete([Message(role="user", content="hi")])
