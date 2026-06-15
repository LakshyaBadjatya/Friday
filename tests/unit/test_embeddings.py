"""Unit tests for ``friday.providers.embeddings``.

The embedding boundary is an :class:`EmbeddingProvider` protocol with one async
method, ``embed``. Two implementations exist: a deterministic, offline
:class:`FakeEmbeddings` (used by every test — no key, no network) and the real
:class:`NvidiaEmbeddings` adapter over the OpenAI-compatible NVIDIA NIM
``/embeddings`` endpoint (its transport errors map to :class:`ProviderError`).

These tests pin the fake's contract: it returns one vector per input text, each
of the configured dimension, deterministic across calls and instances, unit
normalized, and distinct for distinct inputs. The NVIDIA adapter is exercised
with ``respx`` so no real network is touched.
"""

from __future__ import annotations

import math

import httpx
import pytest
import respx

from friday.errors import ProviderError
from friday.providers.embeddings import (
    EmbeddingProvider,
    FakeEmbeddings,
    NvidiaEmbeddings,
)

BASE_URL = "https://integrate.api.nvidia.com/v1"
ENDPOINT = f"{BASE_URL}/embeddings"


def _norm(vec: list[float]) -> float:
    return math.sqrt(sum(component * component for component in vec))


async def test_fake_embeddings_satisfies_protocol() -> None:
    provider: EmbeddingProvider = FakeEmbeddings()
    out = await provider.embed(["hello"])
    assert isinstance(out, list)
    assert len(out) == 1


async def test_fake_embeddings_returns_one_vector_per_text() -> None:
    fake = FakeEmbeddings(dim=64)
    out = await fake.embed(["a", "b", "c"])
    assert len(out) == 3
    assert all(len(vec) == 64 for vec in out)


async def test_fake_embeddings_default_dimension_is_64() -> None:
    fake = FakeEmbeddings()
    out = await fake.embed(["anything"])
    assert len(out[0]) == 64


async def test_fake_embeddings_respects_custom_dimension() -> None:
    fake = FakeEmbeddings(dim=16)
    out = await fake.embed(["x", "y"])
    assert all(len(vec) == 16 for vec in out)


async def test_fake_embeddings_are_deterministic_across_calls() -> None:
    fake = FakeEmbeddings()
    first = await fake.embed(["the quick brown fox"])
    second = await fake.embed(["the quick brown fox"])
    assert first == second


async def test_fake_embeddings_are_deterministic_across_instances() -> None:
    a = await FakeEmbeddings(dim=32).embed(["stable text"])
    b = await FakeEmbeddings(dim=32).embed(["stable text"])
    assert a == b


async def test_fake_embeddings_are_unit_normalized() -> None:
    fake = FakeEmbeddings()
    out = await fake.embed(["normalize me please"])
    assert _norm(out[0]) == pytest.approx(1.0, abs=1e-6)


async def test_fake_embeddings_differ_for_different_texts() -> None:
    fake = FakeEmbeddings()
    out = await fake.embed(["completely different one", "another unrelated string"])
    assert out[0] != out[1]


async def test_fake_embeddings_empty_input_returns_empty() -> None:
    fake = FakeEmbeddings()
    assert await fake.embed([]) == []


@respx.mock
async def test_nvidia_embeddings_parses_response() -> None:
    respx.post(ENDPOINT).mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [
                    {"embedding": [0.1, 0.2, 0.3], "index": 0},
                    {"embedding": [0.4, 0.5, 0.6], "index": 1},
                ],
                "usage": {"prompt_tokens": 4, "total_tokens": 4},
            },
        )
    )
    provider = NvidiaEmbeddings(
        api_key="nvapi-test",
        base_url=BASE_URL,
        model="nvidia/nv-embed-v1",
        dim=3,
    )
    out = await provider.embed(["first", "second"])
    assert out == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]


@respx.mock
async def test_nvidia_embeddings_sends_model_and_input() -> None:
    route = respx.post(ENDPOINT).mock(
        return_value=httpx.Response(
            200,
            json={"data": [{"embedding": [1.0], "index": 0}], "usage": {}},
        )
    )
    provider = NvidiaEmbeddings(
        api_key="nvapi-test",
        base_url=BASE_URL,
        model="nvidia/nv-embed-v1",
        dim=1,
    )
    await provider.embed(["hello world"])
    assert route.called
    import json

    body = json.loads(route.calls.last.request.content)
    assert body["model"] == "nvidia/nv-embed-v1"
    assert body["input"] == ["hello world"]


@respx.mock
async def test_nvidia_embeddings_empty_input_short_circuits() -> None:
    route = respx.post(ENDPOINT)
    provider = NvidiaEmbeddings(
        api_key="nvapi-test",
        base_url=BASE_URL,
        model="nvidia/nv-embed-v1",
        dim=3,
    )
    out = await provider.embed([])
    assert out == []
    assert not route.called


@respx.mock
async def test_nvidia_embeddings_wraps_http_error() -> None:
    respx.post(ENDPOINT).mock(return_value=httpx.Response(500, json={"error": "boom"}))
    provider = NvidiaEmbeddings(
        api_key="nvapi-test",
        base_url=BASE_URL,
        model="nvidia/nv-embed-v1",
        dim=3,
    )
    with pytest.raises(ProviderError):
        await provider.embed(["x"])


@respx.mock
async def test_nvidia_embeddings_wraps_network_error() -> None:
    respx.post(ENDPOINT).mock(side_effect=httpx.ConnectError("no route"))
    provider = NvidiaEmbeddings(
        api_key="nvapi-test",
        base_url=BASE_URL,
        model="nvidia/nv-embed-v1",
        dim=3,
    )
    with pytest.raises(ProviderError):
        await provider.embed(["x"])
