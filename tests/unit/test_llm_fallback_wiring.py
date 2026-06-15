"""Wiring tests for the provider-abstracted Gemini fallback (no network).

Two concerns are pinned here:

* :func:`friday.app._build_llm` selects a :class:`FallbackLLM` (NVIDIA primary,
  Gemini secondary) only when ``llm_fallback_provider == "gemini"`` *and* a
  Gemini key is present; otherwise it falls back to the existing single-provider
  behaviour. None of this touches the network — the provider clients are lazy.
* A :class:`FallbackLLM` whose primary raises :class:`ProviderError` returns the
  secondary's answer (the secondary stands in for the live Gemini adapter).
"""

from __future__ import annotations

from friday.app import _build_llm
from friday.config import Settings
from friday.errors import ProviderError
from friday.providers.llm import (
    FakeLLM,
    FallbackLLM,
    GeminiProvider,
    LLMProvider,
    LLMResponse,
    Message,
    NvidiaNIMProvider,
    ToolSpec,
    Usage,
)


def test_build_llm_wires_gemini_fallback() -> None:
    settings = Settings(
        _env_file=None,
        llm_provider="nvidia",
        nvidia_api_key="nvapi-test",
        llm_fallback_provider="gemini",
        gemini_api_key="gemini-test",
    )
    llm = _build_llm(settings)
    assert isinstance(llm, FallbackLLM)
    assert isinstance(llm._primary, NvidiaNIMProvider)
    assert isinstance(llm._secondary, GeminiProvider)


def test_build_llm_no_fallback_when_provider_none() -> None:
    settings = Settings(
        _env_file=None,
        llm_provider="nvidia",
        nvidia_api_key="nvapi-test",
        llm_fallback_provider="none",
        gemini_api_key="gemini-test",
    )
    llm = _build_llm(settings)
    assert isinstance(llm, NvidiaNIMProvider)


def test_build_llm_no_fallback_when_gemini_key_missing() -> None:
    settings = Settings(
        _env_file=None,
        llm_provider="nvidia",
        nvidia_api_key="nvapi-test",
        llm_fallback_provider="gemini",
        gemini_api_key=None,
    )
    llm = _build_llm(settings)
    assert isinstance(llm, NvidiaNIMProvider)


def test_build_llm_fake_path_unchanged_with_fallback_configured() -> None:
    settings = Settings(
        _env_file=None,
        llm_provider="fake",
        llm_fallback_provider="gemini",
        gemini_api_key="gemini-test",
    )
    llm = _build_llm(settings)
    assert isinstance(llm, FakeLLM)


async def test_fallback_returns_gemini_secondary_answer() -> None:
    class Boom(LLMProvider):
        async def complete(
            self,
            messages: list[Message],
            tools: list[ToolSpec] | None = None,
        ) -> LLMResponse:
            raise ProviderError("primary down")

    secondary = FakeLLM(
        responses=[LLMResponse(text="from-gemini", tool_calls=[], usage=Usage())]
    )
    fb = FallbackLLM(primary=Boom(), secondary=secondary)
    r = await fb.complete([Message(role="user", content="hi")], tools=None)
    assert r.text == "from-gemini"
