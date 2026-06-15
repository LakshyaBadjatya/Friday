"""Unit tests for the offline degraded-mode LLM provider.

These pin the contract for the ``offline`` feature slice:

* :class:`OfflineLLM` is a network-free :class:`LLMProvider` whose
  ``complete`` returns a deterministic, honest "offline mode" response without
  performing any network I/O or fabricating an answer.
* :func:`select_llm` returns an :class:`OfflineLLM` when
  ``settings.enable_offline_mode`` is on, and otherwise returns the supplied
  primary provider unchanged.

All tests are fully offline: no respx, no sockets, no monkeypatching of the
network is needed because :class:`OfflineLLM` never touches a transport.
"""

from __future__ import annotations

from friday.config import Settings
from friday.providers.llm import (
    FakeLLM,
    LLMProvider,
    LLMResponse,
    Message,
    ToolSpec,
    Usage,
)
from friday.providers.offline import OfflineLLM, select_llm


def test_offline_is_llm_provider() -> None:
    assert issubclass(OfflineLLM, LLMProvider)
    assert isinstance(OfflineLLM(), LLMProvider)


async def test_offline_complete_returns_honest_response() -> None:
    provider = OfflineLLM()
    resp = await provider.complete([Message(role="user", content="hello")])

    assert isinstance(resp, LLMResponse)
    assert resp.text is not None
    lowered = resp.text.lower()
    # Honest, on-message degraded-mode notice.
    assert "offline" in lowered
    assert "unavailable" in lowered
    # It must not fabricate an answer or emit any tool calls.
    assert resp.tool_calls == []
    # No tokens are consumed because no model was queried.
    assert resp.usage == Usage()


async def test_offline_complete_is_deterministic() -> None:
    provider = OfflineLLM()
    first = await provider.complete([Message(role="user", content="a")])
    second = await provider.complete(
        [Message(role="user", content="something completely different")],
        tools=[ToolSpec(name="t", description="d", parameters={"type": "object"})],
    )
    # Same fixed response regardless of prompt or available tools.
    assert first.text == second.text
    assert first == second


async def test_offline_complete_ignores_tools_no_calls() -> None:
    provider = OfflineLLM()
    resp = await provider.complete(
        [Message(role="user", content="search the web")],
        tools=[
            ToolSpec(
                name="web_search",
                description="search",
                parameters={"type": "object"},
            )
        ],
    )
    # Even when tools are offered, offline mode never invokes them.
    assert resp.tool_calls == []


def test_select_llm_picks_offline_when_flag_on() -> None:
    primary = FakeLLM(responses=[LLMResponse(text="primary", usage=Usage())])
    settings = Settings(enable_offline_mode=True)
    chosen = select_llm(settings, primary)
    assert isinstance(chosen, OfflineLLM)
    assert chosen is not primary


def test_select_llm_returns_primary_when_flag_off() -> None:
    primary = FakeLLM(responses=[LLMResponse(text="primary", usage=Usage())])
    settings = Settings(enable_offline_mode=False)
    chosen = select_llm(settings, primary)
    assert chosen is primary


def test_select_llm_default_settings_returns_primary() -> None:
    # The flag is off by default, so the unmodified build keeps the primary.
    primary: LLMProvider = FakeLLM(responses=[LLMResponse(text="p", usage=Usage())])
    chosen = select_llm(Settings(), primary)
    assert chosen is primary
