"""Offline degraded-mode LLM provider.

When FRIDAY runs in strict offline mode (``settings.enable_offline_mode``), no
outbound network is permitted, so the real language model is unreachable. Rather
than fail hard, the assistant degrades gracefully: routing, memory, and tools
still work, and the LLM boundary is satisfied by :class:`OfflineLLM` — a local,
network-free :class:`~friday.providers.llm.LLMProvider` that returns a fixed,
honest notice instead of fabricating an answer.

This module performs **no** network I/O and imports no LLM SDK. The selection
helper :func:`select_llm` lets the runtime swap the primary provider for an
:class:`OfflineLLM` based purely on configuration.
"""

from __future__ import annotations

from friday.config import Settings
from friday.providers.llm import (
    LLMProvider,
    LLMResponse,
    Message,
    ToolSpec,
    Usage,
)

#: The single, honest degraded-mode reply. It is deliberately fixed so the
#: response is fully deterministic and never fabricates a model answer.
OFFLINE_MESSAGE = (
    "I'm in offline mode; the language model is unavailable, but routing, "
    "memory, and tools still work."
)


class OfflineLLM(LLMProvider):
    """A local, network-free :class:`LLMProvider` for degraded offline mode.

    :meth:`complete` always returns the same :class:`LLMResponse` carrying
    :data:`OFFLINE_MESSAGE`. It performs no network I/O, requests no tool calls,
    and reports zero token usage (no model was queried). The response is
    independent of the prompt and of any offered tools, so it is fully
    deterministic.
    """

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        *,
        model: str | None = None,
    ) -> LLMResponse:
        """Return the fixed offline notice without touching the network.

        ``model`` is accepted for contract parity but ignored: the offline reply
        is deliberately fixed and never queries any model.
        """
        return LLMResponse(text=OFFLINE_MESSAGE, tool_calls=[], usage=Usage())


def select_llm(settings: Settings, primary: LLMProvider) -> LLMProvider:
    """Return the provider appropriate for the current offline configuration.

    When ``settings.enable_offline_mode`` is on, return a fresh
    :class:`OfflineLLM` so no outbound LLM call is ever attempted. Otherwise
    return ``primary`` unchanged.
    """
    if settings.enable_offline_mode:
        return OfflineLLM()
    return primary
