"""LLM provider abstraction, fakes, fallback wrapper, and the NVIDIA NIM adapter.

This module owns the typed LLM boundary for FRIDAY:

* pydantic v2 models: :class:`Message`, :class:`ToolSpec`, :class:`ToolCall`,
  :class:`Usage`, :class:`LLMResponse`.
* :class:`LLMProvider` — the abstract async ``complete`` contract.
* :class:`FakeLLM` — pops scripted responses (zero network, for tests).
* :class:`FallbackLLM` — primary then secondary exactly once on failure.
* :class:`NvidiaNIMProvider` — real adapter over the OpenAI-compatible NVIDIA
  NIM endpoint.

IMPORTANT: this is the **only** module in the codebase permitted to import an
LLM SDK (``openai``). All business logic depends on :class:`LLMProvider`.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import Any

from openai import AsyncOpenAI, OpenAIError
from pydantic import BaseModel, Field

from friday.errors import ProviderError

logger = logging.getLogger("friday.providers.llm")


# --------------------------------------------------------------------------- #
# Typed boundary models
# --------------------------------------------------------------------------- #
class Message(BaseModel):
    """A single chat message exchanged with an LLM."""

    role: str
    content: str | None = None
    tool_call_id: str | None = None
    name: str | None = None


class ToolSpec(BaseModel):
    """A declarative description of a callable tool exposed to the LLM."""

    name: str
    description: str
    parameters: dict[str, Any]


class ToolCall(BaseModel):
    """A tool invocation requested by the LLM."""

    id: str
    name: str
    arguments: dict[str, Any]


class Usage(BaseModel):
    """Token accounting for a completion."""

    prompt_tokens: int = 0
    completion_tokens: int = 0


class LLMResponse(BaseModel):
    """A normalized response from any :class:`LLMProvider`."""

    text: str | None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)


# --------------------------------------------------------------------------- #
# Provider contract
# --------------------------------------------------------------------------- #
class LLMProvider(ABC):
    """Abstract async LLM contract.

    Implementations must turn a list of :class:`Message` (and optional
    :class:`ToolSpec`) into a normalized :class:`LLMResponse`.
    """

    @abstractmethod
    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
    ) -> LLMResponse:
        """Return a completion for ``messages``, optionally exposing ``tools``."""
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Fakes / wrappers
# --------------------------------------------------------------------------- #
class FakeLLM(LLMProvider):
    """A scripted provider that pops pre-canned responses in order.

    Raises :class:`ProviderError` when the script is exhausted so misconfigured
    tests fail loudly instead of hanging or returning ``None``.
    """

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses: list[LLMResponse] = list(responses)

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
    ) -> LLMResponse:
        if not self._responses:
            raise ProviderError("FakeLLM: no scripted responses remaining")
        return self._responses.pop(0)


class FallbackLLM(LLMProvider):
    """Try ``primary``; on failure fall back to ``secondary`` exactly once.

    A failure is a :class:`ProviderError` or a timeout. If ``secondary`` also
    fails, the secondary's :class:`ProviderError` propagates. Any non-provider
    error from ``primary`` is wrapped in :class:`ProviderError` before the
    switch so the contract stays consistent.
    """

    def __init__(self, primary: LLMProvider, secondary: LLMProvider) -> None:
        self._primary = primary
        self._secondary = secondary

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
    ) -> LLMResponse:
        try:
            return await self._primary.complete(messages, tools)
        except (ProviderError, TimeoutError) as exc:
            logger.warning(
                "LLM primary failed, switching to fallback secondary: %s",
                exc,
            )
            try:
                return await self._secondary.complete(messages, tools)
            except (ProviderError, TimeoutError) as secondary_exc:
                raise ProviderError(
                    f"both LLM providers failed; secondary error: {secondary_exc}"
                ) from secondary_exc


# --------------------------------------------------------------------------- #
# NVIDIA NIM adapter (OpenAI-compatible)
# --------------------------------------------------------------------------- #
class NvidiaNIMProvider(LLMProvider):
    """Real :class:`LLMProvider` over the OpenAI-compatible NVIDIA NIM API.

    Uses the ``openai`` :class:`AsyncOpenAI` client pointed at ``base_url``.
    Maps FRIDAY's typed models to/from the OpenAI wire format and wraps every
    client/transport error in :class:`ProviderError`.
    """

    def __init__(self, api_key: str, base_url: str, model: str) -> None:
        self._model = model
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    @staticmethod
    def _to_openai_messages(messages: list[Message]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for m in messages:
            wire: dict[str, Any] = {"role": m.role, "content": m.content}
            if m.tool_call_id is not None:
                wire["tool_call_id"] = m.tool_call_id
            if m.name is not None:
                wire["name"] = m.name
            out.append(wire)
        return out

    @staticmethod
    def _to_openai_tools(tools: list[ToolSpec] | None) -> list[dict[str, Any]] | None:
        if not tools:
            return None
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in tools
        ]

    @staticmethod
    def _parse_tool_calls(raw_tool_calls: Any) -> list[ToolCall]:
        if not raw_tool_calls:
            return []
        calls: list[ToolCall] = []
        for tc in raw_tool_calls:
            function = tc.function
            raw_args = function.arguments
            if isinstance(raw_args, str):
                try:
                    arguments: dict[str, Any] = json.loads(raw_args) if raw_args else {}
                except json.JSONDecodeError:
                    arguments = {}
            elif isinstance(raw_args, dict):
                arguments = raw_args
            else:
                arguments = {}
            calls.append(ToolCall(id=tc.id, name=function.name, arguments=arguments))
        return calls

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": self._to_openai_messages(messages),
        }
        wire_tools = self._to_openai_tools(tools)
        if wire_tools is not None:
            kwargs["tools"] = wire_tools

        try:
            completion = await self._client.chat.completions.create(**kwargs)
        except OpenAIError as exc:
            raise ProviderError(f"NVIDIA NIM request failed: {exc}") from exc
        except Exception as exc:  # pragma: no cover - defensive transport guard
            raise ProviderError(f"NVIDIA NIM request failed: {exc}") from exc

        try:
            choice = completion.choices[0]
            message = choice.message
            text = message.content
            tool_calls = self._parse_tool_calls(getattr(message, "tool_calls", None))
            raw_usage = completion.usage
            usage = Usage(
                prompt_tokens=getattr(raw_usage, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(raw_usage, "completion_tokens", 0) or 0,
            )
        except (IndexError, AttributeError) as exc:
            raise ProviderError(f"NVIDIA NIM returned an unexpected payload: {exc}") from exc

        return LLMResponse(text=text, tool_calls=tool_calls, usage=usage)
