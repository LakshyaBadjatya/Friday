# © Lakshya Badjatya — Author
"""Unit tests for the PII-redacting LLM wrapper."""

from __future__ import annotations

from friday.providers.llm import LLMProvider, LLMResponse, Message, ToolSpec
from friday.providers.redacting import RedactingLLM


class _RecordingLLM(LLMProvider):
    """Captures exactly what messages/tools/model the wrapper forwarded."""

    def __init__(self) -> None:
        self.seen: list[Message] = []
        self.tools: list[ToolSpec] | None = None
        self.model: str | None = None

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        *,
        model: str | None = None,
    ) -> LLMResponse:
        self.seen = messages
        self.tools = tools
        self.model = model
        return LLMResponse(text="ok")


async def test_pii_scrubbed_before_inner() -> None:
    inner = _RecordingLLM()
    await RedactingLLM(inner).complete(
        [Message(role="user", content="mail me at a@b.com or call 555-123-4567")]
    )
    assert inner.seen[0].content == "mail me at [EMAIL] or call [PHONE]"


async def test_clean_message_passes_through() -> None:
    inner = _RecordingLLM()
    msgs = [Message(role="user", content="just a normal question")]
    await RedactingLLM(inner).complete(msgs)
    assert inner.seen[0].content == "just a normal question"


async def test_none_content_preserved() -> None:
    inner = _RecordingLLM()
    await RedactingLLM(inner).complete([Message(role="assistant", content=None)])
    assert inner.seen[0].content is None


async def test_tools_and_model_forwarded() -> None:
    inner = _RecordingLLM()
    spec = ToolSpec(name="t", description="d", parameters={})
    await RedactingLLM(inner).complete(
        [Message(role="user", content="hi")], [spec], model="m1"
    )
    assert inner.tools == [spec]
    assert inner.model == "m1"
