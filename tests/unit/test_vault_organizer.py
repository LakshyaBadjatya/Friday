"""Unit tests for automatic classification of captures."""

from __future__ import annotations

import pytest

from friday.errors import ProviderError
from friday.providers.llm import FakeLLM, LLMProvider, LLMResponse, Message, ToolSpec
from friday.vault.organizer import Organizer


class RecordingLLM(LLMProvider):
    """Wraps a :class:`FakeLLM` and records every call it receives.

    ``FakeLLM`` itself keeps no record of calls made, so this thin wrapper
    is what lets tests assert "the model was/wasn't called" and inspect the
    exact prompt that went out.
    """

    def __init__(self, reply: str) -> None:
        self._inner = FakeLLM(responses=[LLMResponse(text=reply)])
        self.calls: list[list[Message]] = []

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        *,
        model: str | None = None,
    ) -> LLMResponse:
        self.calls.append(messages)
        return await self._inner.complete(messages, tools, model=model)


class RaisingLLM(LLMProvider):
    """A provider that always fails, simulating an outage or timeout."""

    def __init__(self, exc: BaseException) -> None:
        self._exc = exc

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        *,
        model: str | None = None,
    ) -> LLMResponse:
        raise self._exc


@pytest.mark.asyncio
async def test_classifies_from_the_model_reply() -> None:
    llm = RecordingLLM(
        reply='{"kind": "problem", "subject": "physics", '
        '"chapter": "current electricity", "tags": ["emf", "internal resistance"]}'
    )
    result = await Organizer(llm).classify(ocr_text="A cell of emf 4 V...", caption="")
    assert result.kind == "problem"
    assert result.subject == "physics"
    assert result.chapter == "current electricity"
    assert "emf" in result.tags


@pytest.mark.asyncio
async def test_falls_back_to_unknown_on_unparseable_reply() -> None:
    result = await Organizer(RecordingLLM(reply="I have no idea")).classify(
        ocr_text="???", caption=""
    )
    assert result.kind == "unknown"
    assert result.subject == ""
    assert result.tags == []


@pytest.mark.asyncio
async def test_tags_are_lowercased_and_deduped() -> None:
    llm = RecordingLLM(reply='{"kind": "note", "tags": ["Physics", "physics", "EMF"]}')
    result = await Organizer(llm).classify(ocr_text="x", caption="")
    assert result.tags == ["physics", "emf"]


@pytest.mark.asyncio
async def test_empty_input_short_circuits_without_a_model_call() -> None:
    llm = RecordingLLM(reply='{"kind": "problem"}')
    result = await Organizer(llm).classify(ocr_text="", caption="")
    assert result.kind == "unknown"
    assert llm.calls == []


@pytest.mark.asyncio
async def test_whitespace_only_ocr_with_caption_still_calls_the_model() -> None:
    llm = RecordingLLM(reply='{"kind": "photo"}')
    result = await Organizer(llm).classify(ocr_text="   \n\t", caption="a whiteboard shot")
    assert len(llm.calls) == 1
    assert result.kind == "photo"


@pytest.mark.asyncio
async def test_provider_outage_degrades_to_unknown_instead_of_raising() -> None:
    """The single most important guarantee: a dead/slow provider must never

    fail the capture it is describing. Both a domain ProviderError (the real
    adapters raise this on connection/timeout failure) and a bare TimeoutError
    must be swallowed.
    """
    result = await Organizer(RaisingLLM(ProviderError("NIM is down"))).classify(
        ocr_text="A cell of emf 4 V...", caption=""
    )
    assert result.kind == "unknown"
    assert result.subject == ""
    assert result.tags == []

    result = await Organizer(RaisingLLM(TimeoutError("timed out"))).classify(
        ocr_text="A cell of emf 4 V...", caption=""
    )
    assert result.kind == "unknown"


@pytest.mark.asyncio
async def test_reply_wrapped_in_json_code_fence() -> None:
    llm = RecordingLLM(reply='```json\n{"kind": "receipt", "tags": ["grocery"]}\n```')
    result = await Organizer(llm).classify(ocr_text="Total: $42.10", caption="")
    assert result.kind == "receipt"
    assert result.tags == ["grocery"]


@pytest.mark.asyncio
async def test_reply_with_prose_before_and_after_the_json() -> None:
    llm = RecordingLLM(
        reply='Sure thing! Here you go:\n{"kind": "document", "tags": ["form"]}\nHope that helps.'
    )
    result = await Organizer(llm).classify(ocr_text="Please fill in section B", caption="")
    assert result.kind == "document"
    assert result.tags == ["form"]


@pytest.mark.asyncio
async def test_valid_json_but_not_an_object_falls_back_to_unknown() -> None:
    for reply in ('["problem", "physics"]', '"just a string"', "null", "42"):
        result = await Organizer(RecordingLLM(reply=reply)).classify(
            ocr_text="something", caption=""
        )
        assert result.kind == "unknown", reply
        assert result.tags == []


@pytest.mark.asyncio
async def test_tags_field_wrong_shape_is_ignored_not_fatal() -> None:
    llm = RecordingLLM(reply='{"kind": "note", "tags": "physics"}')
    result = await Organizer(llm).classify(ocr_text="x", caption="")
    assert result.kind == "note"
    assert result.tags == []


@pytest.mark.asyncio
async def test_non_string_tag_entries_are_dropped_not_fatal() -> None:
    llm = RecordingLLM(reply='{"kind": "note", "tags": ["ok", 7, {"nested": true}, null]}')
    result = await Organizer(llm).classify(ocr_text="x", caption="")
    assert result.kind == "note"
    assert result.tags == ["ok"]


@pytest.mark.asyncio
async def test_more_than_five_tags_is_capped_at_five() -> None:
    llm = RecordingLLM(
        reply='{"kind": "note", "tags": ["a", "b", "c", "d", "e", "f", "g"]}'
    )
    result = await Organizer(llm).classify(ocr_text="x", caption="")
    assert result.tags == ["a", "b", "c", "d", "e"]


@pytest.mark.asyncio
async def test_long_ocr_text_is_truncated_in_the_prompt() -> None:
    ocr_text = ("A" * 4000) + "OVERFLOW-MARKER-SHOULD-BE-CUT"
    llm = RecordingLLM(reply='{"kind": "problem"}')
    await Organizer(llm).classify(ocr_text=ocr_text, caption="")

    assert len(llm.calls) == 1
    sent_content = llm.calls[0][0].content or ""
    assert "AAAAAAAAAA" in sent_content
    assert "OVERFLOW-MARKER-SHOULD-BE-CUT" not in sent_content
