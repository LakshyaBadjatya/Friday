# © Lakshya Badjatya — Author
"""Unit tests for the ensemble / debate engine (over a scripted FakeLLM)."""

from __future__ import annotations

from friday.core.ensemble import Ensemble
from friday.providers.llm import FakeLLM, LLMResponse

_OPERATORS: list[tuple[str, str]] = [
    ("VISION", "You are VISION."),
    ("GECKO", "You are GECKO."),
]


async def test_debate_synthesizes_multiple_drafts() -> None:
    # Two drafts, then one synthesis reply.
    llm = FakeLLM(
        responses=[
            LLMResponse(text="vision draft"),
            LLMResponse(text="gecko draft"),
            LLMResponse(text="fused answer"),
        ]
    )
    result = await Ensemble(llm).debate("What is the plan?", _OPERATORS)

    assert [d.operator for d in result.drafts] == ["VISION", "GECKO"]
    assert all(d.ok for d in result.drafts)
    assert result.contributors == ["VISION", "GECKO"]
    assert result.synthesis == "fused answer"


async def test_failed_draft_is_captured_not_raised() -> None:
    # Only one scripted response: the second operator's draft exhausts the
    # FakeLLM (ProviderError), which must be captured as a failed draft. With one
    # usable draft, synthesis returns it verbatim (no further model call).
    llm = FakeLLM(responses=[LLMResponse(text="vision draft")])
    result = await Ensemble(llm).debate("Q", _OPERATORS)

    assert result.drafts[0].ok is True
    assert result.drafts[1].ok is False
    assert result.drafts[1].error  # human-readable reason present
    assert result.contributors == ["VISION"]
    assert result.synthesis == "vision draft"


async def test_synthesis_failure_falls_back_to_first_draft() -> None:
    # Exactly two responses: both drafts succeed, but the synthesis call then
    # exhausts the FakeLLM and must fall back to the first draft.
    llm = FakeLLM(
        responses=[LLMResponse(text="vision draft"), LLMResponse(text="gecko draft")]
    )
    result = await Ensemble(llm).debate("Q", _OPERATORS)

    assert all(d.ok for d in result.drafts)
    assert result.synthesis == "vision draft"


async def test_empty_reply_draft_marked_not_ok() -> None:
    llm = FakeLLM(responses=[LLMResponse(text="")])
    result = await Ensemble(llm).debate("Q", [("VISION", "You are VISION.")])

    assert result.drafts[0].ok is False
    assert result.contributors == []
    assert result.synthesis == ""


async def test_empty_operators_yields_empty_result() -> None:
    llm = FakeLLM(responses=[])  # never called
    result = await Ensemble(llm).debate("Q", [])

    assert result.drafts == []
    assert result.contributors == []
    assert result.synthesis == ""
