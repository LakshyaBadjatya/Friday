"""Unit tests for the ensemble solver.

``_ScriptedLLM`` implements the *real* :class:`~friday.providers.llm.LLMProvider`
contract — ``complete(messages: list[Message], tools=None, *, model=None) ->
LLMResponse`` — not the bare prompt-string shape an earlier draft of this plan
assumed. See ``src/friday/providers/llm.py`` for the ABC it mirrors.
"""

from __future__ import annotations

import pytest

from friday.providers.llm import LLMProvider, LLMResponse, Message, ToolSpec
from friday.vault.models import VerificationStatus
from friday.vault.solver import Solver, normalise_answer, verify_with_sympy


def test_normalise_answer_strips_units_and_formatting() -> None:
    assert normalise_answer("**9 V**") == "9 v"
    assert normalise_answer("9.0 V") == "9 v"
    assert normalise_answer(" x = 9 ") == "x = 9"


def test_normalise_answer_does_not_mangle_non_trailing_decimals() -> None:
    """Only a genuinely *trailing* run of zeros collapses — not digits that
    merely follow a zero, and not a decimal that already ends in a non-zero."""
    assert normalise_answer("10.50") == "10.50"
    assert normalise_answer("0.0") == "0"
    assert normalise_answer("1.05") == "1.05"


def test_sympy_confirms_correct_algebra() -> None:
    result = verify_with_sympy("2*x + 4 = 10", "x = 3")
    assert result.ok is True
    assert result.status is VerificationStatus.VERIFIED


def test_sympy_catches_a_confident_wrong_answer() -> None:
    result = verify_with_sympy("2*x + 4 = 10", "x = 5")
    assert result.ok is False
    assert result.status is VerificationStatus.REFUTED
    assert "3" in result.detail


def test_sympy_declines_gracefully_on_prose() -> None:
    result = verify_with_sympy("explain photosynthesis", "it makes sugar")
    assert result.ok is False
    assert result.status is VerificationStatus.NOT_VERIFIABLE
    assert "not verifiable" in result.detail


def test_sympy_accepts_implicit_multiplication_and_caret() -> None:
    """A vision model has no reason to prefer ``2*x`` over ``2x``, or ``**``
    over ``^`` — both need to parse, not just the canonical SymPy spelling."""
    assert verify_with_sympy("2x + 4 = 10", "x = 3").ok is True
    assert verify_with_sympy("x^2 = 4", "x = 2").ok is True


def test_sympy_accepts_either_root_of_a_quadratic() -> None:
    plus = verify_with_sympy("x^2 = 4", "x = 2")
    minus = verify_with_sympy("x^2 = 4", "x = -2")
    wrong = verify_with_sympy("x^2 = 4", "x = 3")
    assert plus.ok is True
    assert minus.ok is True
    assert wrong.ok is False
    assert "-2" in wrong.detail and "2" in wrong.detail


def test_sympy_compares_answer_with_or_without_the_variable_name() -> None:
    assert verify_with_sympy("2*x + 4 = 10", "3").ok is True
    assert verify_with_sympy("2*x + 4 = 10", "x = 3").ok is True


def test_sympy_declines_a_multivariable_statement() -> None:
    result = verify_with_sympy("2*x + y = 10", "x = 3")
    assert result.ok is False
    assert "not verifiable" in result.detail


def test_sympy_declines_a_word_problem_wrapped_around_a_real_equation() -> None:
    """The central limitation: statement must be a BARE equation. Framing
    around an otherwise-valid equation ("Solve for x: ...", "..., find x.")
    still fails to parse, because the extra words are not valid SymPy syntax.
    This is the evidence behind the module-level honesty note — on realistic
    OCR of a word problem, this check declines far more often than it rules."""
    wrapped_prefix = verify_with_sympy("Solve for x: 2x + 4 = 10", "x = 3")
    wrapped_suffix = verify_with_sympy("If 2x + 4 = 10, find x.", "x = 3")
    prose_word_problem = verify_with_sympy(
        "A cell of emf 4 V and internal resistance 1 ohm is connected across "
        "a 2 ohm resistor. Find the terminal voltage.",
        "8 V",
    )
    assert wrapped_prefix.ok is False
    assert wrapped_suffix.ok is False
    assert prose_word_problem.ok is False
    assert "not verifiable" in prose_word_problem.detail


class _ScriptedLLM(LLMProvider):
    """Returns a queued :class:`LLMResponse` per call and records the prompts."""

    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.prompts: list[str] = []

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        *,
        model: str | None = None,
    ) -> LLMResponse:
        self.prompts.append(messages[-1].content or "" if messages else "")
        text = self._replies.pop(0) if self._replies else ""
        return LLMResponse(text=text)


_UNANIMOUS = [
    '{"subject": "physics", "statement": "find V", "steps": ["I = 1 A"], '
    '"final_answer": "9 V", "confidence": 0.9}'
] * 3


@pytest.mark.asyncio
async def test_unanimous_panel_needs_no_judge() -> None:
    llm = _ScriptedLLM(list(_UNANIMOUS))
    solve = await Solver(llm, operators=["VISION", "ORACLE", "GECKO"]).solve(
        item_ids=["i1"], ocr_text="a cell of emf 4 V"
    )
    assert solve.consensus.final_answer == "9 V"
    assert solve.consensus.agreement == "3/3"
    assert solve.consensus.judged is False
    assert solve.dissent == []
    assert len(llm.prompts) == 3


@pytest.mark.asyncio
async def test_disagreement_escalates_to_a_judge_and_records_dissent() -> None:
    llm = _ScriptedLLM(
        [
            '{"final_answer": "9 V", "confidence": 0.9, "steps": []}',
            '{"final_answer": "9 V", "confidence": 0.8, "steps": []}',
            '{"final_answer": "6 V", "confidence": 0.7, "steps": []}',
            '{"final_answer": "9 V", "steps": ["the third dropped the internal drop"]}',
        ]
    )
    solve = await Solver(llm, operators=["VISION", "ORACLE", "GECKO"]).solve(
        item_ids=["i1"], ocr_text="x"
    )
    assert solve.consensus.final_answer == "9 V"
    assert solve.consensus.judged is True
    assert any("6 V" in d for d in solve.dissent)
    assert len(llm.prompts) == 4


@pytest.mark.asyncio
async def test_one_dead_operator_does_not_sink_the_solve() -> None:
    class _Flaky(_ScriptedLLM):
        async def complete(
            self,
            messages: list[Message],
            tools: list[ToolSpec] | None = None,
            *,
            model: str | None = None,
        ) -> LLMResponse:
            if not self.prompts:
                self.prompts.append(messages[-1].content or "" if messages else "")
                raise RuntimeError("provider exploded")
            return await super().complete(messages, tools, model=model)

    solve = await Solver(
        _Flaky(list(_UNANIMOUS)), operators=["VISION", "ORACLE", "GECKO"]
    ).solve(item_ids=["i1"], ocr_text="x")
    assert solve.consensus.final_answer == "9 V"
    assert len(solve.drafts) == 2


@pytest.mark.asyncio
async def test_a_provider_that_raises_every_time_yields_an_honest_empty_solve() -> None:
    class _AlwaysDead(_ScriptedLLM):
        async def complete(
            self,
            messages: list[Message],
            tools: list[ToolSpec] | None = None,
            *,
            model: str | None = None,
        ) -> LLMResponse:
            raise RuntimeError("provider exploded")

    solve = await Solver(_AlwaysDead([]), operators=["VISION", "ORACLE", "GECKO"]).solve(
        item_ids=["i1"], ocr_text="x"
    )
    assert solve.drafts == []
    assert solve.consensus.agreement == "0/0"
    assert solve.verification.ok is False


@pytest.mark.asyncio
async def test_empty_operator_list_yields_an_honest_empty_solve() -> None:
    llm = _ScriptedLLM(list(_UNANIMOUS))
    solve = await Solver(llm, operators=[]).solve(item_ids=["i1"], ocr_text="x")
    assert solve.drafts == []
    assert solve.consensus.agreement == "0/0"
    assert llm.prompts == []


@pytest.mark.asyncio
async def test_a_three_way_disagreement_still_picks_a_deterministic_winner() -> None:
    """When every operator disagrees, ``Counter.most_common`` breaks the tie by
    first-seen order — so the first operator's answer wins the plurality (and a
    judge is still asked, since 1/3 is not everyone agreeing)."""
    llm = _ScriptedLLM(
        [
            '{"final_answer": "9 V", "confidence": 0.9, "steps": []}',
            '{"final_answer": "7 V", "confidence": 0.9, "steps": []}',
            '{"final_answer": "6 V", "confidence": 0.9, "steps": []}',
            '{"final_answer": "9 V", "steps": ["judge picks the first"]}',
        ]
    )
    solve = await Solver(llm, operators=["VISION", "ORACLE", "GECKO"]).solve(
        item_ids=["i1"], ocr_text="x"
    )
    assert solve.consensus.agreement == "1/3"
    assert solve.consensus.judged is True
    assert len(solve.dissent) == 2


@pytest.mark.asyncio
async def test_a_non_numeric_confidence_does_not_crash_the_panel() -> None:
    """A draft whose ``confidence`` is a string (or missing) must be dropped
    into a Draft with confidence 0.0, not raise and take the whole solve down."""
    llm = _ScriptedLLM(
        [
            '{"final_answer": "9 V", "confidence": "very high", "steps": []}',
            '{"final_answer": "9 V", "steps": []}',
            '{"final_answer": "9 V", "confidence": null, "steps": []}',
        ]
    )
    solve = await Solver(llm, operators=["VISION", "ORACLE", "GECKO"]).solve(
        item_ids=["i1"], ocr_text="x"
    )
    assert len(solve.drafts) == 3
    assert all(d.confidence == 0.0 for d in solve.drafts)
    assert solve.consensus.final_answer == "9 V"


@pytest.mark.asyncio
async def test_solve_carries_a_required_created_at_timestamp() -> None:
    llm = _ScriptedLLM(list(_UNANIMOUS))
    solve = await Solver(llm, operators=["VISION", "ORACLE", "GECKO"]).solve(
        item_ids=["i1"], ocr_text="x"
    )
    assert solve.created_at != ""


# --- Change 2: verification runs against the chosen draft's own equation ---


@pytest.mark.asyncio
async def test_a_good_equation_with_a_right_answer_is_verified() -> None:
    """The chosen draft's equation, not the raw OCR statement, is what gets
    checked — this is what lets a raw word problem (no bare equation
    anywhere in the OCR text) verify at all."""
    llm = _ScriptedLLM(
        [
            '{"final_answer": "x = 3", "equation": "2*x + 4 = 10", "steps": []}',
            '{"final_answer": "x = 3", "equation": "2*x + 4 = 10", "steps": []}',
            '{"final_answer": "x = 3", "equation": "2*x + 4 = 10", "steps": []}',
        ]
    )
    solve = await Solver(llm, operators=["VISION", "ORACLE", "GECKO"]).solve(
        item_ids=["i1"],
        ocr_text="Solve for x: 2x + 4 = 10, in volts.",
    )
    assert solve.verification.status is VerificationStatus.VERIFIED
    assert solve.verification.ok is True


@pytest.mark.asyncio
async def test_a_good_equation_with_a_wrong_answer_is_refuted() -> None:
    """The case that justifies the whole feature: a model that sets up the
    equation correctly but then slips on the arithmetic must be caught, not
    waved through as 'not verifiable' the way raw-OCR verification would."""
    llm = _ScriptedLLM(
        [
            '{"final_answer": "x = 5", "equation": "2*x + 4 = 10", "steps": []}',
            '{"final_answer": "x = 5", "equation": "2*x + 4 = 10", "steps": []}',
            '{"final_answer": "x = 5", "equation": "2*x + 4 = 10", "steps": []}',
        ]
    )
    solve = await Solver(llm, operators=["VISION", "ORACLE", "GECKO"]).solve(
        item_ids=["i1"],
        ocr_text="Solve for x: 2x + 4 = 10, in volts.",
    )
    assert solve.verification.status is VerificationStatus.REFUTED
    assert solve.verification.ok is False
    assert "3" in solve.verification.detail


@pytest.mark.asyncio
async def test_an_empty_equation_falls_back_to_the_raw_statement() -> None:
    """A chosen draft that (honestly) supplies no equation — chemistry naming,
    prose, and so on — falls back to attempting the raw OCR statement, same
    as before this feature existed, and reports NOT_VERIFIABLE when that also
    has nothing to parse."""
    llm = _ScriptedLLM(
        [
            '{"final_answer": "it makes sugar", "equation": "", "steps": []}',
            '{"final_answer": "it makes sugar", "equation": "", "steps": []}',
            '{"final_answer": "it makes sugar", "equation": "", "steps": []}',
        ]
    )
    solve = await Solver(llm, operators=["VISION", "ORACLE", "GECKO"]).solve(
        item_ids=["i1"], ocr_text="explain photosynthesis"
    )
    assert solve.verification.status is VerificationStatus.NOT_VERIFIABLE
    assert solve.verification.ok is False


@pytest.mark.asyncio
async def test_a_malformed_equation_is_not_verifiable_and_never_crashes() -> None:
    llm = _ScriptedLLM(
        [
            '{"final_answer": "x = 3", "equation": "2*x + ) = (( 10", "steps": []}',
            '{"final_answer": "x = 3", "equation": "2*x + ) = (( 10", "steps": []}',
            '{"final_answer": "x = 3", "equation": "2*x + ) = (( 10", "steps": []}',
        ]
    )
    solve = await Solver(llm, operators=["VISION", "ORACLE", "GECKO"]).solve(
        item_ids=["i1"], ocr_text="whatever the OCR saw"
    )
    assert solve.verification.status is VerificationStatus.NOT_VERIFIABLE


@pytest.mark.asyncio
async def test_an_equation_in_a_different_variable_than_the_answer_still_verifies() -> None:
    """The operator may name the equation's variable differently from how it
    phrases the final answer ("y" in the equation, "x = 3" in the answer) —
    verification compares by value, not by symbol name, same as
    ``verify_with_sympy`` already did for the raw-statement path."""
    llm = _ScriptedLLM(
        [
            '{"final_answer": "x = 3", "equation": "2*y + 4 = 10", "steps": []}',
            '{"final_answer": "x = 3", "equation": "2*y + 4 = 10", "steps": []}',
            '{"final_answer": "x = 3", "equation": "2*y + 4 = 10", "steps": []}',
        ]
    )
    solve = await Solver(llm, operators=["VISION", "ORACLE", "GECKO"]).solve(
        item_ids=["i1"], ocr_text="whatever the OCR saw"
    )
    assert solve.verification.status is VerificationStatus.VERIFIED
