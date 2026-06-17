# © Lakshya Badjatya — Author
"""Unit tests for the offline prompt-eval harness + its CLI registration."""

from __future__ import annotations

from friday.cli import _handle_eval, build_parser
from friday.eval.harness import CaseResult, EvalCase, EvalReport, run_eval
from friday.providers.llm import FakeLLM, LLMResponse, Usage


def _resp(text: str) -> LLMResponse:
    return LLMResponse(text=text, tool_calls=[], usage=Usage())


async def test_expect_substrings_pass_case_insensitively() -> None:
    llm = FakeLLM(responses=[_resp("The capital is Paris, Boss.")])
    cases = [EvalCase(name="cap", prompt="capital of France?", expect=["paris"])]
    report = await run_eval(cases, llm)
    assert report.passed == 1
    assert report.total == 1
    assert report.pass_rate == 1.0
    assert report.results[0].missing == []


async def test_missing_expectation_fails_and_is_reported() -> None:
    llm = FakeLLM(responses=[_resp("I'm not sure, Boss.")])
    cases = [EvalCase(name="cap", prompt="capital of France?", expect=["Paris"])]
    report = await run_eval(cases, llm)
    assert report.passed == 0
    assert report.results[0].missing == ["Paris"]
    assert report.results[0].passed is False


async def test_forbidden_substring_fails() -> None:
    llm = FakeLLM(responses=[_resp("As an AI language model, I cannot help.")])
    cases = [EvalCase(name="persona", prompt="hi", forbid=["as an ai language model"])]
    report = await run_eval(cases, llm)
    assert report.passed == 0
    assert report.results[0].forbidden_hit == ["as an ai language model"]


async def test_provider_error_scores_case_failed_not_crash() -> None:
    llm = FakeLLM(responses=[])  # exhausted -> ProviderError on first call
    cases = [EvalCase(name="x", prompt="hi", expect=["anything"])]
    report = await run_eval(cases, llm)
    assert report.passed == 0
    assert report.results[0].response is None
    assert report.results[0].missing == ["anything"]


async def test_pass_rate_and_ok_threshold() -> None:
    llm = FakeLLM(responses=[_resp("yes"), _resp("no")])
    cases = [
        EvalCase(name="a", prompt="q1", expect=["yes"]),
        EvalCase(name="b", prompt="q2", expect=["yes"]),  # reply is "no" -> fail
    ]
    report = await run_eval(cases, llm)
    assert report.pass_rate == 0.5
    assert report.ok(0.5) is True
    assert report.ok(1.0) is False


def test_empty_report_passes_trivially() -> None:
    report = EvalReport()
    assert report.pass_rate == 1.0
    assert report.ok() is True


def test_render_lists_cases_and_summary() -> None:
    report = EvalReport(results=[CaseResult(name="a", passed=True, response="ok")])
    rendered = report.render()
    assert "[PASS] a" in rendered
    assert "1/1 passed" in rendered


def test_cli_registers_eval_subcommand() -> None:
    args = build_parser().parse_args(["eval", "cases.json", "--min-pass-rate", "0.8"])
    assert args.func is _handle_eval
    assert args.cases == "cases.json"
    assert args.min_pass_rate == 0.8
