# © Lakshya Badjatya — Author
"""A tiny, offline prompt-eval harness — regression scoring without a backend.

Runs a list of :class:`EvalCase` (a prompt plus simple substring expectations)
through any :class:`~friday.providers.llm.LLMProvider` and scores each
pass/fail, yielding an :class:`EvalReport` with a pass-rate the ``friday eval``
CLI can gate CI on. Checks are intentionally simple and deterministic — every
``expect`` substring must appear (case-insensitive) and no ``forbid`` substring
may — so a run is reproducible and needs no judge model. Against the offline
:class:`~friday.providers.llm.FakeLLM` the whole thing runs with zero network.

The harness is *pure over the injected provider*: it imports no LLM SDK and opens
no socket itself; a provider error on one case scores that case failed (the run
never crashes) so a flaky backend yields a low score, not a stack trace.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, Field

from friday.errors import FridayError
from friday.providers.llm import LLMProvider, Message


class EvalCase(BaseModel):
    """One eval: a ``prompt`` plus substring expectations on the reply.

    ``expect`` substrings must *all* appear in the reply (case-insensitive);
    ``forbid`` substrings must *none* appear. A case with neither always passes
    (a smoke check that the model merely replied without error).
    """

    name: str
    prompt: str
    expect: list[str] = Field(default_factory=list)
    forbid: list[str] = Field(default_factory=list)


class CaseResult(BaseModel):
    """The outcome of one :class:`EvalCase`.

    ``missing`` lists expected substrings that were absent; ``forbidden_hit``
    lists forbidden substrings that appeared. ``passed`` is true iff both are
    empty. ``response`` is the model's reply (``None`` if it errored / came back
    empty).
    """

    name: str
    passed: bool
    response: str | None
    missing: list[str] = Field(default_factory=list)
    forbidden_hit: list[str] = Field(default_factory=list)


class EvalReport(BaseModel):
    """The full set of case results with pass-rate accounting."""

    results: list[CaseResult] = Field(default_factory=list)

    @property
    def total(self) -> int:
        """Number of cases run."""
        return len(self.results)

    @property
    def passed(self) -> int:
        """Number of cases that passed."""
        return sum(1 for r in self.results if r.passed)

    @property
    def pass_rate(self) -> float:
        """Fraction of cases that passed (``1.0`` for an empty run)."""
        return self.passed / self.total if self.total else 1.0

    def ok(self, min_pass_rate: float = 1.0) -> bool:
        """Whether the pass-rate meets ``min_pass_rate`` (default: all must pass)."""
        return self.pass_rate >= min_pass_rate

    def render(self) -> str:
        """A terminal-friendly rendering: one line per case + a summary."""
        lines = []
        for r in self.results:
            tag = "PASS" if r.passed else "FAIL"
            detail = ""
            if r.missing:
                detail += f" missing={r.missing}"
            if r.forbidden_hit:
                detail += f" forbidden={r.forbidden_hit}"
            lines.append(f"[{tag}] {r.name}{detail}")
        pct = round(self.pass_rate * 100, 1)
        lines.append("")
        lines.append(f"eval: {self.passed}/{self.total} passed ({pct}%)")
        return "\n".join(lines)


def _score(case: EvalCase, text: str | None) -> CaseResult:
    """Score one case's reply against its expect/forbid substring sets."""
    haystack = (text or "").lower()
    missing = [s for s in case.expect if s.lower() not in haystack]
    forbidden = [s for s in case.forbid if s.lower() in haystack]
    return CaseResult(
        name=case.name,
        passed=not missing and not forbidden,
        response=text,
        missing=missing,
        forbidden_hit=forbidden,
    )


async def run_eval(cases: Sequence[EvalCase], llm: LLMProvider) -> EvalReport:
    """Run every case through ``llm`` and return the scored :class:`EvalReport`.

    Each case is one ``complete`` call with the prompt as a single user message.
    A provider failure scores that case failed (with all ``expect`` substrings
    marked missing) rather than aborting the run, so one bad backend response
    can't sink the whole report.
    """
    results: list[CaseResult] = []
    for case in cases:
        try:
            response = await llm.complete(
                [Message(role="user", content=case.prompt)], None
            )
        except FridayError:
            results.append(
                CaseResult(
                    name=case.name,
                    passed=False,
                    response=None,
                    missing=list(case.expect),
                    forbidden_hit=[],
                )
            )
            continue
        results.append(_score(case, response.text))
    return EvalReport(results=results)
