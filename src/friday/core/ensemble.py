# © Lakshya Badjatya — Author
"""Ensemble / debate: several operators draft an answer, a judge synthesizes one.

The single-operator turn loop answers in one voice. The *ensemble* path instead
asks several of FRIDAY's named operators to draft an answer to the same turn,
collects the drafts on a shared :class:`~friday.core.blackboard.Blackboard`, and
then runs ONE synthesis pass that fuses them into a single answer — recording
which operators actually contributed. It complements the model-level fan-out the
gateway already exposes (:meth:`~friday.models.gateway.ModelGateway.compare` /
:meth:`~friday.models.gateway.ModelGateway.judge`): there several *models* race
on one prompt; here several *operators* (distinct system-prompt personas) reason
on one question and a judge fuses their views.

Like the rest of ``core/`` it imports NO LLM SDK: it depends only on the
:class:`~friday.providers.llm.LLMProvider` contract (so the multi-model gateway,
a single provider, or a :class:`~friday.providers.llm.FakeLLM` all satisfy it)
and reads no configuration — the enable flag is read by ``app.py`` and the
operators are injected by the caller.

It is **bounded and non-fatal**: exactly one draft per operator plus at most one
synthesis pass. A provider error (or an empty reply) on one operator's draft
drops just that draft rather than sinking the debate; a failed or empty synthesis
falls back to the first available draft. With zero usable drafts it returns an
empty synthesis (honest) rather than inventing one.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from friday.core.blackboard import Blackboard, Draft
from friday.errors import ProviderError
from friday.providers.llm import LLMProvider, Message

logger = logging.getLogger("friday.core.ensemble")


class OperatorDraft(BaseModel):
    """One operator's draft answer in an ensemble debate.

    ``ok`` is ``False`` (with a human-readable ``error`` and empty ``content``)
    when that operator's draft could not be produced — a provider failure or an
    empty reply — so one operator going dark never sinks the whole debate.
    """

    operator: str
    content: str
    ok: bool = True
    error: str | None = None


class EnsembleResult(BaseModel):
    """The outcome of one ensemble debate.

    Carries every operator's :class:`OperatorDraft` (including failed ones, for
    transparency), the fused ``synthesis`` answer, and ``contributors`` — the
    operators whose drafts were actually available to the synthesis pass.
    """

    question: str
    drafts: list[OperatorDraft] = Field(default_factory=list)
    synthesis: str = ""
    contributors: list[str] = Field(default_factory=list)


class Ensemble:
    """Runs a bounded operator debate over an injected :class:`LLMProvider`.

    Args:
        llm: The provider used for every draft and the synthesis pass — the
            multi-model gateway in production, or any single provider / FakeLLM
            in tests. Only the abstract ``complete`` contract is depended upon.
    """

    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm

    async def debate(
        self, question: str, operators: list[tuple[str, str]]
    ) -> EnsembleResult:
        """Run one debate round across ``operators`` and synthesize one answer.

        ``operators`` is a list of ``(name, system_prompt)`` pairs — typically a
        subset of the roster, each contributing its persona's voice. Each drafts
        once (failures captured, not raised); the usable drafts are posted to a
        fresh :class:`Blackboard` and fused by :meth:`_synthesize`.
        """
        board = Blackboard()
        drafts: list[OperatorDraft] = []
        for name, system_prompt in operators:
            draft = await self._draft(name, system_prompt, question)
            drafts.append(draft)
            if draft.ok:
                board.post(draft.operator, draft.content)
        available = board.drafts()
        synthesis = await self._synthesize(question, available)
        return EnsembleResult(
            question=question,
            drafts=drafts,
            synthesis=synthesis,
            contributors=[d.operator for d in available],
        )

    async def _draft(
        self, operator: str, system_prompt: str, question: str
    ) -> OperatorDraft:
        """Produce one operator's draft, capturing (never raising) any failure."""
        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=question),
        ]
        try:
            response = await self._llm.complete(messages, None)
        except ProviderError as exc:
            logger.warning("ensemble draft by %s failed: %s", operator, exc)
            return OperatorDraft(
                operator=operator, content="", ok=False, error=str(exc)
            )
        text = (response.text or "").strip()
        if not text:
            return OperatorDraft(
                operator=operator, content="", ok=False, error="empty draft"
            )
        return OperatorDraft(operator=operator, content=text, ok=True, error=None)

    async def _synthesize(self, question: str, drafts: list[Draft]) -> str:
        """Fuse the usable drafts into one answer (one bounded, non-fatal pass).

        Zero drafts -> ``""`` (honest: nothing to fuse). One draft -> that draft
        verbatim (no model call). Otherwise one synthesis call; a provider error
        or empty reply falls back to the first draft so a debate always yields an
        answer when at least one operator spoke.
        """
        if not drafts:
            return ""
        if len(drafts) == 1:
            return drafts[0].content
        listing = "\n\n".join(f"[{d.operator}]\n{d.content}" for d in drafts)
        prompt = (
            "Several operators each drafted an answer to the same question. "
            "Synthesize ONE best answer, drawing on the strongest points across "
            "the drafts. Do not invent facts beyond what the drafts contain.\n\n"
            f"Question: {question}\n\nDrafts:\n{listing}"
        )
        messages = [Message(role="user", content=prompt)]
        try:
            response = await self._llm.complete(messages, None)
        except ProviderError as exc:
            logger.warning("ensemble synthesis failed; using first draft: %s", exc)
            return drafts[0].content
        text = (response.text or "").strip()
        return text if text else drafts[0].content
