# © Lakshya Badjatya — Author
"""Task decomposition: turn a multi-step goal into a DAG of plannable steps.

A single turn answers one ask. A *plan* breaks a larger goal into an ordered
graph of steps — each owned by a roster operator and (optionally) backed by a
tool — so FRIDAY can show the owner what it intends to do before doing any of it,
then execute the steps in dependency order with every side effect still passing
through the broker's confirm/permission/audit gates.

This module is the **decomposition + shape** half of that feature: the
:class:`PlanStep` / :class:`Plan` value objects, an LLM-backed
:meth:`Planner.decompose` that asks the model for a step graph, and a pure
:meth:`Plan.topological_order` (Kahn's algorithm, with cycle and missing-dependency
detection) plus a deterministic :meth:`Plan.render` for the "show the plan"
surface. *Execution* of the ordered steps through the broker is wired separately
by the caller, so this module — like the rest of ``core/`` — imports NO LLM SDK
(only the :class:`~friday.providers.llm.LLMProvider` contract) and reads no
configuration.

:meth:`Planner.decompose` is **non-fatal**: a provider error or an unparseable /
empty model reply degrades to a single-step plan that simply restates the goal,
so the planner always yields a usable :class:`Plan` rather than raising.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import BaseModel, Field

from friday.errors import ProviderError
from friday.providers.llm import LLMProvider, Message

logger = logging.getLogger("friday.core.planner")


class PlanStep(BaseModel):
    """One node in a :class:`Plan`'s dependency graph.

    Attributes:
        id: A short, plan-unique identifier (e.g. ``"s1"``) other steps reference
            in their ``depends_on``.
        description: A human-readable statement of what the step does.
        operator: The roster code-name that owns the step (e.g. ``"VISION"``);
            the prime ``"FRIDAY"`` when unspecified.
        tool: The registry tool (or capability token) the step invokes, or
            ``None`` for a reasoning-only step.
        args: Arguments for ``tool`` (empty when none / not a tool step).
        depends_on: Ids of steps that must complete before this one may run.
        side_effecting: Whether the step performs a real-world action — the
            executor confirm-gates these through the broker.
    """

    id: str
    description: str
    operator: str = "FRIDAY"
    tool: str | None = None
    args: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)
    side_effecting: bool = False


class Plan(BaseModel):
    """A goal decomposed into a DAG of :class:`PlanStep`s."""

    goal: str
    steps: list[PlanStep] = Field(default_factory=list)

    def topological_order(self) -> list[PlanStep]:
        """Return the steps in a valid dependency order (Kahn's algorithm).

        Ties are broken by the steps' declared order, so the ordering is
        deterministic. Raises :class:`ValueError` if a step references an unknown
        dependency or if the graph contains a cycle (so an unrunnable plan fails
        loudly rather than executing a partial, ill-defined order).
        """
        by_id: dict[str, PlanStep] = {s.id: s for s in self.steps}
        if len(by_id) != len(self.steps):
            raise ValueError("plan has duplicate step ids")
        indegree: dict[str, int] = {s.id: 0 for s in self.steps}
        for step in self.steps:
            # Dedupe per step: the decrement loop below tests membership
            # (``current in step.depends_on``) once per processed id, so a
            # duplicated dependency must only count once here or the step's
            # indegree could never reach zero (a spurious "cycle").
            for dep in set(step.depends_on):
                if dep not in by_id:
                    raise ValueError(
                        f"step {step.id!r} depends on unknown step {dep!r}"
                    )
                indegree[step.id] += 1
        # Seed with the zero-indegree steps in declared order; process likewise.
        ready = [s.id for s in self.steps if indegree[s.id] == 0]
        ordered: list[PlanStep] = []
        while ready:
            current = ready.pop(0)
            ordered.append(by_id[current])
            for step in self.steps:
                if current in step.depends_on:
                    indegree[step.id] -= 1
                    if indegree[step.id] == 0:
                        ready.append(step.id)
        if len(ordered) != len(self.steps):
            raise ValueError("plan dependency graph contains a cycle")
        return ordered

    def render(self) -> str:
        """A deterministic, human-readable rendering of the plan for confirmation.

        Lists the goal then each step in dependency order, marking side-effecting
        steps and noting dependencies, so the owner can read exactly what will run
        before authorizing execution. No model call — always exact.
        """
        lines = [f"Plan for: {self.goal}"]
        for index, step in enumerate(self.topological_order(), start=1):
            mark = " [real-world action]" if step.side_effecting else ""
            tool = f" via {step.tool}" if step.tool else ""
            deps = (
                f" (after {', '.join(step.depends_on)})" if step.depends_on else ""
            )
            lines.append(
                f"{index}. [{step.operator}] {step.description}{tool}{deps}{mark}"
            )
        return "\n".join(lines)


_DECOMPOSE_INSTRUCTIONS = (
    "Break the goal into an ordered list of small steps. Reply with ONLY a JSON "
    "array; each element is an object with keys: id (short string), description "
    "(string), operator (one of FRIDAY, EDITH, ORACLE, GECKO, KAREN, VERONICA, "
    "JOCASTA, VISION, FORGE), tool (string or null), depends_on (array of step "
    "ids), side_effecting (boolean). Use depends_on to express ordering."
)


class Planner:
    """Decomposes a goal into a :class:`Plan` via an injected LLM (non-fatal).

    Args:
        llm: The provider used for the one decomposition call. Only the abstract
            ``complete`` contract is depended upon (gateway / provider / FakeLLM).
    """

    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm

    async def decompose(self, goal: str) -> Plan:
        """Ask the model to decompose ``goal``; degrade to a single step on any failure.

        Makes exactly one model call. A provider error, an empty reply, or a
        payload that does not parse into at least one valid step all fall back to
        a single-step plan restating the goal — so the caller always receives a
        usable :class:`Plan`.
        """
        messages = [
            Message(role="system", content=_DECOMPOSE_INSTRUCTIONS),
            Message(role="user", content=f"Goal: {goal}"),
        ]
        try:
            response = await self._llm.complete(messages, None)
        except ProviderError as exc:
            logger.warning("planner decompose failed; single-step fallback: %s", exc)
            return self._fallback(goal)
        steps = self._parse_steps(response.text)
        if not steps:
            return self._fallback(goal)
        plan = Plan(goal=goal, steps=steps)
        # Per-step pydantic validation does not catch dependency-graph defects
        # (unknown deps, cycles, duplicate ids). Probe the assembled graph so a
        # malformed LLM decomposition degrades to the single-step fallback rather
        # than returning a Plan whose render()/topological_order() raises in the
        # caller — honoring decompose()'s "always yields a usable Plan" contract.
        try:
            plan.topological_order()
        except ValueError as exc:
            logger.warning("planner produced an unrunnable graph; single-step fallback: %s", exc)
            return self._fallback(goal)
        return plan

    @staticmethod
    def _fallback(goal: str) -> Plan:
        """A one-step plan that simply restates the goal (honest degradation)."""
        return Plan(
            goal=goal,
            steps=[PlanStep(id="s1", description=goal, operator="FRIDAY")],
        )

    @staticmethod
    def _parse_steps(text: str | None) -> list[PlanStep]:
        """Parse a JSON step array out of ``text``; ``[]`` on any malformed input.

        Tolerates a leading/trailing code fence or prose by extracting the first
        ``[...]`` span. Each element that validates as a :class:`PlanStep` is kept;
        a wholly unparseable payload yields ``[]`` (the caller then falls back).
        """
        if not text:
            return []
        start = text.find("[")
        end = text.rfind("]")
        if start == -1 or end == -1 or end < start:
            return []
        try:
            raw = json.loads(text[start : end + 1])
        except (json.JSONDecodeError, ValueError):
            return []
        if not isinstance(raw, list):
            return []
        steps: list[PlanStep] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            try:
                steps.append(PlanStep.model_validate(item))
            except ValueError:
                continue
        return steps
