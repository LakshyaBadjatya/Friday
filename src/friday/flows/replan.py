# © Lakshya Badjatya — Author
"""Adaptive re-planning: turn a failed step into a revised remainder-plan.

When a flow step fails, the :class:`~friday.flows.engine.FlowEngine` asks a
:class:`Replanner` for the steps that would still reach the goal given what has
already happened. This wraps the existing :class:`~friday.core.planner.Planner`
(no new model contract): it frames the original goal plus a short progress
summary and the failure, asks for a fresh decomposition, and returns those
:class:`~friday.core.planner.PlanStep`s for the engine to splice in.

Non-fatal by inheritance: ``Planner.decompose`` already degrades to a single-step
plan on any provider/parse failure, so :meth:`Replanner.replan` always returns a
usable list (possibly a single restated step) rather than raising — and the
engine treats an unhelpful re-plan as "stop and report", never as fabricated
success.

Imports no LLM SDK — only the planner, which itself depends on the provider
contract.
"""

from __future__ import annotations

from friday.core.planner import Planner, PlanStep


class Replanner:
    """Produce a revised remainder-plan from a failure, via the injected planner.

    Args:
        planner: The decomposition planner re-used to generate recovery steps.
    """

    def __init__(self, planner: Planner) -> None:
        self._planner = planner

    async def replan(
        self, goal: str, completed: str, failure: str
    ) -> list[PlanStep]:
        """Return recovery steps toward ``goal`` given progress + the failure.

        ``completed`` is a short summary of settled steps; ``failure`` is the
        description of the step that failed. The result is the planner's fresh
        decomposition of "continue toward the goal" — empty only if the planner
        itself yields nothing (the engine then reports the flow failed).
        """
        prompt = (
            f"Goal: {goal}\n"
            f"Progress so far: {completed or 'none'}\n"
            f"The step '{failure}' failed. Produce the steps still needed to "
            f"reach the goal, accounting for that failure."
        )
        plan = await self._planner.decompose(prompt)
        return list(plan.steps)
