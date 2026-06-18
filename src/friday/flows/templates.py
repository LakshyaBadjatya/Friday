# © Lakshya Badjatya — Author
"""Reusable flow templates + their in-memory store.

A :class:`FlowTemplate` is a named, parameterized step graph. :meth:`FlowTemplateStore.instantiate`
stamps out a fresh :class:`~friday.flows.models.Flow` from one — resetting every
step's runtime state and substituting ``{param}`` placeholders in the goal — so a
common workflow ("morning briefing", "research-then-notify") is defined once and
launched many times, and a ``subflow`` step can run one nested inside another.

The store is process-local and dependency-free (no SQLite needed for Phase 4);
the engine reads it to resolve ``subflow`` steps and the routes expose save/list/
instantiate.
"""

from __future__ import annotations

import uuid

from pydantic import BaseModel, Field

from friday.flows.models import Flow, FlowStep, StepStatus


class FlowTemplate(BaseModel):
    """A named, parameterized step graph a :class:`Flow` is stamped from."""

    name: str
    goal: str
    steps: list[FlowStep] = Field(default_factory=list)


class FlowTemplateStore:
    """A process-local registry of :class:`FlowTemplate`s."""

    def __init__(self) -> None:
        self._templates: dict[str, FlowTemplate] = {}

    def save(self, template: FlowTemplate) -> FlowTemplate:
        """Register (or replace) ``template`` by name."""
        self._templates[template.name] = template
        return template

    def get(self, name: str) -> FlowTemplate | None:
        """Return the template registered under ``name`` (or ``None``)."""
        return self._templates.get(name)

    def list(self) -> list[FlowTemplate]:
        """Every registered template."""
        return list(self._templates.values())

    def instantiate(
        self, name: str, params: dict[str, str] | None = None
    ) -> Flow | None:
        """Build a fresh ``planned`` :class:`Flow` from template ``name``.

        Steps are deep-copied with their runtime state reset; ``{param}``
        placeholders in the goal are substituted from ``params`` (an unknown
        placeholder is left intact rather than raising). ``None`` when no such
        template exists.
        """
        template = self._templates.get(name)
        if template is None:
            return None
        goal = self._format(template.goal, params or {})
        steps: list[FlowStep] = []
        for original in template.steps:
            step = original.model_copy(deep=True)
            step.status = StepStatus.PENDING
            step.attempts = 0
            step.result = {}
            step.rationale = ""
            steps.append(step)
        return Flow(id=uuid.uuid4().hex, goal=goal, steps=steps, template=name)

    @staticmethod
    def _format(goal: str, params: dict[str, str]) -> str:
        """Substitute ``{key}`` placeholders, leaving unknown ones intact."""
        for key, value in params.items():
            goal = goal.replace("{" + key + "}", value)
        return goal
