# © Lakshya Badjatya — Author
"""Value objects for the Flow Engine: steps, flows, guards, events, and status.

These mirror the *shape* of :class:`friday.core.planner.PlanStep` / ``Plan`` and
add the runtime fields the executor needs (status, attempts, result, rationale)
plus the controls the later phases use (approval, retry, timeout, compensation, a
conditional guard). Everything is a plain pydantic model so a whole :class:`Flow`
serializes to one JSON blob the SQLite store persists as a checkpoint.

``StepGuard`` is deliberately a *restricted* predicate — ``exists|eq|ne|truthy``
over a single context key — never an arbitrary expression, so a conditional step
can never become an ``eval`` foothold (FRIDAY is fail-closed by construction).

This module imports no LLM SDK and reads no configuration.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field

StepKind = Literal["tool", "reason", "subflow"]
GuardOp = Literal["exists", "eq", "ne", "truthy"]


class FlowStatus(StrEnum):
    """Lifecycle state of a whole :class:`Flow`."""

    PLANNED = "planned"
    RUNNING = "running"
    PAUSED = "paused"
    AWAITING_APPROVAL = "awaiting_approval"
    NEEDS_CONFIRMATION = "needs_confirmation"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StepStatus(StrEnum):
    """Lifecycle state of a single :class:`FlowStep`."""

    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"
    COMPENSATED = "compensated"


class StepGuard(BaseModel):
    """A restricted condition over the context bus that gates a step.

    ``op`` is one of ``exists`` (key present), ``truthy`` (present and truthy),
    ``eq`` / ``ne`` (compares the stored value to ``value``). Evaluated by the
    pure :meth:`matches` — there is no expression language and no ``eval``, so a
    conditional step can never execute arbitrary code.
    """

    key: str
    op: GuardOp = "truthy"
    value: Any = None

    def matches(self, context: dict[str, Any]) -> bool:
        """Whether this guard holds against ``context`` (pure, side-effect free)."""
        present = self.key in context
        if self.op == "exists":
            return present
        if not present:
            # ``truthy``/``eq``/``ne`` over an absent key are all false: a guard
            # never passes on data that was never produced.
            return False
        actual = context[self.key]
        if self.op == "truthy":
            return bool(actual)
        if self.op == "eq":
            return bool(actual == self.value)
        # op == "ne"
        return bool(actual != self.value)


class FlowStep(BaseModel):
    """One node in a :class:`Flow`'s dependency graph, with runtime state.

    The plan half (``id``/``description``/``operator``/``tool``/``args``/
    ``depends_on``/``side_effecting``) mirrors :class:`~friday.core.planner.PlanStep`.
    The rest is execution state and controls: ``kind`` selects the executor path
    (a ``tool`` step dispatches through the broker, a ``reason`` step calls the
    LLM, a ``subflow`` step runs a nested flow); ``requires_approval`` pauses the
    flow before the step runs; ``retry``/``timeout_s``/``compensation`` drive the
    saga behaviour; ``when`` conditionally skips the step; ``status``/``attempts``/
    ``result``/``rationale`` record what happened (the last feeding the
    explainable trace).
    """

    # -- plan shape (mirrors PlanStep) --
    id: str
    description: str
    operator: str = "FRIDAY"
    tool: str | None = None
    args: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)
    side_effecting: bool = False
    # -- executor controls --
    kind: StepKind = "reason"
    requires_approval: bool = False
    retry: int = 0
    timeout_s: float | None = None
    compensation: str | None = None
    when: StepGuard | None = None
    # -- runtime state --
    status: StepStatus = StepStatus.PENDING
    attempts: int = 0
    result: dict[str, Any] = Field(default_factory=dict)
    rationale: str = ""


class Flow(BaseModel):
    """A goal decomposed into steps, plus the runtime state of one execution.

    ``context`` is the shared blackboard steps publish into and read from;
    ``cursor`` is the checkpoint position (steps before it are settled);
    ``budget_tokens``/``spent_tokens`` drive the per-flow cost governor;
    ``template``/``parent_flow_id`` support templates and sub-flow nesting.
    """

    id: str
    goal: str
    steps: list[FlowStep] = Field(default_factory=list)
    status: FlowStatus = FlowStatus.PLANNED
    context: dict[str, Any] = Field(default_factory=dict)
    cursor: int = 0
    budget_tokens: int | None = None
    spent_tokens: int = 0
    created_at: str | None = None
    updated_at: str | None = None
    template: str | None = None
    parent_flow_id: str | None = None
    #: Ids of steps the owner has explicitly approved (clears their HITL gate).
    approvals: list[str] = Field(default_factory=list)


class FlowEvent(BaseModel):
    """One audited transition in a flow's life (also streamed to the HUD).

    ``kind`` is a controlled token (e.g. ``planned``, ``step_ok``, ``step_failed``,
    ``paused``, ``resumed``, ``succeeded``, ``cancelled``). ``detail`` carries a
    small, JSON-safe payload for the trace. Appended to the hash-chained audit
    ledger so a flow's history is tamper-evident.
    """

    flow_id: str
    step_id: str | None = None
    kind: str
    detail: dict[str, Any] = Field(default_factory=dict)
