# © Lakshya Badjatya — Author
"""Unit tests for the Flow Engine value objects + the ``enable_flows`` settings.

Covers the default-off flag (the core boots unchanged) and the model contract the
store/engine depend on: step defaults, a JSON round-trip, and the restricted
``StepGuard`` matcher (``exists|eq|ne|truthy`` over the context bus — never
``eval``).
"""

from __future__ import annotations

from friday.config import Settings
from friday.flows.models import (
    Flow,
    FlowEvent,
    FlowStatus,
    FlowStep,
    StepGuard,
    StepStatus,
)


def test_flows_disabled_by_default() -> None:
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.enable_flows is False
    assert s.flow_max_steps == 50
    assert s.flow_max_replans == 3
    assert s.flow_step_timeout_s == 120.0
    assert s.flow_default_budget_tokens == 200_000


def test_flowstep_defaults() -> None:
    step = FlowStep(id="s1", description="do a thing")
    assert step.operator == "FRIDAY"
    assert step.kind == "reason"
    assert step.requires_approval is False
    assert step.retry == 0
    assert step.status == StepStatus.PENDING
    assert step.attempts == 0
    assert step.depends_on == []
    assert step.when is None


def test_flow_defaults_and_roundtrip() -> None:
    flow = Flow(id="f1", goal="ship it", steps=[FlowStep(id="s1", description="x")])
    assert flow.status == FlowStatus.PLANNED
    assert flow.cursor == 0
    assert flow.spent_tokens == 0
    again = Flow.model_validate_json(flow.model_dump_json())
    assert again.goal == "ship it"
    assert again.status == FlowStatus.PLANNED
    assert again.steps[0].id == "s1"


def test_flowevent_shape() -> None:
    event = FlowEvent(flow_id="f1", step_id="s1", kind="step_ok", detail={"n": 1})
    assert event.flow_id == "f1"
    assert event.step_id == "s1"
    assert event.kind == "step_ok"
    assert event.detail == {"n": 1}


def test_stepguard_truthy() -> None:
    guard = StepGuard(key="ok", op="truthy")
    assert guard.matches({"ok": True}) is True
    assert guard.matches({"ok": False}) is False
    assert guard.matches({}) is False


def test_stepguard_exists_eq_ne() -> None:
    assert StepGuard(key="n", op="exists").matches({"n": 0}) is True
    assert StepGuard(key="n", op="exists").matches({}) is False
    assert StepGuard(key="n", op="eq", value=3).matches({"n": 3}) is True
    assert StepGuard(key="n", op="eq", value=3).matches({"n": 4}) is False
    assert StepGuard(key="n", op="ne", value=3).matches({"n": 4}) is True
    assert StepGuard(key="n", op="ne", value=3).matches({"n": 3}) is False
