# © Lakshya Badjatya — Author
"""Unit tests for the Flow Engine Phase-4 reuse + governors.

Covers flow templates (#11), nested sub-flows (#12), and dry-run simulation
(#13): instantiating a parameterized template, running one nested inside a
parent, and previewing a side-effecting flow without ever touching the real
broker.
"""

from __future__ import annotations

from typing import Any

from friday.core.planner import Planner
from friday.flows.engine import FlowEngine
from friday.flows.models import Flow, FlowStatus, FlowStep, StepStatus
from friday.flows.store import SQLiteFlowStore
from friday.flows.templates import FlowTemplate, FlowTemplateStore
from friday.providers.llm import FakeLLM, LLMResponse


class BoomBroker:
    """A broker that must never be dispatched (used to prove dry-run isolation)."""

    def __init__(self) -> None:
        self.called = False

    async def dispatch(self, *args: Any, **kwargs: Any) -> Any:
        self.called = True
        raise AssertionError("real broker dispatched during a dry-run")


class ListAudit:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def append(self, record: dict[str, Any]) -> dict[str, Any]:
        self.records.append(record)
        return record


def _engine(
    *, broker: Any | None = None, templates: FlowTemplateStore | None = None
) -> tuple[FlowEngine, SQLiteFlowStore]:
    store = SQLiteFlowStore()
    engine = FlowEngine(
        planner=Planner(FakeLLM(responses=[LLMResponse(text="[]")])),
        broker=broker,  # type: ignore[arg-type]
        store=store,
        audit=ListAudit(),
        llm=FakeLLM(responses=[LLMResponse(text="ok")] * 5),
        allowed_tools=frozenset({"notify"}),
        templates=templates,
    )
    return engine, store


def test_template_instantiate() -> None:
    store = FlowTemplateStore()
    store.save(
        FlowTemplate(
            name="brief",
            goal="brief {topic}",
            steps=[
                FlowStep(id="s1", description="summarize", status=StepStatus.SUCCEEDED)
            ],
        )
    )
    flow = store.instantiate("brief", {"topic": "markets"})
    assert flow is not None
    assert flow.goal == "brief markets"
    assert flow.template == "brief"
    assert flow.steps[0].status == StepStatus.PENDING  # runtime state reset
    assert store.instantiate("missing") is None


async def test_subflow_runs_nested() -> None:
    templates = FlowTemplateStore()
    templates.save(
        FlowTemplate(
            name="child",
            goal="child goal",
            steps=[FlowStep(id="c1", description="do child work", kind="reason")],
        )
    )
    engine, store = _engine(templates=templates)
    parent = Flow(
        id="p",
        goal="parent",
        steps=[
            FlowStep(
                id="s1", description="run child", kind="subflow",
                args={"template": "child"},
            )
        ],
    )
    store.create(parent)
    parent = await engine.run(parent)
    assert parent.status == FlowStatus.SUCCEEDED
    assert parent.steps[0].status == StepStatus.SUCCEEDED
    children = [f for f in store.list_flows() if f.parent_flow_id == "p"]
    assert len(children) == 1
    assert children[0].status == FlowStatus.SUCCEEDED


async def test_dry_run_predicts_without_executing() -> None:
    broker = BoomBroker()
    engine, store = _engine(broker=broker)
    flow = Flow(
        id="f",
        goal="send something irreversible",
        steps=[
            FlowStep(
                id="s1", description="notify", kind="tool", tool="notify",
                side_effecting=True,
            )
        ],
    )
    store.create(flow)
    flow = await engine.run(flow, simulate=True)
    assert flow.status == FlowStatus.SUCCEEDED
    assert broker.called is False  # the real broker was never touched
    assert flow.steps[0].result.get("simulated") is True
