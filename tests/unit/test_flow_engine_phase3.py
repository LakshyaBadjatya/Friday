# © Lakshya Badjatya — Author
"""Unit tests for the Flow Engine Phase-3 control surface.

Covers human-in-the-loop approval gates (#7), owner steering — approve / cancel /
skip / pause (#6, #15) — and the per-flow token budget governor (#9).
"""

from __future__ import annotations

from typing import Any

from friday.core.planner import Planner
from friday.flows.engine import FlowEngine
from friday.flows.models import Flow, FlowStatus, FlowStep, StepStatus
from friday.flows.store import SQLiteFlowStore
from friday.providers.llm import FakeLLM, LLMResponse, Message, Usage


class TokenLLM:
    """An LLM that reports a fixed token cost per call (for budget tests)."""

    def __init__(self, per_call: int) -> None:
        self._per_call = per_call

    async def complete(self, messages: list[Message], tools: Any = None) -> LLMResponse:
        return LLMResponse(
            text="ok", usage=Usage(prompt_tokens=self._per_call, completion_tokens=0)
        )


class ListAudit:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def append(self, record: dict[str, Any]) -> dict[str, Any]:
        self.records.append(record)
        return record


def _engine(
    *, llm: Any | None = None, audit: ListAudit | None = None
) -> tuple[FlowEngine, ListAudit, SQLiteFlowStore]:
    audit = audit or ListAudit()
    store = SQLiteFlowStore()
    engine = FlowEngine(
        planner=Planner(FakeLLM(responses=[LLMResponse(text="[]")])),
        broker=None,  # type: ignore[arg-type]
        store=store,
        audit=audit,
        llm=llm or FakeLLM(responses=[LLMResponse(text="ok")] * 5),
        allowed_tools=frozenset({"notify"}),
    )
    return engine, audit, store


def _flow(steps: list[FlowStep]) -> Flow:
    return Flow(id="f", goal="g", steps=steps)


async def test_requires_approval_pauses() -> None:
    engine, audit, store = _engine()
    flow = _flow(
        [FlowStep(id="s1", description="risky", kind="reason", requires_approval=True)]
    )
    store.create(flow)
    flow = await engine.run(flow)
    assert flow.status == FlowStatus.AWAITING_APPROVAL
    assert flow.steps[0].status == StepStatus.PENDING
    assert "awaiting_approval" in [r["kind"] for r in audit.records]


async def test_approve_resumes_and_runs() -> None:
    engine, audit, store = _engine()
    flow = _flow(
        [FlowStep(id="s1", description="risky", kind="reason", requires_approval=True)]
    )
    store.create(flow)
    await engine.run(flow)  # pauses awaiting approval

    resumed = await engine.approve("f")
    assert resumed is not None
    assert resumed.status == FlowStatus.SUCCEEDED
    assert resumed.steps[0].status == StepStatus.SUCCEEDED
    assert "approved" in [r["kind"] for r in audit.records]


async def test_cancel_sets_cancelled() -> None:
    engine, _, store = _engine()
    flow = _flow([FlowStep(id="s1", description="x", kind="reason")])
    store.create(flow)
    cancelled = await engine.cancel("f")
    assert cancelled is not None
    assert cancelled.status == FlowStatus.CANCELLED


async def test_skip_marks_step_skipped() -> None:
    engine, _, store = _engine()
    flow = _flow(
        [
            FlowStep(id="s1", description="a", kind="reason"),
            FlowStep(id="s2", description="b", kind="reason"),
        ]
    )
    store.create(flow)
    updated = await engine.skip("f", "s1")
    assert updated is not None
    assert updated.steps[0].status == StepStatus.SKIPPED


async def test_pause_sets_paused() -> None:
    engine, _, store = _engine()
    flow = _flow([FlowStep(id="s1", description="x", kind="reason")])
    store.create(flow)
    paused = await engine.pause("f")
    assert paused is not None
    assert paused.status == FlowStatus.PAUSED


async def test_budget_abort() -> None:
    engine, audit, store = _engine(llm=TokenLLM(per_call=20))
    flow = _flow(
        [
            FlowStep(id="s1", description="a", kind="reason"),
            FlowStep(id="s2", description="b", kind="reason"),
        ]
    )
    flow.budget_tokens = 10  # the first step already blows the budget
    store.create(flow)
    flow = await engine.run(flow)
    assert flow.status == FlowStatus.FAILED
    assert "budget_abort" in [r["kind"] for r in audit.records]
    assert flow.steps[1].status == StepStatus.PENDING  # second step never ran
    assert flow.spent_tokens >= 20
