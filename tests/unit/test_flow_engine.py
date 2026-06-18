# © Lakshya Badjatya — Author
"""Unit tests for :class:`friday.flows.engine.FlowEngine` (Phase-1 spine).

Drives the engine with offline fakes: a real :class:`Planner` over a ``FakeLLM``
that returns a scripted step graph, a recording broker, a list-backed audit, and
a real in-memory :class:`SQLiteFlowStore`. Covers: plan→run success, brokered
tool dispatch, honest stop on failure (no half-completion), a broker
``needs_confirmation`` pause, resume skipping settled steps, and that every
transition is audited.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from friday.core.planner import Planner
from friday.flows.engine import FlowEngine
from friday.flows.models import FlowStatus, StepStatus
from friday.flows.store import SQLiteFlowStore
from friday.providers.llm import FakeLLM, LLMResponse
from friday.tools.base import ToolError, ToolResult


class RecordingBroker:
    """A fake broker that records dispatch calls and returns scripted results."""

    def __init__(self, results: list[ToolResult]) -> None:
        self._results = list(results)
        self.calls: list[SimpleNamespace] = []

    async def dispatch(
        self,
        tool_name: str,
        raw_args: dict[str, Any],
        *,
        allowed_tools: frozenset[str] | set[str],
        confirmed: bool = False,
        actor: str = "owner",
        channel: str = "chat",
    ) -> ToolResult:
        self.calls.append(
            SimpleNamespace(
                tool=tool_name, args=raw_args, allowed_tools=allowed_tools,
                confirmed=confirmed,
            )
        )
        return self._results.pop(0)


class ListAudit:
    """A fake audit ledger that just collects appended records."""

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def append(self, record: dict[str, Any]) -> dict[str, Any]:
        self.records.append(record)
        return record


def _plan_json(*steps: dict[str, Any]) -> str:
    return json.dumps(list(steps))


def _reason_step(sid: str) -> dict[str, Any]:
    return {"id": sid, "description": f"think about {sid}", "tool": None}


def _tool_step(sid: str, tool: str, *, side_effecting: bool = False) -> dict[str, Any]:
    return {
        "id": sid, "description": f"use {tool}", "tool": tool,
        "side_effecting": side_effecting,
    }


def _engine(
    plan_json: str,
    *,
    reason_replies: list[str] | None = None,
    broker: RecordingBroker | None = None,
    audit: ListAudit | None = None,
) -> tuple[FlowEngine, RecordingBroker, ListAudit]:
    planner = Planner(FakeLLM(responses=[LLMResponse(text=plan_json)]))
    llm = FakeLLM(responses=[LLMResponse(text=t) for t in (reason_replies or [])])
    broker = broker or RecordingBroker([])
    audit = audit or ListAudit()
    engine = FlowEngine(
        planner=planner,
        broker=broker,
        store=SQLiteFlowStore(),
        audit=audit,
        llm=llm,
        allowed_tools=frozenset({"web_search", "notify"}),
    )
    return engine, broker, audit


async def test_plan_then_run_succeeds() -> None:
    engine, _, _ = _engine(
        _plan_json(_reason_step("s1"), _reason_step("s2")),
        reason_replies=["did s1", "did s2"],
    )
    flow = await engine.plan("do two things")
    assert flow.status == FlowStatus.PLANNED
    assert len(flow.steps) == 2

    flow = await engine.run(flow)
    assert flow.status == FlowStatus.SUCCEEDED
    assert all(s.status == StepStatus.SUCCEEDED for s in flow.steps)


async def test_tool_step_dispatches_through_broker() -> None:
    broker = RecordingBroker([ToolResult(ok=True, data={"hits": 1})])
    engine, broker, _ = _engine(
        _plan_json(_tool_step("s1", "web_search")), broker=broker,
    )
    flow = await engine.run(await engine.plan("search a thing"))
    assert flow.status == FlowStatus.SUCCEEDED
    assert len(broker.calls) == 1
    assert broker.calls[0].tool == "web_search"
    assert "web_search" in broker.calls[0].allowed_tools


async def test_failed_step_stops_and_never_half_completes() -> None:
    broker = RecordingBroker(
        [
            ToolResult(ok=True, data={}),
            ToolResult(ok=False, data={}, error=ToolError(code="boom", message="x")),
        ]
    )
    engine, broker, _ = _engine(
        _plan_json(
            _tool_step("s1", "web_search"),
            _tool_step("s2", "web_search"),
            _tool_step("s3", "web_search"),
        ),
        broker=broker,
    )
    flow = await engine.run(await engine.plan("three things"))
    assert flow.status == FlowStatus.FAILED
    assert flow.steps[1].status == StepStatus.FAILED
    assert flow.steps[2].status == StepStatus.PENDING  # later step never ran
    assert len(broker.calls) == 2  # stopped after the failure


async def test_needs_confirmation_pauses_the_flow() -> None:
    broker = RecordingBroker(
        [
            ToolResult(
                ok=False,
                data={"needs_confirmation": True},
                error=ToolError(code="needs_confirmation", message="confirm"),
            )
        ]
    )
    engine, _, _ = _engine(
        _plan_json(_tool_step("s1", "notify", side_effecting=True)), broker=broker,
    )
    flow = await engine.run(await engine.plan("notify someone"))
    assert flow.status == FlowStatus.NEEDS_CONFIRMATION
    assert flow.steps[0].status != StepStatus.SUCCEEDED


async def test_resume_skips_completed_steps() -> None:
    engine, _, _ = _engine(
        _plan_json(_reason_step("s1"), _reason_step("s2")),
        reason_replies=["only s2 runs"],  # one reply: s1 must NOT re-run
    )
    flow = await engine.plan("two things")
    flow.steps[0].status = StepStatus.SUCCEEDED
    flow = await engine.run(flow)
    assert flow.status == FlowStatus.SUCCEEDED


async def test_every_transition_is_audited() -> None:
    engine, _, audit = _engine(
        _plan_json(_reason_step("s1")), reason_replies=["done"],
    )
    await engine.run(await engine.plan("one thing"))
    kinds = [r["kind"] for r in audit.records]
    assert "planned" in kinds
    assert "step_ok" in kinds
    assert "succeeded" in kinds
