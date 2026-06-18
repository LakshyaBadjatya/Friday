# © Lakshya Badjatya — Author
"""Unit tests for the Flow Engine Phase-2 dynamics.

Covers the four additive behaviours layered on the spine: conditional ``when``
skipping (#10), per-step retry / timeout / compensation (#8), shared-context
injection into reason steps (#3), and adaptive re-planning on failure (#2). Each
is exercised on a directly-constructed :class:`Flow` (these controls are
``FlowStep``-only fields set after planning).
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

from friday.core.planner import Planner
from friday.flows.engine import FlowEngine
from friday.flows.models import Flow, FlowStatus, FlowStep, StepGuard, StepStatus
from friday.flows.replan import Replanner
from friday.flows.store import SQLiteFlowStore
from friday.providers.llm import FakeLLM, LLMResponse, Message
from friday.tools.base import ToolError, ToolResult


class ScriptedBroker:
    """Returns scripted results per dispatch and records calls."""

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
        self.calls.append(SimpleNamespace(tool=tool_name, confirmed=confirmed))
        if self._results:
            return self._results.pop(0)
        return ToolResult(ok=True, data={})


class SleepyBroker:
    """A broker whose dispatch never returns in time (for timeout tests)."""

    async def dispatch(self, *args: Any, **kwargs: Any) -> ToolResult:
        await asyncio.sleep(5)
        return ToolResult(ok=True, data={})


class RecordingLLM:
    """An LLM that records the messages it was asked to complete."""

    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.seen: list[list[Message]] = []

    async def complete(
        self, messages: list[Message], tools: Any = None
    ) -> LLMResponse:
        self.seen.append(list(messages))
        return LLMResponse(text=self._replies.pop(0) if self._replies else "ok")


class ListAudit:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def append(self, record: dict[str, Any]) -> dict[str, Any]:
        self.records.append(record)
        return record


def _engine(
    *,
    broker: Any | None = None,
    llm: Any | None = None,
    replanner: Replanner | None = None,
    audit: ListAudit | None = None,
    plan_json: str = "[]",
) -> tuple[FlowEngine, ListAudit, SQLiteFlowStore]:
    audit = audit or ListAudit()
    store = SQLiteFlowStore()
    engine = FlowEngine(
        planner=Planner(FakeLLM(responses=[LLMResponse(text=plan_json)])),
        broker=broker or ScriptedBroker([]),
        store=store,
        audit=audit,
        llm=llm or FakeLLM(responses=[LLMResponse(text="ok")]),
        allowed_tools=frozenset({"web_search", "rollback", "notify"}),
        replanner=replanner,
        max_replans=2,
    )
    return engine, audit, store


def _flow(steps: list[FlowStep]) -> Flow:
    return Flow(id="f", goal="g", steps=steps)


async def test_when_guard_skips_step() -> None:
    engine, audit, store = _engine()
    flow = _flow(
        [
            FlowStep(id="s1", description="reason", kind="reason"),
            FlowStep(
                id="s2", description="gated", kind="reason",
                when=StepGuard(key="missing", op="exists"),
            ),
        ]
    )
    store.create(flow)
    flow = await engine.run(flow)
    assert flow.status == FlowStatus.SUCCEEDED
    assert flow.steps[1].status == StepStatus.SKIPPED
    assert "skipped" in [r["kind"] for r in audit.records]


async def test_retry_then_success() -> None:
    broker = ScriptedBroker(
        [
            ToolResult(ok=False, data={}, error=ToolError(code="flaky", message="x")),
            ToolResult(ok=True, data={"done": True}),
        ]
    )
    engine, _, store = _engine(broker=broker)
    flow = _flow(
        [FlowStep(id="s1", description="call", kind="tool", tool="web_search", retry=1)]
    )
    store.create(flow)
    flow = await engine.run(flow)
    assert flow.status == FlowStatus.SUCCEEDED
    assert flow.steps[0].status == StepStatus.SUCCEEDED
    assert flow.steps[0].attempts == 2
    assert len(broker.calls) == 2


async def test_retry_exhausted_fails() -> None:
    broker = ScriptedBroker(
        [ToolResult(ok=False, data={}, error=ToolError(code="boom", message="x"))] * 3
    )
    engine, _, store = _engine(broker=broker)
    flow = _flow(
        [FlowStep(id="s1", description="call", kind="tool", tool="web_search", retry=1)]
    )
    store.create(flow)
    flow = await engine.run(flow)
    assert flow.status == FlowStatus.FAILED
    assert flow.steps[0].attempts == 2  # 1 try + 1 retry


async def test_timeout_fails_the_step() -> None:
    engine, _, store = _engine(broker=SleepyBroker())
    flow = _flow(
        [
            FlowStep(
                id="s1", description="slow", kind="tool", tool="web_search",
                timeout_s=0.05,
            )
        ]
    )
    store.create(flow)
    flow = await engine.run(flow)
    assert flow.status == FlowStatus.FAILED
    assert flow.steps[0].result.get("error") == "timeout"


async def test_compensation_runs_on_failure() -> None:
    broker = ScriptedBroker(
        [
            ToolResult(ok=False, data={}, error=ToolError(code="boom", message="x")),
            ToolResult(ok=True, data={}),  # the compensation call
        ]
    )
    engine, audit, store = _engine(broker=broker)
    flow = _flow(
        [
            FlowStep(
                id="s1", description="risky", kind="tool", tool="web_search",
                compensation="rollback",
            )
        ]
    )
    store.create(flow)
    flow = await engine.run(flow)
    assert flow.status == FlowStatus.FAILED
    assert flow.steps[0].status == StepStatus.COMPENSATED
    assert "compensated" in [r["kind"] for r in audit.records]
    assert broker.calls[-1].tool == "rollback"


async def test_reason_step_sees_prior_context() -> None:
    llm = RecordingLLM(["first result", "second result"])
    engine, _, store = _engine(llm=llm)
    flow = _flow(
        [
            FlowStep(id="s1", description="produce a fact", kind="reason"),
            FlowStep(id="s2", description="use the fact", kind="reason"),
        ]
    )
    store.create(flow)
    await engine.run(flow)
    # The second reason step's prompt must carry the first step's result.
    second_prompt = " ".join(m.content for m in llm.seen[1])
    assert "first result" in second_prompt


async def test_adaptive_replan_recovers() -> None:
    # The failed step triggers a re-plan; the planner (FakeLLM) returns a single
    # recovery reason step that then succeeds.
    recovery = '[{"id": "r1", "description": "recover", "tool": null}]'
    replanner = Replanner(Planner(FakeLLM(responses=[LLMResponse(text=recovery)])))
    broker = ScriptedBroker(
        [ToolResult(ok=False, data={}, error=ToolError(code="boom", message="x"))]
    )
    engine, audit, store = _engine(broker=broker, replanner=replanner)
    flow = _flow(
        [FlowStep(id="s1", description="risky", kind="tool", tool="web_search")]
    )
    store.create(flow)
    flow = await engine.run(flow)
    assert flow.status == FlowStatus.SUCCEEDED
    assert "replanned" in [r["kind"] for r in audit.records]
    assert any(s.id.startswith("r1") and s.status == StepStatus.SUCCEEDED
               for s in flow.steps)
