# © Lakshya Badjatya — Author
"""The :class:`FlowEngine`: execute a planned goal as a checkpointed workflow.

Phase-1 spine. The engine decomposes a goal into a :class:`Flow` (via the
existing :class:`~friday.core.planner.Planner`), then runs the steps in dependency
order:

* a ``reason`` step calls the injected :class:`~friday.providers.llm.LLMProvider`;
* a ``tool`` step dispatches through the fail-closed :class:`~friday.broker.broker.Broker`
  (validate → classify → gate → inject secrets → execute → audit), so the engine
  never touches a side-effecting tool directly;
* after **every** transition the flow is checkpointed to the store and a
  :class:`~friday.flows.models.FlowEvent` is appended to the hash-chained audit.

Honest by construction, like :class:`~friday.protocols.runner.ProtocolRunner`: the
run stops at the first failure (no silent half-completion) and pauses — rather
than guessing — when the broker returns ``needs_confirmation``. Because each step
checkpoints and a settled step is skipped on re-entry, :meth:`run` is also the
resume path: a crashed flow reloaded from the store continues exactly where it
stopped.

This module imports no LLM SDK — only the :class:`LLMProvider` contract — and
reads no configuration (every collaborator is injected).
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import UTC, datetime
from typing import Any, Protocol

from friday.core.planner import Plan, Planner, PlanStep
from friday.errors import ProviderError
from friday.flows.models import Flow, FlowEvent, FlowStatus, FlowStep, StepStatus
from friday.flows.simulate import DryRunBroker
from friday.flows.store import SQLiteFlowStore
from friday.flows.templates import FlowTemplate, FlowTemplateStore
from friday.providers.llm import LLMProvider, Message
from friday.tools.base import ToolResult

logger = logging.getLogger("friday.flows.engine")

# Step run outcomes the inner executor returns to the run loop.
_OK = "ok"
_FAIL = "fail"
_PAUSE = "pause"


class _Broker(Protocol):
    """The slice of :class:`~friday.broker.broker.Broker` the engine depends on."""

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
        """Run one brokered tool call and return its result."""
        ...


class _Audit(Protocol):
    """The slice of the audit ledger the engine depends on (one append per event)."""

    def append(self, record: dict[str, Any]) -> Any:
        """Persist one (hash-chained) record."""
        ...


class _Replanner(Protocol):
    """The adaptive re-planner contract (see :class:`friday.flows.replan.Replanner`)."""

    async def replan(
        self, goal: str, completed: str, failure: str
    ) -> list[PlanStep]:
        """Return recovery steps toward ``goal`` after ``failure``."""
        ...


class FlowEngine:
    """Plan, run, and resume :class:`Flow`s over the broker + audit + store.

    Args:
        planner: Decomposes a goal into a :class:`~friday.core.planner.Plan`.
        broker: The fail-closed gate every ``tool`` step dispatches through.
        store: The durable checkpoint; ``update`` is called after every step.
        audit: The hash-chained ledger; one :class:`FlowEvent` per transition.
        llm: The provider a ``reason`` step calls (only the contract).
        allowed_tools: The per-call allow-list passed to the broker — the set of
            tools a flow's steps may invoke.
        actor / channel: Audit attribution for brokered calls.
        max_steps: Hard cap on executed steps per :meth:`run`, so a degenerate
            plan can never run unbounded.
    """

    def __init__(
        self,
        planner: Planner,
        broker: _Broker,
        store: SQLiteFlowStore,
        audit: _Audit,
        llm: LLMProvider,
        allowed_tools: frozenset[str] | set[str],
        *,
        replanner: _Replanner | None = None,
        max_replans: int = 3,
        templates: FlowTemplateStore | None = None,
        actor: str = "owner",
        channel: str = "flow",
        max_steps: int = 200,
    ) -> None:
        self._planner = planner
        self._broker = broker
        self._store = store
        self._audit = audit
        self._llm = llm
        self._allowed_tools = frozenset(allowed_tools)
        self._replanner = replanner
        self._max_replans = max_replans
        self._templates = templates
        self._actor = actor
        self._channel = channel
        self._max_steps = max_steps
        # Dry-run state: set while ``run(simulate=True)`` is executing so tool
        # steps predict instead of execute (restored around nested sub-flows).
        self._simulate = False
        self._dry: DryRunBroker | None = None

    # -- planning ---------------------------------------------------------- #
    async def plan(self, goal: str) -> Flow:
        """Decompose ``goal`` into a persisted ``planned`` :class:`Flow`."""
        plan = await self._planner.decompose(goal)
        now = datetime.now(UTC).isoformat()
        flow = Flow(
            id=uuid.uuid4().hex,
            goal=goal,
            steps=[self._to_flow_step(step) for step in plan.steps],
            status=FlowStatus.PLANNED,
            created_at=now,
            updated_at=now,
        )
        self._store.create(flow)
        self._emit(flow, None, "planned", {"steps": len(flow.steps)})
        return flow

    def get(self, flow_id: str) -> Flow | None:
        """Return the stored flow with ``flow_id`` (or ``None``)."""
        return self._store.get(flow_id)

    def list_flows(self, status: FlowStatus | None = None) -> list[Flow]:
        """Return stored flows, optionally filtered by ``status``."""
        return self._store.list_flows(status)

    async def simulate(self, flow_id: str) -> Flow | None:
        """Dry-run a stored flow (predict side effects); ``None`` if absent."""
        flow = self._store.get(flow_id)
        if flow is None:
            return None
        return await self.run(flow, simulate=True)

    async def resume_all(self) -> list[Flow]:
        """Resume every flow left ``RUNNING`` by a crash (startup recovery).

        Only ``RUNNING`` flows are auto-resumed; a ``PAUSED`` flow stays paused
        (the owner paused it on purpose). Each resumes from its checkpoint, so a
        settled step never re-runs.
        """
        resumed: list[Flow] = []
        for flow in self._store.list_flows(status=FlowStatus.RUNNING):
            resumed.append(await self.run(flow))
        return resumed

    def save_template(self, template: FlowTemplate) -> FlowTemplate | None:
        """Register a reusable flow template; ``None`` when templates are off."""
        if self._templates is None:
            return None
        return self._templates.save(template)

    def list_templates(self) -> list[FlowTemplate]:
        """Every registered flow template (empty when templates are off)."""
        return self._templates.list() if self._templates is not None else []

    async def run_template(
        self,
        name: str,
        params: dict[str, str] | None = None,
        *,
        confirmed: bool = False,
    ) -> Flow | None:
        """Instantiate template ``name`` into a fresh flow and run it; ``None`` if absent."""
        if self._templates is None:
            return None
        flow = self._templates.instantiate(name, params)
        if flow is None:
            return None
        self._store.create(flow)
        return await self.run(flow, confirmed=confirmed)

    async def run_by_id(self, flow_id: str, *, confirmed: bool = False) -> Flow | None:
        """Load ``flow_id`` from the store and :meth:`run` it; ``None`` if absent."""
        flow = self._store.get(flow_id)
        if flow is None:
            return None
        return await self.run(flow, confirmed=confirmed)

    async def resume(self, flow_id: str, *, confirmed: bool = False) -> Flow | None:
        """Resume a stored flow from its checkpoint (alias of :meth:`run_by_id`)."""
        return await self.run_by_id(flow_id, confirmed=confirmed)

    async def approve(
        self, flow_id: str, *, step_id: str | None = None, confirmed: bool = False
    ) -> Flow | None:
        """Clear a step's approval gate and resume; ``None`` if the flow is absent.

        Approves ``step_id`` (or the next step awaiting approval) and runs the
        flow on past the gate. The approval is persisted on the flow, so a step
        cleared once is never re-gated.
        """
        flow = self._store.get(flow_id)
        if flow is None:
            return None
        target = step_id or self._awaiting_step_id(flow)
        if target is not None and target not in flow.approvals:
            flow.approvals.append(target)
            self._emit(flow, None, "approved", {"step": target})
            self._checkpoint(flow)
        return await self.run(flow, confirmed=confirmed)

    async def cancel(self, flow_id: str) -> Flow | None:
        """Cancel a stored flow (terminal); ``None`` if absent."""
        flow = self._store.get(flow_id)
        if flow is None:
            return None
        flow.status = FlowStatus.CANCELLED
        self._emit(flow, None, "cancelled", {})
        self._checkpoint(flow)
        return flow

    async def pause(self, flow_id: str) -> Flow | None:
        """Pause a stored flow (owner steering); ``None`` if absent."""
        flow = self._store.get(flow_id)
        if flow is None:
            return None
        flow.status = FlowStatus.PAUSED
        self._emit(flow, None, "paused", {"reason": "owner"})
        self._checkpoint(flow)
        return flow

    async def skip(self, flow_id: str, step_id: str) -> Flow | None:
        """Skip a still-pending step (owner steering); ``None`` if the flow is absent."""
        flow = self._store.get(flow_id)
        if flow is None:
            return None
        for step in flow.steps:
            if step.id == step_id and step.status == StepStatus.PENDING:
                step.status = StepStatus.SKIPPED
                self._emit(flow, step, "skipped", {"reason": "owner"})
        self._checkpoint(flow)
        return flow

    def _awaiting_step_id(self, flow: Flow) -> str | None:
        """The id of the next pending step still awaiting approval (or ``None``)."""
        for step in self._ordered(flow):
            if (
                step.status == StepStatus.PENDING
                and step.requires_approval
                and step.id not in flow.approvals
            ):
                return step.id
        return None

    @staticmethod
    def _to_flow_step(step: PlanStep) -> FlowStep:
        """Map a planner :class:`PlanStep` onto a runtime :class:`FlowStep`."""
        return FlowStep(
            id=step.id,
            description=step.description,
            operator=step.operator,
            tool=step.tool,
            args=dict(step.args),
            depends_on=list(step.depends_on),
            side_effecting=step.side_effecting,
            kind="tool" if step.tool else "reason",
        )

    # -- execution / resume ------------------------------------------------ #
    async def run(
        self, flow: Flow, *, confirmed: bool = False, simulate: bool = False
    ) -> Flow:
        """Drive ``flow`` to a terminal state or a pause; checkpoint throughout.

        Re-entrant: a step already ``SUCCEEDED``/``SKIPPED`` is not re-run, so the
        same call resumes a reloaded flow. Stops at the first failure (``FAILED``)
        and pauses (``NEEDS_CONFIRMATION``) when the broker withholds a
        side-effecting call — never half-completing silently. With ``simulate``,
        tool steps predict (a :class:`DryRunBroker`) instead of executing.
        """
        prev_sim, prev_dry = self._simulate, self._dry
        self._simulate = simulate
        self._dry = DryRunBroker(self._broker) if simulate else None
        try:
            return await self._drive(flow, confirmed=confirmed)
        finally:
            self._simulate, self._dry = prev_sim, prev_dry

    async def _drive(self, flow: Flow, *, confirmed: bool) -> Flow:
        """The run loop (wrapped by :meth:`run` for dry-run state management)."""
        flow.status = FlowStatus.RUNNING
        self._checkpoint(flow)
        executed = 0
        replans = 0
        while True:
            step = self._next_step(flow)
            if step is None:
                break
            # Conditional branch: a guard that does not hold against the context
            # bus skips the step (settled, not run) — its dependents still proceed.
            if step.when is not None and not step.when.matches(flow.context):
                step.status = StepStatus.SKIPPED
                self._emit(flow, step, "skipped", {"guard": step.when.key})
                self._checkpoint(flow)
                continue
            # Human-in-the-loop: an unapproved gated step pauses the flow before it
            # runs; the owner clears it via approve().
            if step.requires_approval and step.id not in flow.approvals:
                flow.status = FlowStatus.AWAITING_APPROVAL
                self._emit(flow, step, "awaiting_approval", {"step": step.id})
                self._checkpoint(flow)
                return flow
            # Cost governor: stop before a step would run over the token budget.
            if (
                flow.budget_tokens is not None
                and flow.spent_tokens >= flow.budget_tokens
            ):
                self._emit(
                    flow,
                    step,
                    "budget_abort",
                    {"spent": flow.spent_tokens, "budget": flow.budget_tokens},
                )
                return self._finish(flow, FlowStatus.FAILED, "budget exceeded")
            executed += 1
            if executed > self._max_steps:  # pragma: no cover - defensive bound
                return self._finish(flow, FlowStatus.FAILED, "max_steps exceeded")
            outcome = await self._execute_step(flow, step, confirmed=confirmed)
            self._checkpoint(flow)
            if outcome == _PAUSE:
                flow.status = FlowStatus.NEEDS_CONFIRMATION
                self._emit(flow, step, "paused", {"reason": "needs_confirmation"})
                self._checkpoint(flow)
                return flow
            if outcome == _FAIL:
                await self._compensate(flow, step, confirmed=confirmed)
                if self._replanner is not None and replans < self._max_replans:
                    replans += 1
                    if await self._replan(flow, step, replans):
                        self._emit(flow, step, "replanned", {"round": replans})
                        self._checkpoint(flow)
                        continue
                return self._finish(flow, FlowStatus.FAILED, f"step {step.id} failed")
        return self._finish(flow, FlowStatus.SUCCEEDED, "all steps complete")

    def _next_step(self, flow: Flow) -> FlowStep | None:
        """The next ``PENDING`` step (topo order) whose deps are all settled.

        ``SUCCEEDED`` and ``SKIPPED`` both count as settled, so a skipped
        conditional branch never blocks its dependents. ``None`` when no runnable
        pending step remains (the flow is then complete).
        """
        settled = {StepStatus.SUCCEEDED, StepStatus.SKIPPED}
        by_id = {s.id: s for s in flow.steps}
        for step in self._ordered(flow):
            if step.status != StepStatus.PENDING:
                continue
            if all(
                by_id[dep].status in settled
                for dep in step.depends_on
                if dep in by_id
            ):
                return step
        return None

    async def _execute_step(self, flow: Flow, step: FlowStep, *, confirmed: bool) -> str:
        """Run a step with its retry budget and (optional) timeout.

        Each attempt increments ``step.attempts``. A timeout or a failure consumes
        an attempt; the step succeeds/pauses immediately on a non-failure outcome,
        and only reports ``_FAIL`` once the retry budget is spent.
        """
        attempts_allowed = step.retry + 1
        for attempt in range(attempts_allowed):
            try:
                if step.timeout_s is not None:
                    outcome = await asyncio.wait_for(
                        self._run_step(flow, step, confirmed=confirmed),
                        step.timeout_s,
                    )
                else:
                    outcome = await self._run_step(flow, step, confirmed=confirmed)
            except TimeoutError:
                step.status = StepStatus.FAILED
                step.result = {"error": "timeout"}
                self._emit(flow, step, "step_failed", {"error": "timeout"})
                outcome = _FAIL
            if outcome in (_OK, _PAUSE):
                return outcome
            if attempt < attempts_allowed - 1:
                self._emit(flow, step, "retried", {"attempt": step.attempts})
        return _FAIL

    async def _compensate(self, flow: Flow, step: FlowStep, *, confirmed: bool) -> None:
        """Run a failed step's compensation tool (best-effort, brokered) if set.

        Compensation is system-initiated cleanup, so it dispatches ``confirmed``
        through the broker and marks the step ``COMPENSATED`` regardless of the
        tool's own result — the audit records whether it succeeded.
        """
        if not step.compensation:
            return
        result = await self._broker.dispatch(
            step.compensation,
            {},
            allowed_tools=self._allowed_tools,
            confirmed=True,
            actor=self._actor,
            channel=self._channel,
        )
        step.status = StepStatus.COMPENSATED
        self._emit(
            flow, step, "compensated", {"tool": step.compensation, "ok": result.ok}
        )
        self._checkpoint(flow)

    async def _replan(self, flow: Flow, failed: FlowStep, round_no: int) -> bool:
        """Splice a re-planned remainder onto ``flow`` after ``failed``; True if any.

        Supersedes any still-pending steps (they become ``SKIPPED``) and appends
        the recovery steps with re-namespaced ids so the dependency graph stays
        valid. ``False`` (no recovery) when no re-planner is wired or it yields
        nothing.
        """
        if self._replanner is None:
            return False
        completed = ", ".join(
            f"{s.id}:{s.status.value}"
            for s in flow.steps
            if s.status != StepStatus.PENDING
        )
        recovery = await self._replanner.replan(
            flow.goal, completed, failed.description
        )
        if not recovery:
            return False
        for pending in flow.steps:
            if pending.status == StepStatus.PENDING:
                pending.status = StepStatus.SKIPPED
        suffix = f"__r{round_no}"
        id_map = {ps.id: f"{ps.id}{suffix}" for ps in recovery}
        for ps in recovery:
            flow.steps.append(
                FlowStep(
                    id=id_map[ps.id],
                    description=ps.description,
                    operator=ps.operator,
                    tool=ps.tool,
                    args=dict(ps.args),
                    depends_on=[id_map[d] for d in ps.depends_on if d in id_map],
                    side_effecting=ps.side_effecting,
                    kind="tool" if ps.tool else "reason",
                )
            )
        return True

    async def _run_step(self, flow: Flow, step: FlowStep, *, confirmed: bool) -> str:
        """Run one step; return ``_OK`` / ``_FAIL`` / ``_PAUSE``."""
        step.status = StepStatus.RUNNING
        step.attempts += 1
        if step.kind == "subflow":
            return await self._run_subflow_step(flow, step)
        if step.kind == "tool" and step.tool is not None:
            return await self._run_tool_step(flow, step, confirmed=confirmed)
        return await self._run_reason_step(flow, step)

    async def _run_subflow_step(self, flow: Flow, step: FlowStep) -> str:
        """Run a nested flow from a template; succeed iff the child succeeds."""
        if self._templates is None:
            step.status = StepStatus.FAILED
            step.result = {"error": "no_templates"}
            self._emit(flow, step, "step_failed", {"error": "no_templates"})
            return _FAIL
        name = step.args.get("template")
        raw_params = step.args.get("params") or {}
        params = {str(k): str(v) for k, v in dict(raw_params).items()}
        child = self._templates.instantiate(str(name), params) if name else None
        if child is None:
            step.status = StepStatus.FAILED
            step.result = {"error": "unknown_template"}
            self._emit(flow, step, "step_failed", {"error": "unknown_template"})
            return _FAIL
        child.parent_flow_id = flow.id
        self._store.create(child)
        child = await self.run(child, simulate=self._simulate)
        if child.status == FlowStatus.SUCCEEDED:
            step.status = StepStatus.SUCCEEDED
            step.result = {"child_flow_id": child.id}
            step.rationale = f"sub-flow {child.id} succeeded"
            flow.context[step.id] = {"child_flow_id": child.id}
            self._emit(flow, step, "step_ok", {"child_flow_id": child.id})
            return _OK
        step.status = StepStatus.FAILED
        step.result = {"child_flow_id": child.id, "child_status": child.status.value}
        self._emit(flow, step, "step_failed", {"child_flow_id": child.id})
        return _FAIL

    async def _run_tool_step(
        self, flow: Flow, step: FlowStep, *, confirmed: bool
    ) -> str:
        broker = self._dry if self._simulate and self._dry is not None else self._broker
        result = await broker.dispatch(
            step.tool or "",
            dict(step.args),
            allowed_tools=self._allowed_tools,
            confirmed=confirmed,
            actor=self._actor,
            channel=self._channel,
        )
        if (
            not result.ok
            and result.error is not None
            and result.error.code == "needs_confirmation"
        ):
            # The broker withheld an irreversible call. Leave the step pending and
            # let the run loop pause the flow — never guess past a confirm-gate.
            step.status = StepStatus.PENDING
            return _PAUSE
        if not result.ok:
            code = result.error.code if result.error is not None else "error"
            step.status = StepStatus.FAILED
            step.result = {"error": code}
            self._emit(flow, step, "step_failed", {"error": code})
            return _FAIL
        step.status = StepStatus.SUCCEEDED
        step.result = dict(result.data)
        step.rationale = f"ran {step.tool}"
        flow.context[step.id] = result.data
        self._emit(flow, step, "step_ok", {"tool": step.tool})
        return _OK

    def _reason_messages(self, flow: Flow, step: FlowStep) -> list[Message]:
        """Build a reason step's prompt, injecting prior results from the context bus.

        Earlier steps publish their outputs into ``flow.context``; a later reason
        step receives them as a system preamble so knowledge flows step→step and
        operator→operator (the shared context bus). Values are truncated so the
        prompt stays bounded.
        """
        messages: list[Message] = []
        if flow.context:
            rendered = "\n".join(
                f"- {key}: {self._short(value)}"
                for key, value in flow.context.items()
            )
            messages.append(
                Message(
                    role="system",
                    content=f"Context from prior steps:\n{rendered}",
                )
            )
        messages.append(Message(role="user", content=step.description))
        return messages

    @staticmethod
    def _short(value: Any, limit: int = 200) -> str:
        """A bounded string rendering of a context value (JSON for non-strings)."""
        text = value if isinstance(value, str) else json.dumps(value, default=str)
        return text[:limit]

    async def _run_reason_step(self, flow: Flow, step: FlowStep) -> str:
        try:
            response = await self._llm.complete(
                self._reason_messages(flow, step), None
            )
        except ProviderError as exc:
            logger.warning("flow %s reason step %s failed: %s", flow.id, step.id, exc)
            step.status = StepStatus.FAILED
            step.result = {"error": "provider"}
            self._emit(flow, step, "step_failed", {"error": "provider"})
            return _FAIL
        text = (response.text or "").strip()
        if not text:
            step.status = StepStatus.FAILED
            step.result = {"error": "empty"}
            self._emit(flow, step, "step_failed", {"error": "empty"})
            return _FAIL
        step.status = StepStatus.SUCCEEDED
        step.result = {"text": text}
        step.rationale = text[:200]
        flow.context[step.id] = text
        flow.spent_tokens += (
            response.usage.prompt_tokens + response.usage.completion_tokens
        )
        self._emit(flow, step, "step_ok", {})
        return _OK

    # -- ordering ---------------------------------------------------------- #
    def _ordered(self, flow: Flow) -> list[FlowStep]:
        """Steps in dependency order, reusing the planner's tested Kahn sort.

        Falls back to declared order if the graph is degenerate (the planner has
        already validated decomposed graphs, so this only guards hand-built flows).
        """
        probe = Plan(
            goal=flow.goal,
            steps=[
                PlanStep(id=s.id, description=s.description, depends_on=s.depends_on)
                for s in flow.steps
            ],
        )
        by_id = {s.id: s for s in flow.steps}
        try:
            return [by_id[ps.id] for ps in probe.topological_order()]
        except (ValueError, KeyError):
            return list(flow.steps)

    # -- bookkeeping ------------------------------------------------------- #
    def _finish(self, flow: Flow, status: FlowStatus, detail: str) -> Flow:
        flow.status = status
        kind = "succeeded" if status is FlowStatus.SUCCEEDED else "failed"
        self._emit(flow, None, kind, {"detail": detail})
        self._checkpoint(flow)
        return flow

    def _checkpoint(self, flow: Flow) -> None:
        """Persist the flow's current state (the resumable checkpoint)."""
        flow.updated_at = datetime.now(UTC).isoformat()
        self._store.update(flow)

    def _emit(
        self, flow: Flow, step: FlowStep | None, kind: str, detail: dict[str, Any]
    ) -> None:
        """Append one :class:`FlowEvent` to the audit ledger (tamper-evident)."""
        event = FlowEvent(
            flow_id=flow.id,
            step_id=step.id if step is not None else None,
            kind=kind,
            detail=detail,
        )
        self._audit.append(event.model_dump())
