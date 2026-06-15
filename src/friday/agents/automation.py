"""The automation agent: bounded, runaway-safe multi-step job execution.

:class:`AutomationAgent` executes a :class:`Job` — an ordered list of ``steps``
with a hard ``max_steps`` cap — by stepping through the list until one of two
things happens, whichever comes first:

* a **termination predicate** fires (the job decided it is done), or
* the **``max_steps`` cap** is reached.

The cap is the safety rail and the whole reason the executor is written as a
counted ``for`` loop rather than an open ``while``: it is the *structural*
guarantee that a runaway job — one whose termination predicate never fires, or a
job handed effectively unbounded work — cannot loop forever. The executor will
run at most ``min(len(job.steps), job.max_steps)`` steps and then stop, every
time, deterministically.

There is no LLM, no tool call, no network, and no wall-clock in this path:
execution is a pure loop over an in-memory ``Job``, so a given
``(job, predicate)`` always yields the same :class:`AgentResult`. That keeps the
``agents`` package clean for the SDK-isolation guard and the behaviour trivially
testable offline.

The agent declares no tools (``allowed_tools == frozenset()``): step execution
is in-process bookkeeping. When a future stage wires real side-effecting steps,
they go through the registry's confirm-step (build-spec section 12), but the
runaway guarantee proven here is orthogonal to — and survives — that change.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum

from pydantic import BaseModel, Field

from friday.agents.base import AgentResult
from friday.core.state import GraphState

# A termination predicate decides, *after* a step is recorded, whether the job
# is finished. It is handed the step's text and its zero-based index so callers
# can terminate on either content ("saw the 'done' marker") or position.
TerminationPredicate = Callable[[str, int], bool]

# The default predicate never fires: with it, a job is bounded purely by
# ``min(len(steps), max_steps)`` — i.e. the cap and step list are the only
# limits, which is exactly the runaway-safe baseline.
def _never_terminate(step: str, index: int) -> bool:
    return False


class StopReason(StrEnum):
    """Why the executor stopped — surfaced in the agent's summary."""

    #: The termination predicate fired before the cap was reached.
    TERMINATED = "terminated"
    #: The hard ``max_steps`` cap was hit (the runaway guard engaged).
    MAX_STEPS = "max_steps"
    #: Every step in the job ran without the predicate firing or the cap hitting.
    COMPLETED = "completed"


class Job(BaseModel):
    """An ordered automation job: a list of steps under a hard step cap.

    ``steps`` is the (possibly long) sequence of step descriptions to execute in
    order. ``max_steps`` is the inclusive upper bound on how many steps the
    executor will ever run for this job; it must be ``>= 1`` so the cap can
    never be defeated by a zero/negative limit.
    """

    steps: list[str] = Field(default_factory=list)
    max_steps: int = Field(gt=0)


class AutomationAgent:
    """Executes a :class:`Job` step-by-step under a runaway-safe step cap.

    Args:
        terminate: Optional predicate ``(step_text, index) -> bool`` evaluated
            after each executed step; returning ``True`` halts the job early.
            Defaults to a predicate that never fires, so the job is bounded only
            by ``min(len(steps), max_steps)``.
    """

    name: str = "automation"
    allowed_tools: frozenset[str] = frozenset()

    def __init__(self, terminate: TerminationPredicate | None = None) -> None:
        self._terminate: TerminationPredicate = terminate or _never_terminate

    # -- job extraction ----------------------------------------------------- #
    @staticmethod
    def _job_from_state(state: GraphState) -> Job:
        """Pull the :class:`Job` the orchestrator staged in ``scratchpad['job']``.

        The job is carried as a plain dict (state round-trips through JSON), so it
        is re-validated here into a typed :class:`Job` before execution.
        """
        raw = state.scratchpad.get("job", {})
        if isinstance(raw, Job):
            return raw
        return Job.model_validate(raw)

    # -- the bounded executor ----------------------------------------------- #
    def _execute(self, job: Job) -> tuple[list[str], StopReason]:
        """Run ``job`` and return ``(executed_steps, why_it_stopped)``.

        The loop is bounded two ways at once — by the cap and by the step list —
        so it provably cannot run more than ``min(len(steps), max_steps)`` times.
        That is the runaway guard: a never-firing predicate caps at ``max_steps``.
        """
        executed: list[str] = []
        for index, step in enumerate(job.steps):
            if len(executed) >= job.max_steps:
                # Cap reached with steps still pending: the runaway guard engaged.
                return executed, StopReason.MAX_STEPS
            executed.append(step)
            if self._terminate(step, index):
                return executed, StopReason.TERMINATED
        # Ran the whole step list without the predicate firing or hitting the cap.
        # (If the list length exactly equalled the cap, that is still a clean
        # completion: every requested step ran.)
        return executed, StopReason.COMPLETED

    # -- summary ------------------------------------------------------------ #
    @staticmethod
    def _summarize(executed: list[str], reason: StopReason, job: Job) -> str:
        """Human-readable summary: how many steps ran and exactly why it stopped."""
        count = len(executed)
        if reason is StopReason.MAX_STEPS:
            why = (
                f"hit the max_steps cap of {job.max_steps} with "
                f"{len(job.steps) - count} step(s) still pending; stopped to "
                "avoid a runaway"
            )
        elif reason is StopReason.TERMINATED:
            why = "the termination predicate fired, so it stopped early"
        else:  # COMPLETED
            why = "completed all requested steps"
        return f"Automation ran {count} step(s); {why}."

    # -- agent entrypoint --------------------------------------------------- #
    async def run(self, state: GraphState) -> AgentResult:
        """Execute the staged job under the cap and report what happened.

        Returns an :class:`AgentResult` whose ``output`` summarizes the number of
        steps run and the stop reason, with the executed steps recorded in
        ``memory_writes`` for audit. ``confidence`` is ``1.0`` on a clean
        termination/completion and lower when the cap had to engage (a capped run
        did not necessarily achieve the job's goal).
        """
        job = self._job_from_state(state)
        executed, reason = self._execute(job)
        confidence = 0.5 if reason is StopReason.MAX_STEPS else 1.0
        return AgentResult(
            output=self._summarize(executed, reason, job),
            tool_calls_made=[],
            memory_writes=list(executed),
            confidence=confidence,
        )
