"""Unit tests for the automation agent (Stage 3, build-spec section 9.7).

The :class:`~friday.agents.automation.AutomationAgent` executes a multi-step
:class:`~friday.agents.automation.Job` by stepping through ``job.steps`` until a
*termination predicate* fires or a hard ``max_steps`` cap is reached — whichever
comes first. The cap is the safety rail: it is the structural guarantee that a
runaway job (one whose termination predicate never fires) cannot loop forever.

The pinned behaviours:

* **Runaway -> capped.** A job whose termination predicate never fires runs
  EXACTLY ``max_steps`` steps and then stops, with the result attributing the
  stop to the cap (proves no infinite loop).
* **Termination -> early stop.** A predicate that fires mid-run halts the
  executor before the cap, and the result attributes the stop to the predicate.
* **Exhaustion -> clean finish.** A job with fewer steps than the cap and no
  firing predicate runs every step and reports that it completed all of them.

All execution is deterministic and offline: no network, no model, no
wall-clock. The executor is a pure loop over an in-memory ``Job``.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from friday.agents.automation import AutomationAgent, Job
from friday.agents.base import Agent, AgentResult
from friday.core.state import GraphState, Mode


def _state(job: Job, session_id: str = "auto-test") -> GraphState:
    """Build a graph state carrying ``job`` in the scratchpad for the agent."""
    return GraphState(
        session_id=session_id,
        mode=Mode.AUTOMATION,
        user_input="run the automation",
        scratchpad={"job": job.model_dump()},
    )


# --------------------------------------------------------------------------- #
# Protocol + model contract
# --------------------------------------------------------------------------- #
def test_automation_agent_satisfies_agent_protocol() -> None:
    agent = AutomationAgent()
    assert isinstance(agent, Agent)
    assert agent.name == "automation"
    # allowed_tools is a frozenset of tool names (the Agent contract).
    assert isinstance(agent.allowed_tools, frozenset)


def test_job_requires_positive_max_steps() -> None:
    # A non-positive cap would defeat the runaway guard, so it is rejected.
    with pytest.raises(ValidationError):
        Job(steps=["a", "b"], max_steps=0)
    with pytest.raises(ValidationError):
        Job(steps=["a"], max_steps=-3)


def test_job_round_trips_through_model_dump() -> None:
    job = Job(steps=["one", "two"], max_steps=5)
    restored = Job.model_validate(job.model_dump())
    assert restored == job


# --------------------------------------------------------------------------- #
# The runaway guarantee (the headline test)
# --------------------------------------------------------------------------- #
async def test_runaway_job_stops_at_max_steps() -> None:
    # A predicate that NEVER fires would loop forever without the cap. With more
    # steps available than the cap allows, the executor must stop at exactly the
    # cap and say so.
    agent = AutomationAgent(terminate=lambda step, index: False)
    job = Job(steps=[f"step-{i}" for i in range(100)], max_steps=7)

    result = await agent.run(_state(job))

    assert isinstance(result, AgentResult)
    # The headline assertion: executed step count == cap (no infinite loop).
    assert result.memory_writes == [f"step-{i}" for i in range(7)]
    assert len(result.memory_writes) == job.max_steps
    # The summary must explain WHY it stopped: the cap, not natural completion.
    lowered = result.output.lower()
    assert "max_steps" in lowered or "cap" in lowered or "limit" in lowered
    assert "7" in result.output


async def test_runaway_with_infinite_step_supplier_still_caps() -> None:
    # Even if there were effectively unbounded steps, the cap bounds the work.
    # Here we hand it far more steps than the cap and a never-firing predicate.
    agent = AutomationAgent(terminate=lambda step, index: False)
    job = Job(steps=["loop"] * 10_000, max_steps=3)

    result = await agent.run(_state(job))

    assert len(result.memory_writes) == 3


# --------------------------------------------------------------------------- #
# Early termination + clean exhaustion
# --------------------------------------------------------------------------- #
async def test_termination_predicate_halts_before_cap() -> None:
    # Predicate fires when it sees the "done" step at index 2: the executor runs
    # steps 0, 1, 2 and then halts — before the much larger cap.
    agent = AutomationAgent(terminate=lambda step, index: step == "done")
    job = Job(steps=["a", "b", "done", "should-not-run"], max_steps=50)

    result = await agent.run(_state(job))

    assert result.memory_writes == ["a", "b", "done"]
    # The "should-not-run" step must never have executed.
    assert "should-not-run" not in result.memory_writes
    lowered = result.output.lower()
    assert "terminat" in lowered or "predicate" in lowered or "stop" in lowered


async def test_job_completes_all_steps_when_under_cap() -> None:
    # No predicate fires and there are fewer steps than the cap: every step runs
    # and the result reports natural completion (not a cap hit).
    agent = AutomationAgent(terminate=lambda step, index: False)
    job = Job(steps=["x", "y", "z"], max_steps=10)

    result = await agent.run(_state(job))

    assert result.memory_writes == ["x", "y", "z"]
    lowered = result.output.lower()
    assert "complet" in lowered or "finished" in lowered or "all" in lowered
    # Did not hit the cap, so the cap language should not dominate the summary.
    assert "3" in result.output


async def test_default_agent_runs_all_steps_to_cap() -> None:
    # With no predicate injected, the default never terminates early, so a job
    # is bounded only by min(len(steps), max_steps).
    agent = AutomationAgent()
    job = Job(steps=["s0", "s1", "s2", "s3", "s4"], max_steps=3)

    result = await agent.run(_state(job))

    assert result.memory_writes == ["s0", "s1", "s2"]
    assert len(result.memory_writes) == 3


# --------------------------------------------------------------------------- #
# Result shape
# --------------------------------------------------------------------------- #
async def test_output_summarizes_steps_run_and_reason() -> None:
    agent = AutomationAgent()
    job = Job(steps=["alpha", "beta"], max_steps=4)

    result = await agent.run(_state(job))

    # The output mentions how many steps ran and is a non-empty human summary.
    assert "2" in result.output
    assert result.output.strip() != ""
    assert result.tool_calls_made == []
    assert 0.0 <= result.confidence <= 1.0


async def test_empty_job_runs_no_steps_and_reports_so() -> None:
    agent = AutomationAgent()
    job = Job(steps=[], max_steps=5)

    result = await agent.run(_state(job))

    assert result.memory_writes == []
    assert "0" in result.output
