"""Unit tests for :class:`friday.protocols.runner.ProtocolRunner` (Tier 1).

Offline, with fake tools registered in a real
:class:`~friday.tools.registry.ToolRegistry` so the runner exercises the exact
``execute`` contract — permission gate, args validation, and the confirm-step —
the production path uses. No network, no LLM.

Pinned behaviours:

* Steps execute in registry order; each step's outcome is reported.
* A side-effecting, non-idempotent step with ``confirmed=False`` STOPS the run
  before that step (``ran=False``, ``needs_confirmation=True``); later steps do
  NOT run.
* ``confirmed=True`` runs every step (no pause).
* A step that fails (``ok=False``) stops the run and reports it; later steps do
  NOT run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

import friday.config as config
import friday.core.orchestrator as orch_mod
from friday.core.orchestrator import Orchestrator
from friday.core.state import GraphState, Mode
from friday.memory.short_term import ShortTermMemory
from friday.protocols.runner import ProtocolRunner
from friday.protocols.store import Protocol, ProtocolStep, SQLiteProtocolStore
from friday.providers.llm import FakeLLM
from friday.tools.base import ToolError, ToolResult
from friday.tools.registry import ToolRegistry

PERSONA_PATH = (
    Path(__file__).resolve().parents[2] / "src" / "friday" / "persona" / "friday.md"
)


class _Args(BaseModel):
    """Permissive args model: accepts whatever the step supplies."""

    model_config = {"extra": "allow"}


class _RecordingTool:
    """A read-only fake tool that records each invocation order-wise."""

    def __init__(self, name: str, log: list[str], *, fail: bool = False) -> None:
        self.name = name
        self.description = f"fake {name}"
        self.args_model: type[BaseModel] = _Args
        self.required_permission = "none"
        self.idempotent = True
        self.side_effecting = False
        self._log = log
        self._fail = fail

    async def __call__(self, args: Any) -> ToolResult:
        self._log.append(self.name)
        if self._fail:
            return ToolResult(
                ok=False,
                error=ToolError(code="boom", message="step failed", retriable=False),
            )
        return ToolResult(ok=True, data={"tool": self.name})


class _SideEffectingTool(_RecordingTool):
    """A side-effecting, non-idempotent fake (gated by the confirm-step)."""

    def __init__(self, name: str, log: list[str]) -> None:
        super().__init__(name, log)
        self.idempotent = False
        self.side_effecting = True


def _registry(*tools: _RecordingTool) -> tuple[ToolRegistry, frozenset[str]]:
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)  # type: ignore[arg-type]
    return registry, frozenset(t.name for t in tools)


def _protocol(*step_tools: str) -> Protocol:
    return Protocol(
        id=1,
        name="P",
        trigger_phrase="p",
        steps=[ProtocolStep(tool=t, args={}) for t in step_tools],
        enabled=True,
    )


# --------------------------------------------------------------------------- #
# order
# --------------------------------------------------------------------------- #
async def test_runs_steps_in_order() -> None:
    log: list[str] = []
    a = _RecordingTool("a", log)
    b = _RecordingTool("b", log)
    c = _RecordingTool("c", log)
    registry, allowed = _registry(a, b, c)
    runner = ProtocolRunner(registry, allowed)

    result = await runner.run(_protocol("a", "b", "c"))

    assert log == ["a", "b", "c"]
    assert result.ran is True
    assert result.needs_confirmation is False
    assert [s.tool for s in result.steps] == ["a", "b", "c"]
    assert all(s.ok for s in result.steps)
    assert result.protocol == "P"


# --------------------------------------------------------------------------- #
# confirm-step
# --------------------------------------------------------------------------- #
async def test_unconfirmed_side_effecting_step_stops_run() -> None:
    log: list[str] = []
    a = _RecordingTool("a", log)
    danger = _SideEffectingTool("danger", log)
    after = _RecordingTool("after", log)
    registry, allowed = _registry(a, danger, after)
    runner = ProtocolRunner(registry, allowed)

    result = await runner.run(_protocol("a", "danger", "after"))

    # The read-only first step ran; the side-effecting step did NOT execute, and
    # nothing after it ran either.
    assert log == ["a"]
    assert result.ran is False
    assert result.needs_confirmation is True
    # Steps reported so far: the completed "a" and the paused "danger".
    assert [s.tool for s in result.steps] == ["a", "danger"]
    danger_outcome = result.steps[-1]
    assert danger_outcome.needs_confirmation is True
    assert danger_outcome.ok is False


async def test_confirmed_runs_all_side_effecting_steps() -> None:
    log: list[str] = []
    a = _RecordingTool("a", log)
    danger = _SideEffectingTool("danger", log)
    after = _RecordingTool("after", log)
    registry, allowed = _registry(a, danger, after)
    runner = ProtocolRunner(registry, allowed)

    result = await runner.run(_protocol("a", "danger", "after"), confirmed=True)

    assert log == ["a", "danger", "after"]
    assert result.ran is True
    assert result.needs_confirmation is False
    assert all(s.ok for s in result.steps)


# --------------------------------------------------------------------------- #
# step error stops the run
# --------------------------------------------------------------------------- #
async def test_step_error_stops_run() -> None:
    log: list[str] = []
    a = _RecordingTool("a", log)
    boom = _RecordingTool("boom", log, fail=True)
    after = _RecordingTool("after", log)
    registry, allowed = _registry(a, boom, after)
    runner = ProtocolRunner(registry, allowed)

    result = await runner.run(_protocol("a", "boom", "after"))

    # "boom" was invoked (it ran and returned ok=False); "after" never ran.
    assert log == ["a", "boom"]
    assert result.ran is False
    assert result.needs_confirmation is False
    assert [s.tool for s in result.steps] == ["a", "boom"]
    boom_outcome = result.steps[-1]
    assert boom_outcome.ok is False
    assert boom_outcome.error == "boom"


async def test_unpermitted_tool_is_reported_as_error() -> None:
    """A step naming a tool outside ``allowed_tools`` stops the run honestly."""
    log: list[str] = []
    a = _RecordingTool("a", log)
    registry, _allowed = _registry(a)
    # Only "a" is allowed; the protocol references an unknown tool.
    runner = ProtocolRunner(registry, frozenset({"a"}))

    result = await runner.run(_protocol("a", "ghost"))

    assert log == ["a"]
    assert result.ran is False
    assert result.needs_confirmation is False
    assert [s.tool for s in result.steps] == ["a", "ghost"]
    assert result.steps[-1].ok is False
    assert result.steps[-1].error is not None


# --------------------------------------------------------------------------- #
# orchestrator trigger-phrase hook
# --------------------------------------------------------------------------- #
def _enable_protocols(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force ``enable_protocols`` on for both config + orchestrator modules."""
    settings = config.Settings(_env_file=None, enable_protocols=True)
    monkeypatch.setattr(config, "get_settings", lambda: settings)
    monkeypatch.setattr(orch_mod, "get_settings", lambda: settings)


def _orchestrator(
    registry: ToolRegistry, store: SQLiteProtocolStore, runner: ProtocolRunner
) -> Orchestrator:
    # The protocol hook short-circuits before routing, so the LLM is never
    # consumed; an empty script proves that.
    return Orchestrator(
        llm=FakeLLM(responses=[]),
        registry=registry,
        memory=ShortTermMemory(),
        persona_path=PERSONA_PATH,
        protocol_store=store,
        protocol_runner=runner,
    )


async def test_orchestrator_runs_protocol_on_trigger_phrase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_protocols(monkeypatch)
    log: list[str] = []
    a = _RecordingTool("a", log)
    b = _RecordingTool("b", log)
    registry, allowed = _registry(a, b)
    runner = ProtocolRunner(registry, allowed)

    store = SQLiteProtocolStore(":memory:")
    store.add(
        name="Goodnight",
        trigger_phrase="goodnight",
        steps=[ProtocolStep(tool="a", args={}), ProtocolStep(tool="b", args={})],
    )
    orch = _orchestrator(registry, store, runner)

    state = GraphState(session_id="s1", user_input="goodnight friday")
    out = await orch.handle(state)

    # The protocol fired its steps in order, short-circuiting routing.
    assert log == ["a", "b"]
    assert out.scratchpad.get("protocol") == "Goodnight"
    assert out.response is not None
    assert "Goodnight" in out.response


async def test_orchestrator_runs_protocol_on_run_the_x_protocol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_protocols(monkeypatch)
    log: list[str] = []
    a = _RecordingTool("a", log)
    registry, allowed = _registry(a)
    runner = ProtocolRunner(registry, allowed)

    store = SQLiteProtocolStore(":memory:")
    store.add(
        name="Goodnight",
        trigger_phrase="zzz never matches this",
        steps=[ProtocolStep(tool="a", args={})],
    )
    orch = _orchestrator(registry, store, runner)

    state = GraphState(session_id="s2", user_input="run the Goodnight protocol")
    out = await orch.handle(state)

    assert log == ["a"]
    assert out.scratchpad.get("protocol") == "Goodnight"


async def test_orchestrator_protocol_surfaces_confirm_question(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_protocols(monkeypatch)
    log: list[str] = []
    danger = _SideEffectingTool("danger", log)
    registry, allowed = _registry(danger)
    runner = ProtocolRunner(registry, allowed)

    store = SQLiteProtocolStore(":memory:")
    store.add(
        name="Risky",
        trigger_phrase="risky",
        steps=[ProtocolStep(tool="danger", args={})],
    )
    orch = _orchestrator(registry, store, runner)

    # Unconfirmed: the side-effecting step does NOT run; a confirm question
    # surfaces.
    state = GraphState(session_id="s3", user_input="risky now")
    out = await orch.handle(state)
    assert log == []
    assert out.response is not None
    assert "confirm" in out.response.lower()

    # Confirmed follow-up re-fires and runs the side-effecting step.
    confirmed = GraphState(
        session_id="s3", user_input="risky now", confirmed=True
    )
    out2 = await orch.handle(confirmed)
    assert log == ["danger"]
    assert out2.mode is Mode.AUTOMATION


async def test_orchestrator_protocol_hook_inert_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Flag off (default): the hook must not fire even if a trigger phrase matches.
    settings = config.Settings(_env_file=None, enable_protocols=False)
    monkeypatch.setattr(config, "get_settings", lambda: settings)
    monkeypatch.setattr(orch_mod, "get_settings", lambda: settings)

    log: list[str] = []
    a = _RecordingTool("a", log)
    registry, allowed = _registry(a)
    runner = ProtocolRunner(registry, allowed)
    store = SQLiteProtocolStore(":memory:")
    store.add(
        name="Goodnight",
        trigger_phrase="goodnight",
        steps=[ProtocolStep(tool="a", args={})],
    )
    orch = _orchestrator(registry, store, runner)

    state = GraphState(session_id="s4", user_input="goodnight")
    out = await orch.handle(state)

    # No protocol fired; it fell through to normal routing.
    assert log == []
    assert out.scratchpad.get("protocol") is None
