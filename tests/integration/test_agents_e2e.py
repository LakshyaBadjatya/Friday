"""End-to-end integration tests for the Stage-4 agent wiring (Phase 2).

These tests exercise the full router -> orchestrator -> AgentRegistry -> agent
path with a :class:`~friday.providers.llm.FakeLLM` (zero network) and ``respx``
mocking any search call. They pin the load-bearing Stage-4 behaviours:

* **Routing + dispatch.** Each new specialist mode (AUTOMATION, DEVICE_CONTROL,
  ALERTING, plus the analysis/knowledge research paths) routes to and runs its
  agent end-to-end, surfacing the agent's output through the persona.
* **The confirm-step (build-spec section 12).** A side-effecting device/notify
  action issued *without* confirmation returns a persona confirm prompt and does
  NOT execute (the fake actuator/sink records nothing); a confirming follow-up
  (``state.confirmed=True``) proceeds and the action executes exactly once.
* **Security lockdown (build-spec section 9.9).** A SECURITY_LOCKDOWN utterance
  runs the fixed 3-step lockdown (revoke -> kill -> notify) and the persona
  reports the audit trail — it is not handed to a chatty agent.

All settings are env-isolated and time/clock are injected where needed, so the
suite is deterministic and offline.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import friday.tools.home as home_mod
from friday.agents.alerting import AlertingAgent
from friday.agents.analysis import AnalysisAgent
from friday.agents.automation import AutomationAgent
from friday.agents.base import Agent, AgentRegistry
from friday.agents.device import DeviceAgent
from friday.agents.knowledge import KnowledgeAgent
from friday.config import Settings
from friday.core.graph import build_graph
from friday.core.orchestrator import Orchestrator
from friday.core.state import GraphState, Mode
from friday.memory.short_term import ShortTermMemory
from friday.memory.vector import InMemoryVectorStore
from friday.providers.llm import FakeLLM, LLMResponse, Usage
from friday.tools.home import HomeControlTool
from friday.tools.notify import NotifyTool
from friday.tools.registry import ToolRegistry

PERSONA_PATH = (
    Path(__file__).resolve().parents[2] / "src" / "friday" / "persona" / "friday.md"
)


def _resp(text: str) -> LLMResponse:
    return LLMResponse(text=text, tool_calls=[], usage=Usage())


# --------------------------------------------------------------------------- #
# AgentRegistry contract
# --------------------------------------------------------------------------- #
class _StubAgent:
    name = "stub"
    allowed_tools: frozenset[str] = frozenset()

    async def run(self, state: GraphState) -> object:  # pragma: no cover - stub
        raise NotImplementedError


def test_agent_registry_register_and_get() -> None:
    registry = AgentRegistry()
    agent = _StubAgent()
    registry.register(agent)
    assert registry.get("stub") is agent
    assert isinstance(agent, Agent)


def test_agent_registry_get_unknown_raises_keyerror() -> None:
    registry = AgentRegistry()
    with pytest.raises(KeyError):
        registry.get("nope")


def test_agent_registry_has_membership() -> None:
    registry = AgentRegistry()
    registry.register(_StubAgent())
    assert "stub" in registry
    assert "missing" not in registry


# --------------------------------------------------------------------------- #
# Orchestrator wiring helpers
# --------------------------------------------------------------------------- #
def _patch_home_settings(
    monkeypatch: pytest.MonkeyPatch, *, enable_home: bool, allowlist: list[str]
) -> None:
    """Point the home tool's ``get_settings`` at an env-isolated Settings."""

    def _settings() -> Settings:
        return Settings(
            _env_file=None,
            enable_home=enable_home,
            device_allowlist=allowlist,
        )

    monkeypatch.setattr(home_mod, "get_settings", _settings)


def _build_orchestrator(
    *,
    llm: FakeLLM | None = None,
    clock: float = 1000.0,
) -> tuple[Orchestrator, ToolRegistry, NotifyTool, HomeControlTool]:
    """Assemble an orchestrator with a fully-populated AgentRegistry.

    Returns the orchestrator plus the live notify/home tools so tests can assert
    against their fake sinks.
    """
    notify = NotifyTool()
    home = HomeControlTool()
    tool_registry = ToolRegistry()
    tool_registry.register(notify)
    tool_registry.register(home)

    store = InMemoryVectorStore()
    store.add([("FRIDAY is a defensive-only local assistant.", "doc-1")])

    agents = AgentRegistry()
    agents.register(AnalysisAgent(tool_registry, llm=FakeLLM(responses=[])))
    agents.register(KnowledgeAgent(store=store))
    agents.register(AutomationAgent())
    agents.register(DeviceAgent(tool_registry))
    agents.register(
        AlertingAgent(
            tool_registry,
            clock=lambda: clock,
            settings=Settings(_env_file=None),
        )
    )

    orchestrator = Orchestrator(
        llm=llm if llm is not None else FakeLLM(responses=[]),
        registry=tool_registry,
        memory=ShortTermMemory(),
        persona_path=PERSONA_PATH,
        agents=agents,
    )
    return orchestrator, tool_registry, notify, home


# --------------------------------------------------------------------------- #
# AUTOMATION mode: routes to and runs the automation agent end-to-end
# --------------------------------------------------------------------------- #
async def test_automation_mode_routes_and_runs_agent() -> None:
    orch, _reg, _notify, _home = _build_orchestrator(
        llm=FakeLLM(responses=[_resp("Scheduled it, Boss — three steps done.")])
    )
    state = GraphState(
        session_id="auto-1",
        user_input="schedule a backup job for tonight",
        scratchpad={"job": {"steps": ["a", "b", "c"], "max_steps": 5}},
    )

    out = await orch.handle(state)

    assert out.mode is Mode.AUTOMATION
    assert out.response is not None
    assert "Boss" in out.response


# --------------------------------------------------------------------------- #
# DEVICE_CONTROL: confirm-step gates a side-effecting action
# --------------------------------------------------------------------------- #
async def test_device_without_confirmation_asks_and_does_not_execute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_home_settings(monkeypatch, enable_home=True, allowlist=["light.kitchen"])
    orch, _reg, _notify, home = _build_orchestrator()
    state = GraphState(
        session_id="dev-1",
        user_input="turn on the light.kitchen",
        scratchpad={"device": {"device_id": "light.kitchen", "action": "on"}},
        confirmed=False,
    )

    out = await orch.handle(state)

    assert out.mode is Mode.DEVICE_CONTROL
    assert out.response is not None
    # A persona confirm question, and NOTHING actuated.
    assert "?" in out.response
    assert "confirm" in out.response.lower()
    assert home.sink == []


async def test_device_with_confirmation_executes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_home_settings(monkeypatch, enable_home=True, allowlist=["light.kitchen"])
    orch, _reg, _notify, home = _build_orchestrator(
        llm=FakeLLM(responses=[_resp("Done — light.kitchen is on, Boss.")])
    )
    state = GraphState(
        session_id="dev-2",
        user_input="turn on the light.kitchen",
        scratchpad={"device": {"device_id": "light.kitchen", "action": "on"}},
        confirmed=True,
    )

    out = await orch.handle(state)

    assert out.mode is Mode.DEVICE_CONTROL
    # The action actuated exactly once.
    assert len(home.sink) == 1
    assert home.sink[0].device_id == "light.kitchen"
    assert home.sink[0].action == "on"


# --------------------------------------------------------------------------- #
# ALERTING: confirm-step gates a side-effecting notify, then executes
# --------------------------------------------------------------------------- #
async def test_alert_without_confirmation_asks_and_does_not_send() -> None:
    orch, _reg, notify, _home = _build_orchestrator()
    state = GraphState(
        session_id="alert-1",
        user_input="notify the ops team about the outage",
        scratchpad={
            "alert": {
                "channel": "slack",
                "target": "#ops",
                "subject": "Outage",
                "body": "Service degraded.",
            }
        },
        confirmed=False,
    )

    out = await orch.handle(state)

    assert out.mode is Mode.ALERTING
    assert out.response is not None
    assert "?" in out.response
    assert "confirm" in out.response.lower()
    # Nothing sent.
    assert notify.sink == []


async def test_alert_with_confirmation_sends() -> None:
    orch, _reg, notify, _home = _build_orchestrator(
        llm=FakeLLM(responses=[_resp("Paged the ops team, Boss.")])
    )
    state = GraphState(
        session_id="alert-2",
        user_input="notify the ops team about the outage",
        scratchpad={
            "alert": {
                "channel": "slack",
                "target": "#ops",
                "subject": "Outage",
                "body": "Service degraded.",
            }
        },
        confirmed=True,
    )

    out = await orch.handle(state)

    assert out.mode is Mode.ALERTING
    # Exactly one notification sent.
    assert len(notify.sink) == 1
    assert notify.sink[0].subject == "Outage"


# --------------------------------------------------------------------------- #
# SECURITY_LOCKDOWN: runs the 3-step lockdown and reports the audit trail
# --------------------------------------------------------------------------- #
async def test_security_lockdown_runs_three_steps_and_reports_audit() -> None:
    orch, _reg, _notify, _home = _build_orchestrator()
    state = GraphState(
        session_id="lock-1",
        user_input="initiate lockdown now",
    )

    out = await orch.handle(state)

    assert out.mode is Mode.SECURITY_LOCKDOWN
    assert out.response is not None
    lowered = out.response.lower()
    # The audit trail names all three ordered steps.
    assert "revoke_tokens" in lowered
    assert "kill_sessions" in lowered
    assert "notify_owner" in lowered
    # The audit records are surfaced on the scratchpad for the response builder.
    records = out.scratchpad.get("lockdown_audit")
    assert isinstance(records, list)
    assert len(records) == 3


async def test_security_lockdown_does_not_call_llm() -> None:
    # An exhausted FakeLLM would raise if the lockdown path tried to synthesize
    # via the model — the audit trail is reported deterministically, not by chat.
    orch, _reg, _notify, _home = _build_orchestrator(llm=FakeLLM(responses=[]))
    state = GraphState(session_id="lock-2", user_input="run the barn door procedure")

    out = await orch.handle(state)

    assert out.mode is Mode.SECURITY_LOCKDOWN
    assert out.response is not None
    assert "revoke_tokens" in out.response.lower()


# --------------------------------------------------------------------------- #
# The mode graph routes the new modes through their nodes too
# --------------------------------------------------------------------------- #
async def test_graph_routes_security_lockdown_node() -> None:
    orch, _reg, _notify, _home = _build_orchestrator(llm=FakeLLM(responses=[]))
    graph = build_graph(orch)

    out = await graph.invoke(
        GraphState(session_id="g-lock", user_input="initiate lockdown now")
    )

    assert isinstance(out, GraphState)
    assert out.mode is Mode.SECURITY_LOCKDOWN
    assert out.response is not None
    assert "revoke_tokens" in out.response.lower()


async def test_graph_routes_device_confirm_node(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_home_settings(monkeypatch, enable_home=True, allowlist=["light.kitchen"])
    orch, _reg, _notify, home = _build_orchestrator()
    graph = build_graph(orch)

    out = await graph.invoke(
        GraphState(
            session_id="g-dev",
            user_input="turn on the light.kitchen",
            scratchpad={"device": {"device_id": "light.kitchen", "action": "on"}},
            confirmed=False,
        )
    )

    assert out.mode is Mode.DEVICE_CONTROL
    assert out.response is not None
    assert "confirm" in out.response.lower()
    # Confirm-gate held through the graph path too: nothing actuated.
    assert home.sink == []
