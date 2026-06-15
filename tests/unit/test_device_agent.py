"""Unit tests for the device agent (Stage 3, build-spec section 9.7).

The :class:`~friday.agents.device.DeviceAgent` routes a device-control action
through the ``home`` tool via the injected
:class:`~friday.tools.registry.ToolRegistry`, honouring the registry's
confirm-step (build-spec section 12): the ``home`` tool is side-effecting and
non-idempotent, so it only executes when ``state.confirmed`` is ``True``.

The pinned behaviours:

* **Protocol contract.** The agent satisfies the
  :class:`~friday.agents.base.Agent` protocol with ``name == "device"`` and
  ``allowed_tools == frozenset({"home"})``.
* **Refusal, never fabrication.** A ``device_id`` that is not on the allow-list
  makes the ``home`` tool return ``device_not_allowed``; the agent SURFACES that
  refusal in its :class:`~friday.agents.base.AgentResult` (no fabricated
  success, the fake actuator records nothing).
* **Confirmation gate.** An allow-listed device action with ``confirmed`` still
  ``False`` is gated by the registry confirm-step (``confirmation_required``);
  the tool never actuates.
* **Happy path.** An allow-listed device + ``enable_home`` on + ``confirmed`` ==
  ``True`` actually invokes the ``home`` tool (the fake actuator's sink records
  the action) and the agent reports success.

Settings are injected by monkeypatching the module-level ``get_settings`` the
``home`` tool imports, following the repo's home-tool / router test convention.
No network, no LLM, no wall-clock.
"""

from __future__ import annotations

import pytest

import friday.tools.home as home_mod
from friday.agents.base import Agent, AgentResult
from friday.agents.device import DeviceAgent
from friday.config import Settings
from friday.core.state import GraphState, Mode
from friday.tools.home import HomeControlTool
from friday.tools.registry import ToolRegistry


def _patch_settings(
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


def _registry_with_home() -> tuple[ToolRegistry, HomeControlTool]:
    """A registry with a fresh :class:`HomeControlTool` registered."""
    registry = ToolRegistry()
    home = HomeControlTool()
    registry.register(home)
    return registry, home


def _state(
    *,
    device_id: str,
    action: str,
    confirmed: bool,
    session_id: str = "device-test",
) -> GraphState:
    """Build a graph state carrying the device action in the scratchpad."""
    return GraphState(
        session_id=session_id,
        mode=Mode.DEVICE_CONTROL,
        user_input=f"{action} {device_id}",
        scratchpad={"device": {"device_id": device_id, "action": action}},
        confirmed=confirmed,
    )


# --------------------------------------------------------------------------- #
# Protocol + contract
# --------------------------------------------------------------------------- #
def test_device_agent_satisfies_agent_protocol() -> None:
    registry, _home = _registry_with_home()
    agent = DeviceAgent(registry)
    assert isinstance(agent, Agent)
    assert agent.name == "device"
    assert agent.allowed_tools == frozenset({"home"})


# --------------------------------------------------------------------------- #
# Refusal: non-allowlisted device -> surfaced refusal, never fabricated success
# --------------------------------------------------------------------------- #
async def test_non_allowlisted_device_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # enable_home on, but the requested device is NOT on the allow-list, so the
    # home tool returns device_not_allowed and the agent must surface it.
    _patch_settings(monkeypatch, enable_home=True, allowlist=["light.kitchen"])
    registry, home = _registry_with_home()
    agent = DeviceAgent(registry)

    result = await agent.run(
        _state(device_id="switch.unknown", action="off", confirmed=True)
    )

    assert isinstance(result, AgentResult)
    # No fabricated success: the fake actuator recorded nothing.
    assert home.sink == []
    # The refusal is surfaced honestly (low confidence, names the cause).
    lowered = result.output.lower()
    assert "device_not_allowed" in lowered or "not allow" in lowered
    assert result.confidence < 1.0


# --------------------------------------------------------------------------- #
# Confirmation gate: allow-listed but unconfirmed -> registry gates it
# --------------------------------------------------------------------------- #
async def test_allowlisted_but_unconfirmed_requires_confirmation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_settings(monkeypatch, enable_home=True, allowlist=["light.kitchen"])
    registry, home = _registry_with_home()
    agent = DeviceAgent(registry)

    result = await agent.run(
        _state(device_id="light.kitchen", action="on", confirmed=False)
    )

    # The registry confirm-step blocked execution: nothing actuated.
    assert home.sink == []
    lowered = result.output.lower()
    assert "confirm" in lowered


# --------------------------------------------------------------------------- #
# Happy path: allow-listed + enable_home + confirmed -> tool actually invoked
# --------------------------------------------------------------------------- #
async def test_allowlisted_confirmed_invokes_home_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_settings(monkeypatch, enable_home=True, allowlist=["light.kitchen"])
    registry, home = _registry_with_home()
    agent = DeviceAgent(registry)

    result = await agent.run(
        _state(device_id="light.kitchen", action="on", confirmed=True)
    )

    # The FAKE actuator recorded the action: the tool actually ran.
    assert len(home.sink) == 1
    recorded = home.sink[0]
    assert recorded.device_id == "light.kitchen"
    assert recorded.action == "on"

    # The agent reports success and audits the issued tool call.
    assert "light.kitchen" in result.output
    assert any(call.name == "home" for call in result.tool_calls_made)
    assert result.confidence == 1.0


# --------------------------------------------------------------------------- #
# Flag off: enable_home off -> home tool refuses even an allow-listed device
# --------------------------------------------------------------------------- #
async def test_home_disabled_flag_refuses_even_allowlisted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_settings(monkeypatch, enable_home=False, allowlist=["light.kitchen"])
    registry, home = _registry_with_home()
    agent = DeviceAgent(registry)

    result = await agent.run(
        _state(device_id="light.kitchen", action="on", confirmed=True)
    )

    assert home.sink == []
    lowered = result.output.lower()
    assert "home_disabled" in lowered or "disabled" in lowered
    assert result.confidence < 1.0


# --------------------------------------------------------------------------- #
# Missing action: no device staged -> honest refusal, nothing actuated
# --------------------------------------------------------------------------- #
async def test_missing_device_action_refuses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_settings(monkeypatch, enable_home=True, allowlist=["light.kitchen"])
    registry, home = _registry_with_home()
    agent = DeviceAgent(registry)

    state = GraphState(
        session_id="device-test",
        mode=Mode.DEVICE_CONTROL,
        user_input="do something",
        scratchpad={},
        confirmed=True,
    )
    result = await agent.run(state)

    assert home.sink == []
    assert result.confidence < 1.0
    assert result.output.strip() != ""
