"""Unit tests for :class:`friday.tools.home.HomeControlTool`.

The home tool is side-effecting and non-idempotent. It actuates a device only
when the ``enable_home`` feature flag is on AND the ``device_id`` is in the
``device_allowlist``; otherwise it returns a typed refusal. The real Home
Assistant adapter is present but flagged off (``NotImplementedError``). No
network is touched.

Settings are injected by monkeypatching the module-level ``get_settings`` name
the tool imports, following the repo's router-test convention.
"""

from __future__ import annotations

import pytest

import friday.tools.home as home_mod
from friday.config import Settings
from friday.tools.base import ToolResult
from friday.tools.home import HomeArgs, HomeAssistantActuator, HomeControlTool


def _patch_settings(
    monkeypatch: pytest.MonkeyPatch, *, enable_home: bool, allowlist: list[str]
) -> None:
    def _settings() -> Settings:
        return Settings(
            _env_file=None,
            enable_home=enable_home,
            device_allowlist=allowlist,
        )

    monkeypatch.setattr(home_mod, "get_settings", _settings)


def test_home_tool_attrs() -> None:
    tool = HomeControlTool()
    assert tool.name == "home"
    assert tool.side_effecting is True
    assert tool.idempotent is False
    assert tool.args_model is HomeArgs


async def test_home_disabled_when_flag_off(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_settings(monkeypatch, enable_home=False, allowlist=["light.kitchen"])
    tool = HomeControlTool()
    result = await tool(HomeArgs(device_id="light.kitchen", action="on"))

    assert isinstance(result, ToolResult)
    assert result.ok is False
    assert result.error is not None
    assert result.error.code == "home_disabled"
    # No actuation recorded.
    assert tool.sink == []


async def test_home_device_not_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_settings(monkeypatch, enable_home=True, allowlist=["light.kitchen"])
    tool = HomeControlTool()
    result = await tool(HomeArgs(device_id="switch.unknown", action="off"))

    assert result.ok is False
    assert result.error is not None
    assert result.error.code == "device_not_allowed"
    assert tool.sink == []


async def test_home_success_for_allowlisted_id(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_settings(monkeypatch, enable_home=True, allowlist=["light.kitchen"])
    tool = HomeControlTool()
    result = await tool(HomeArgs(device_id="light.kitchen", action="on"))

    assert result.ok is True
    assert result.error is None
    assert result.data == {"device_id": "light.kitchen", "action": "on"}

    # The FAKE actuator recorded the action to the in-memory sink.
    assert len(tool.sink) == 1
    recorded = tool.sink[0]
    assert recorded.device_id == "light.kitchen"
    assert recorded.action == "on"


async def test_home_disabled_takes_precedence_over_allowlist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Flag off + allowlisted device still refuses with home_disabled (flag first).
    _patch_settings(monkeypatch, enable_home=False, allowlist=["light.kitchen"])
    tool = HomeControlTool()
    result = await tool(HomeArgs(device_id="light.kitchen", action="on"))
    assert result.ok is False
    assert result.error is not None
    assert result.error.code == "home_disabled"


async def test_home_coerces_raw_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_settings(monkeypatch, enable_home=True, allowlist=["plug.1"])
    tool = HomeControlTool()
    result = await tool({"device_id": "plug.1", "action": "toggle"})
    assert result.ok is True
    assert result.data == {"device_id": "plug.1", "action": "toggle"}


async def test_home_assistant_adapter_not_implemented() -> None:
    # The real adapter is present but flagged off until Phase 4+.
    adapter = HomeAssistantActuator()
    with pytest.raises(NotImplementedError):
        await adapter.actuate("light.kitchen", "on")
