"""Unit tests for :class:`friday.tools.capabilities.CapabilitiesTool`.

The tool is exercised with a **fake** registry that yields a handful of fake
tools, so the tests are isolated and deterministic. They pin the protocol
attributes, the structured capability map (name/description/side_effecting per
tool), the deterministic ordering, an empty-registry case, and a round-trip
through a real :class:`ToolRegistry`.
"""

from __future__ import annotations

from typing import Any, cast

from pydantic import BaseModel

from friday.tools.base import Tool, ToolResult
from friday.tools.capabilities import (
    CapabilitiesArgs,
    CapabilitiesTool,
    ToolLister,
)
from friday.tools.registry import ToolRegistry


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class _NoArgs(BaseModel):
    pass


class _FakeTool:
    """A minimal Tool-shaped object with configurable capability fields."""

    def __init__(
        self,
        name: str,
        description: str,
        *,
        side_effecting: bool,
        idempotent: bool = True,
        required_permission: str = "x",
    ) -> None:
        self.name = name
        self.description = description
        self.args_model = _NoArgs
        self.required_permission = required_permission
        self.idempotent = idempotent
        self.side_effecting = side_effecting

    async def __call__(self, args: Any) -> ToolResult:  # pragma: no cover
        return ToolResult(ok=True)


class FakeRegistry:
    """A fake registry that simply yields a fixed list of tools."""

    def __init__(self, tools: list[Tool]) -> None:
        self._tools = tools

    def iter_tools(self) -> list[Tool]:
        return list(self._tools)


def _registry(tools: list[Tool]) -> ToolLister:
    return FakeRegistry(tools)


# --------------------------------------------------------------------------- #
# Attributes
# --------------------------------------------------------------------------- #
def test_capabilities_tool_attrs() -> None:
    tool = CapabilitiesTool(_registry([]))
    assert isinstance(tool, Tool)
    assert tool.name == "capabilities"
    assert tool.args_model is CapabilitiesArgs
    assert tool.side_effecting is False
    assert tool.idempotent is True


# --------------------------------------------------------------------------- #
# Map assembly
# --------------------------------------------------------------------------- #
async def test_capabilities_reports_each_tool() -> None:
    registry = _registry(
        [
            _FakeTool("web_search", "search the web", side_effecting=False),
            _FakeTool("notify", "send a notification", side_effecting=True),
        ]
    )
    tool = CapabilitiesTool(registry)
    result = await tool(CapabilitiesArgs())

    assert isinstance(result, ToolResult)
    assert result.ok is True
    assert result.data["count"] == 2

    by_name = {entry["name"]: entry for entry in result.data["tools"]}
    assert by_name["web_search"]["description"] == "search the web"
    assert by_name["web_search"]["side_effecting"] is False
    assert by_name["notify"]["side_effecting"] is True
    # Completeness fields also surface.
    assert "idempotent" in by_name["web_search"]
    assert "required_permission" in by_name["web_search"]


async def test_capabilities_is_sorted_by_name() -> None:
    registry = _registry(
        [
            _FakeTool("zebra", "z", side_effecting=False),
            _FakeTool("alpha", "a", side_effecting=False),
            _FakeTool("mango", "m", side_effecting=False),
        ]
    )
    tool = CapabilitiesTool(registry)
    result = await tool(CapabilitiesArgs())
    names = [entry["name"] for entry in result.data["tools"]]
    assert names == ["alpha", "mango", "zebra"]


async def test_capabilities_empty_registry() -> None:
    tool = CapabilitiesTool(_registry([]))
    result = await tool(CapabilitiesArgs())
    assert result.ok is True
    assert result.data == {"tools": [], "count": 0}


async def test_capabilities_accepts_raw_dict_args() -> None:
    # The registry passes a validated model, but calling with {} must also work.
    tool = CapabilitiesTool(
        _registry([_FakeTool("a", "x", side_effecting=False)])
    )
    result = await tool({})
    assert result.ok is True
    assert result.data["count"] == 1


# --------------------------------------------------------------------------- #
# Round-trip through the real registry (via the fake lister it wraps)
# --------------------------------------------------------------------------- #
async def test_capabilities_round_trip_via_registry() -> None:
    lister = _registry(
        [_FakeTool("home", "control home devices", side_effecting=True)]
    )
    registry = ToolRegistry()
    registry.register(cast(Tool, CapabilitiesTool(lister)))

    result = await registry.execute(
        "capabilities", {}, allowed_tools={"capabilities"}
    )
    assert result.ok is True
    assert result.data["count"] == 1
    assert result.data["tools"][0]["name"] == "home"
    assert result.data["tools"][0]["side_effecting"] is True
