"""Unit tests for :class:`friday.tools.registry.ToolRegistry`.

Verifies spec generation from pydantic schemas, permission enforcement
(disallowed tool raises :class:`friday.errors.PermissionError`), and pre-execution
argument validation (bad args are rejected *before* the tool runs).
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from friday.errors import PermissionError as FridayPermissionError
from friday.providers.llm import ToolSpec
from friday.tools.base import ToolResult
from friday.tools.registry import ToolRegistry


class AddArgs(BaseModel):
    """Args for the spying ``add`` tool: two required integers."""

    a: int
    b: int


class SpyAddTool:
    """A tool that records whether it was actually invoked."""

    name = "add"
    description = "Add two integers."
    args_model = AddArgs
    required_permission = "add"
    idempotent = True
    side_effecting = False

    def __init__(self) -> None:
        self.called = False

    async def __call__(self, args: AddArgs) -> ToolResult:
        self.called = True
        return ToolResult(ok=True, data={"sum": args.a + args.b})


def _registry_with_add() -> tuple[ToolRegistry, SpyAddTool]:
    tool = SpyAddTool()
    registry = ToolRegistry()
    registry.register(tool)
    return registry, tool


def test_spec_for_generates_from_args_model() -> None:
    registry, _ = _registry_with_add()
    specs = registry.spec_for(["add"])
    assert len(specs) == 1
    spec = specs[0]
    assert isinstance(spec, ToolSpec)
    assert spec.name == "add"
    assert spec.description == "Add two integers."
    # Parameters are the pydantic JSON schema of the args model.
    assert spec.parameters == AddArgs.model_json_schema()
    assert spec.parameters["properties"]["a"]["type"] == "integer"


def test_spec_for_unknown_tool_raises_key_error() -> None:
    registry, _ = _registry_with_add()
    with pytest.raises(KeyError):
        registry.spec_for(["does_not_exist"])


async def test_execute_runs_allowed_tool() -> None:
    registry, tool = _registry_with_add()
    result = await registry.execute("add", {"a": 2, "b": 3}, allowed_tools={"add"})
    assert result.ok is True
    assert result.data == {"sum": 5}
    assert tool.called is True


async def test_execute_disallowed_tool_raises_permission_error() -> None:
    registry, tool = _registry_with_add()
    with pytest.raises(FridayPermissionError):
        await registry.execute("add", {"a": 1, "b": 2}, allowed_tools=frozenset())
    # The tool must never run when permission is denied.
    assert tool.called is False


async def test_execute_unknown_tool_raises_permission_error() -> None:
    registry, _ = _registry_with_add()
    # Not in the allow-list (and not registered) -> permission denial.
    with pytest.raises(FridayPermissionError):
        await registry.execute("ghost", {}, allowed_tools={"ghost"})


async def test_execute_bad_args_rejected_pre_execution() -> None:
    registry, tool = _registry_with_add()
    # "b" missing and "a" not an int -> validation must fail before the call.
    result = await registry.execute("add", {"a": "not-an-int"}, allowed_tools={"add"})
    assert result.ok is False
    assert result.error is not None
    assert result.error.code == "bad_args"
    assert result.error.retriable is False
    # Critically: the tool body must not have executed.
    assert tool.called is False


async def test_execute_accepts_set_or_frozenset_allowed() -> None:
    registry, _ = _registry_with_add()
    r1 = await registry.execute("add", {"a": 1, "b": 1}, allowed_tools={"add"})
    r2 = await registry.execute("add", {"a": 1, "b": 1}, allowed_tools=frozenset({"add"}))
    assert r1.ok is True
    assert r2.ok is True
