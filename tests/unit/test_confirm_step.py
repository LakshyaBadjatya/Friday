"""Unit tests for the registry confirm-step (build-spec §12 HARD RULE).

A side-effecting, non-idempotent tool must NOT execute without an explicit
``confirmed=True``; instead the registry returns
``ToolResult(ok=False, error=ToolError(code="confirmation_required"))`` carrying
``data={"needs_confirmation": True, "tool": <name>}``. With ``confirmed=True``
(or for non-side-effecting / idempotent tools) the tool executes normally.

The permission check and argument validation still run *before* the confirm
gate: a disallowed tool raises and bad args are rejected without executing,
regardless of ``confirmed``.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from friday.errors import PermissionError as FridayPermissionError
from friday.tools.base import ToolResult
from friday.tools.registry import ToolRegistry


class FireArgs(BaseModel):
    """Args for the spying side-effecting tool."""

    target: str


class SpyFireTool:
    """A side-effecting, non-idempotent tool that records invocation."""

    name = "fire"
    description = "A side-effecting, non-idempotent action."
    args_model = FireArgs
    required_permission = "fire"
    idempotent = False
    side_effecting = True

    def __init__(self) -> None:
        self.called = False

    async def __call__(self, args: FireArgs) -> ToolResult:
        self.called = True
        return ToolResult(ok=True, data={"fired_at": args.target})


class ReadArgs(BaseModel):
    """Args for the spying read-only tool."""

    q: str


class SpyReadTool:
    """A non-side-effecting, idempotent tool that records invocation."""

    name = "read"
    description = "A read-only, idempotent action."
    args_model = ReadArgs
    required_permission = "read"
    idempotent = True
    side_effecting = False

    def __init__(self) -> None:
        self.called = False

    async def __call__(self, args: ReadArgs) -> ToolResult:
        self.called = True
        return ToolResult(ok=True, data={"echo": args.q})


def _registry() -> tuple[ToolRegistry, SpyFireTool, SpyReadTool]:
    fire = SpyFireTool()
    read = SpyReadTool()
    registry = ToolRegistry()
    registry.register(fire)
    registry.register(read)
    return registry, fire, read


async def test_side_effecting_tool_blocked_without_confirmation() -> None:
    registry, fire, _ = _registry()
    result = await registry.execute(
        "fire", {"target": "prod"}, allowed_tools={"fire"}
    )
    # The tool body must NOT have executed.
    assert fire.called is False
    assert result.ok is False
    assert result.error is not None
    assert result.error.code == "confirmation_required"
    assert result.error.retriable is False
    assert result.data["needs_confirmation"] is True
    assert result.data["tool"] == "fire"


async def test_side_effecting_tool_runs_with_confirmation() -> None:
    registry, fire, _ = _registry()
    result = await registry.execute(
        "fire", {"target": "prod"}, allowed_tools={"fire"}, confirmed=True
    )
    assert fire.called is True
    assert result.ok is True
    assert result.data == {"fired_at": "prod"}


async def test_read_only_tool_runs_without_confirmation() -> None:
    registry, _, read = _registry()
    result = await registry.execute("read", {"q": "hi"}, allowed_tools={"read"})
    assert read.called is True
    assert result.ok is True
    assert result.data == {"echo": "hi"}


async def test_permission_check_precedes_confirm_gate() -> None:
    registry, fire, _ = _registry()
    # Disallowed -> permission error regardless of confirm gate; never executes.
    with pytest.raises(FridayPermissionError):
        await registry.execute("fire", {"target": "prod"}, allowed_tools=frozenset())
    assert fire.called is False


async def test_bad_args_rejected_before_confirm_gate() -> None:
    registry, fire, _ = _registry()
    # Missing required "target" -> bad_args, not confirmation_required; no exec.
    result = await registry.execute("fire", {}, allowed_tools={"fire"})
    assert fire.called is False
    assert result.ok is False
    assert result.error is not None
    assert result.error.code == "bad_args"
