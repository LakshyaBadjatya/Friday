"""Unit tests for the tool boundary types (:mod:`friday.tools.base`).

Covers the typed result payload (:class:`ToolError` / :class:`ToolResult`) and a
trivial in-test ``EchoTool`` that satisfies the :class:`Tool` protocol.
"""

from __future__ import annotations

from pydantic import BaseModel

from friday.tools.base import Tool, ToolError, ToolResult


class EchoArgs(BaseModel):
    """Arguments for the trivial echo tool used in tests."""

    text: str


class EchoTool:
    """A minimal :class:`Tool` implementation that echoes its input back."""

    name = "echo"
    description = "Echo the provided text back to the caller."
    args_model = EchoArgs
    required_permission = "echo"
    idempotent = True
    side_effecting = False

    async def __call__(self, args: EchoArgs) -> ToolResult:
        return ToolResult(ok=True, data={"text": args.text}, error=None)


def test_tool_error_is_a_payload_not_the_exception() -> None:
    # The typed payload must be distinct from the FridayError exception family.
    from friday.errors import ToolError as ToolErrorException

    assert ToolError is not ToolErrorException
    assert not issubclass(ToolError, Exception)


def test_tool_error_fields() -> None:
    err = ToolError(code="boom", message="it broke", retriable=True)
    assert err.code == "boom"
    assert err.message == "it broke"
    assert err.retriable is True


def test_tool_result_ok_defaults() -> None:
    res = ToolResult(ok=True, data={"a": 1})
    assert res.ok is True
    assert res.data == {"a": 1}
    assert res.error is None


def test_tool_result_carries_error() -> None:
    err = ToolError(code="bad_args", message="nope", retriable=False)
    res = ToolResult(ok=False, data={}, error=err)
    assert res.ok is False
    assert res.error is err
    assert res.error.code == "bad_args"


def test_tool_result_round_trips_json() -> None:
    err = ToolError(code="x", message="y", retriable=False)
    res = ToolResult(ok=False, data={"k": "v"}, error=err)
    again = ToolResult.model_validate_json(res.model_dump_json())
    assert again == res


async def test_echo_tool_satisfies_protocol_and_runs() -> None:
    tool: Tool = EchoTool()
    assert tool.name == "echo"
    assert tool.side_effecting is False
    assert tool.idempotent is True
    assert tool.args_model is EchoArgs

    result = await tool(EchoArgs(text="hi"))
    assert isinstance(result, ToolResult)
    assert result.ok is True
    assert result.data == {"text": "hi"}
    assert result.error is None
