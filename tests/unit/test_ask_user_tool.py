"""Unit tests for :class:`friday.tools.ask_user.AskUserTool`.

The tool is pure and dependency-free, so the tests construct it bare. They pin
the protocol attributes, the ``needs_input`` pause-signal payload (free-text and
multiple-choice), the empty-options normalization, and a round-trip through a
real :class:`ToolRegistry`.
"""

from __future__ import annotations

from typing import cast

from friday.tools.ask_user import AskUserArgs, AskUserTool
from friday.tools.base import Tool, ToolResult
from friday.tools.registry import ToolRegistry


# --------------------------------------------------------------------------- #
# Attributes
# --------------------------------------------------------------------------- #
def test_ask_user_tool_attrs() -> None:
    tool = AskUserTool()
    assert isinstance(tool, Tool)
    assert tool.name == "ask_user"
    assert tool.args_model is AskUserArgs
    assert tool.side_effecting is False
    assert tool.idempotent is True


# --------------------------------------------------------------------------- #
# Pause-signal payload
# --------------------------------------------------------------------------- #
async def test_ask_user_free_text_question() -> None:
    tool = AskUserTool()
    result = await tool(AskUserArgs(question="What's your departure city?"))

    assert isinstance(result, ToolResult)
    assert result.ok is True
    assert result.error is None
    assert result.data["needs_input"] is True
    assert result.data["question"] == "What's your departure city?"
    # No options provided -> None (free-text prompt).
    assert result.data["options"] is None


async def test_ask_user_multiple_choice() -> None:
    tool = AskUserTool()
    result = await tool(
        AskUserArgs(question="Pick a seat", options=["window", "aisle"])
    )
    assert result.ok is True
    assert result.data["needs_input"] is True
    assert result.data["question"] == "Pick a seat"
    assert result.data["options"] == ["window", "aisle"]


async def test_ask_user_empty_options_normalized_to_none() -> None:
    tool = AskUserTool()
    result = await tool(AskUserArgs(question="anything", options=[]))
    # An empty options list collapses to None so the loop sees one clear signal.
    assert result.data["options"] is None


async def test_ask_user_accepts_raw_dict_args() -> None:
    tool = AskUserTool()
    result = await tool({"question": "ready?", "options": ["yes", "no"]})
    assert result.ok is True
    assert result.data["question"] == "ready?"
    assert result.data["options"] == ["yes", "no"]


# --------------------------------------------------------------------------- #
# Round-trip through the real registry
# --------------------------------------------------------------------------- #
async def test_ask_user_round_trip_via_registry() -> None:
    registry = ToolRegistry()
    registry.register(cast(Tool, AskUserTool()))

    result = await registry.execute(
        "ask_user",
        {"question": "Confirm the address?", "options": ["yes", "no"]},
        allowed_tools={"ask_user"},
    )
    assert result.ok is True
    assert result.data["needs_input"] is True
    assert result.data["options"] == ["yes", "no"]


async def test_ask_user_bad_args_rejected_by_registry() -> None:
    registry = ToolRegistry()
    registry.register(cast(Tool, AskUserTool()))
    # Empty question violates ``min_length=1`` -> rejected pre-execution.
    result = await registry.execute(
        "ask_user", {"question": ""}, allowed_tools={"ask_user"}
    )
    assert result.ok is False
    assert result.error is not None
    assert result.error.code == "bad_args"
