"""Unit tests for :class:`friday.tools.notify.NotifyTool`.

The notify tool is side-effecting and non-idempotent (the registry confirm-step
gates it), but the channel adapters are FAKE: nothing is sent over the wire.
Each call records the message to an in-memory sink exposed for inspection and
reports ``{"sent": True, "channel": ...}``. No network is touched, so no
``respx`` mocking is required.
"""

from __future__ import annotations

from friday.tools.base import ToolResult
from friday.tools.notify import NotifyArgs, NotifyTool


def test_notify_tool_attrs() -> None:
    tool = NotifyTool()
    assert tool.name == "notify"
    assert tool.side_effecting is True
    assert tool.idempotent is False
    assert tool.args_model is NotifyArgs


def test_notify_args_channels() -> None:
    args = NotifyArgs(
        channel="email", target="boss@example.com", subject="hi", body="there"
    )
    assert args.channel == "email"
    assert args.target == "boss@example.com"


async def test_notify_records_to_sink_and_reports_sent() -> None:
    tool = NotifyTool()
    args = NotifyArgs(
        channel="slack",
        target="#ops",
        subject="deploy",
        body="shipping now",
    )
    result = await tool(args)

    assert isinstance(result, ToolResult)
    assert result.ok is True
    assert result.error is None
    assert result.data == {"sent": True, "channel": "slack", "target": "#ops"}

    # The FAKE adapter recorded the message to the in-memory sink.
    assert len(tool.sink) == 1
    recorded = tool.sink[0]
    assert recorded.channel == "slack"
    assert recorded.target == "#ops"
    assert recorded.subject == "deploy"
    assert recorded.body == "shipping now"


async def test_notify_accumulates_multiple_sends() -> None:
    tool = NotifyTool()
    await tool(NotifyArgs(channel="email", target="a@x", subject="s1", body="b1"))
    await tool(NotifyArgs(channel="webhook", target="https://x/h", subject="s2", body="b2"))

    assert len(tool.sink) == 2
    assert [m.channel for m in tool.sink] == ["email", "webhook"]
    assert [m.target for m in tool.sink] == ["a@x", "https://x/h"]


async def test_notify_coerces_raw_mapping() -> None:
    # Called directly with a raw mapping (as the registry would after validation,
    # but defensively coercible) it still validates and records.
    tool = NotifyTool()
    result = await tool(
        {"channel": "webhook", "target": "https://hook", "subject": "s", "body": "b"}
    )
    assert result.ok is True
    assert result.data["channel"] == "webhook"
    assert tool.sink[0].channel == "webhook"
