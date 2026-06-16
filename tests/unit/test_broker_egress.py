# © Lakshya Badjatya — Author
"""Unit tests for the broker's egress allow-list gate."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel

from friday.broker import Broker, HashChainedAudit
from friday.security.egress import EgressPolicy
from friday.tools.base import ToolResult


class FetchArgs(BaseModel):
    """Args for the fake outbound tool."""

    url: str


class FakeFetchTool:
    """A read-only fake tool that takes a URL and records whether it ran."""

    def __init__(self) -> None:
        self.name = "fetch"
        self.description = "fake fetch"
        self.args_model = FetchArgs
        self.required_permission = "fetch"
        self.idempotent = True
        self.side_effecting = False
        self.called = False

    async def __call__(self, args: Any) -> ToolResult:
        self.called = True
        return ToolResult(ok=True, data={"url": args.url})


class FakeRegistry:
    """A one-tool registry."""

    def __init__(self, tool: FakeFetchTool) -> None:
        self._tool = tool

    def get(self, name: str) -> FakeFetchTool:
        return self._tool


def _broker(
    tmp_path: Path, *, egress: EgressPolicy | None
) -> tuple[Broker, FakeFetchTool]:
    audit = HashChainedAudit(str(tmp_path / "audit.jsonl"))
    tool = FakeFetchTool()
    broker = Broker(FakeRegistry(tool), audit, egress_policy=egress)
    return broker, tool


async def test_egress_blocks_disallowed_host(tmp_path: Path) -> None:
    broker, tool = _broker(tmp_path, egress=EgressPolicy(["example.com"]))
    result = await broker.dispatch(
        "fetch", {"url": "https://evil.com/x"}, allowed_tools={"fetch"}
    )
    assert result.ok is False
    assert result.error is not None and result.error.code == "egress_blocked"
    assert tool.called is False  # blocked before execution


async def test_egress_allows_listed_host_and_subdomain(tmp_path: Path) -> None:
    broker, tool = _broker(tmp_path, egress=EgressPolicy(["example.com"]))
    result = await broker.dispatch(
        "fetch", {"url": "https://api.example.com/x"}, allowed_tools={"fetch"}
    )
    assert result.ok is True
    assert tool.called is True


async def test_no_policy_leaves_dispatch_unchanged(tmp_path: Path) -> None:
    broker, tool = _broker(tmp_path, egress=None)
    result = await broker.dispatch(
        "fetch", {"url": "https://evil.com/x"}, allowed_tools={"fetch"}
    )
    assert result.ok is True and tool.called is True


async def test_args_without_a_url_are_unaffected(tmp_path: Path) -> None:
    # Empty allow-list blocks all URLs, but an arg carrying no URL never triggers
    # the gate — so a non-network tool call still runs.
    broker, tool = _broker(tmp_path, egress=EgressPolicy([]))
    result = await broker.dispatch(
        "fetch", {"url": "just plain text, no link"}, allowed_tools={"fetch"}
    )
    assert result.ok is True and tool.called is True
