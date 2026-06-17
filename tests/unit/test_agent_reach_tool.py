"""Unit tests for :class:`friday.tools.agent_reach.AgentReachTool`.

Fully offline. The ``read_url`` HTTP is mocked with ``respx`` (no live network);
``transcribe`` is exercised with ``shutil.which`` / ``asyncio.create_subprocess_exec``
monkeypatched (no real binary, no real subprocess). The tool is read-only and
must NEVER fabricate content on failure.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest
import respx

from friday.tools import agent_reach as agent_reach_module
from friday.tools.agent_reach import AgentReachArgs, AgentReachTool
from friday.tools.base import ToolResult

JINA_BASE = "https://r.jina.ai/"
PAGE_URL = "https://example.com/article"
JINA_URL = f"{JINA_BASE}{PAGE_URL}"

SAMPLE_MARKDOWN = "# Example Article\n\nClean markdown body from Jina Reader.\n"


# -- attributes / args --------------------------------------------------- #


def test_agent_reach_tool_attrs() -> None:
    tool = AgentReachTool()
    assert tool.name == "agent_reach"
    assert tool.side_effecting is False
    assert tool.idempotent is True
    assert tool.required_permission == "web"
    assert tool.args_model is AgentReachArgs


def test_agent_reach_args_defaults() -> None:
    args = AgentReachArgs(target=PAGE_URL)
    assert args.action == "read_url"
    assert args.target == PAGE_URL


def test_agent_reach_args_rejects_empty_target() -> None:
    with pytest.raises(ValueError):
        AgentReachArgs(target="")


@pytest.mark.parametrize("bad", ["-rf", "--output=/etc/x", "-v"])
def test_agent_reach_args_rejects_option_like_target(bad: str) -> None:
    # Argv flag smuggling guard: a target that looks like a CLI flag is rejected
    # before it can reach the agent-reach subprocess.
    with pytest.raises(ValueError):
        AgentReachArgs(action="transcribe", target=bad)


# -- read_url (keyless, binary-independent) ------------------------------ #


@respx.mock
async def test_read_url_returns_markdown_content() -> None:
    respx.get(JINA_URL).mock(
        return_value=httpx.Response(200, text=SAMPLE_MARKDOWN)
    )
    tool = AgentReachTool(jina_base=JINA_BASE)
    result = await tool(AgentReachArgs(action="read_url", target=PAGE_URL))

    assert isinstance(result, ToolResult)
    assert result.ok is True
    assert result.error is None
    assert result.data["content"] == SAMPLE_MARKDOWN
    assert result.data["source"] == "jina-reader"
    assert result.data["url"] == PAGE_URL


@respx.mock
async def test_read_url_does_not_require_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    # read_url must work even when the agent-reach binary is entirely absent.
    monkeypatch.setattr(agent_reach_module.shutil, "which", lambda _name: None)
    respx.get(JINA_URL).mock(
        return_value=httpx.Response(200, text=SAMPLE_MARKDOWN)
    )
    tool = AgentReachTool(jina_base=JINA_BASE)
    result = await tool(AgentReachArgs(action="read_url", target=PAGE_URL))
    assert result.ok is True
    assert result.data["content"] == SAMPLE_MARKDOWN


@respx.mock
async def test_read_url_retries_once_then_fails() -> None:
    route = respx.get(JINA_URL).mock(
        side_effect=httpx.ConnectError("no route to host")
    )
    tool = AgentReachTool(jina_base=JINA_BASE)
    result = await tool(AgentReachArgs(action="read_url", target=PAGE_URL))

    # Exactly two attempts: the initial call plus one bounded retry.
    assert route.call_count == 2
    assert result.ok is False
    assert result.error is not None
    assert result.error.code == "read_failed"
    assert result.error.retriable is True
    # No fabricated content on failure.
    assert "content" not in result.data


@respx.mock
async def test_read_url_succeeds_on_retry() -> None:
    route = respx.get(JINA_URL).mock(
        side_effect=[
            httpx.ConnectError("transient blip"),
            httpx.Response(200, text=SAMPLE_MARKDOWN),
        ]
    )
    tool = AgentReachTool(jina_base=JINA_BASE)
    result = await tool(AgentReachArgs(action="read_url", target=PAGE_URL))

    assert route.call_count == 2
    assert result.ok is True
    assert result.data["content"] == SAMPLE_MARKDOWN


@respx.mock
async def test_read_url_http_error_returns_failure() -> None:
    respx.get(JINA_URL).mock(
        return_value=httpx.Response(503, text="service unavailable")
    )
    tool = AgentReachTool(jina_base=JINA_BASE)
    result = await tool(AgentReachArgs(action="read_url", target=PAGE_URL))
    assert result.ok is False
    assert result.error is not None
    assert result.error.code == "read_failed"
    assert result.error.retriable is True
    assert "content" not in result.data


# -- transcribe (installed CLI; clean missing-binary degradation) -------- #


class _FakeProcess:
    """A stand-in for the object ``create_subprocess_exec`` returns."""

    def __init__(self, *, stdout: bytes, stderr: bytes, returncode: int) -> None:
        self._stdout = stdout
        self._stderr = stderr
        self.returncode = returncode

    async def communicate(self) -> tuple[bytes, bytes]:
        return self._stdout, self._stderr


async def test_transcribe_kills_and_reaps_subprocess_on_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # On timeout the child must be killed and reaped — leaving it running leaks the
    # process and its stdout/stderr pipe FDs.
    monkeypatch.setattr(
        agent_reach_module.shutil, "which", lambda _name: "/usr/bin/agent-reach"
    )

    class _HangingProcess:
        def __init__(self) -> None:
            self.killed = False
            self.waited = False

        async def communicate(self) -> tuple[bytes, bytes]:
            await asyncio.sleep(10)  # longer than the tool's timeout -> wait_for fires
            return b"", b""

        def kill(self) -> None:
            self.killed = True

        async def wait(self) -> int:
            self.waited = True
            return -9

    proc = _HangingProcess()

    async def _fake_exec(*_cmd: str, **_kwargs: Any) -> _HangingProcess:
        return proc

    monkeypatch.setattr(
        agent_reach_module.asyncio, "create_subprocess_exec", _fake_exec
    )

    tool = AgentReachTool(cli_path="agent-reach", timeout=0.05)
    result = await tool(AgentReachArgs(action="transcribe", target=PAGE_URL))

    assert result.ok is False
    assert result.error is not None and result.error.code == "transcribe_failed"
    assert result.error.retriable is True
    assert proc.killed is True and proc.waited is True  # reaped, not leaked


async def test_transcribe_missing_cli_returns_error_without_subprocess(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(agent_reach_module.shutil, "which", lambda _name: None)

    # If the tool tried to spawn a subprocess it would hit this guard and fail.
    async def _boom(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("subprocess must not be spawned when CLI is missing")

    monkeypatch.setattr(
        agent_reach_module.asyncio, "create_subprocess_exec", _boom
    )

    tool = AgentReachTool()
    result = await tool(AgentReachArgs(action="transcribe", target=PAGE_URL))

    assert result.ok is False
    assert result.error is not None
    assert result.error.code == "agent_reach_cli_missing"
    assert result.error.retriable is False
    # The install hint must point the owner at the uv tool install.
    assert "uv tool install" in result.error.message
    # No fabricated transcript on failure.
    assert "transcript" not in result.data


async def test_transcribe_returns_transcript_from_fake_subprocess(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        agent_reach_module.shutil, "which", lambda _name: "/usr/bin/agent-reach"
    )

    captured: dict[str, Any] = {}

    async def _fake_exec(*cmd: str, **_kwargs: Any) -> _FakeProcess:
        captured["cmd"] = cmd
        return _FakeProcess(
            stdout=b"the transcribed text", stderr=b"", returncode=0
        )

    monkeypatch.setattr(
        agent_reach_module.asyncio, "create_subprocess_exec", _fake_exec
    )

    tool = AgentReachTool(cli_path="agent-reach")
    result = await tool(AgentReachArgs(action="transcribe", target=PAGE_URL))

    assert result.ok is True
    assert result.error is None
    assert result.data["transcript"] == "the transcribed text"
    assert result.data["source"] == "agent-reach"
    # The CLI is invoked as `agent-reach transcribe -- <target>` (the `--`
    # end-of-options separator prevents a leading-dash target smuggling a flag).
    assert captured["cmd"] == ("agent-reach", "transcribe", "--", PAGE_URL)


async def test_transcribe_nonzero_exit_returns_failure_without_fabricating(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        agent_reach_module.shutil, "which", lambda _name: "/usr/bin/agent-reach"
    )

    async def _fake_exec(*_cmd: str, **_kwargs: Any) -> _FakeProcess:
        return _FakeProcess(
            stdout=b"", stderr=b"download failed", returncode=2
        )

    monkeypatch.setattr(
        agent_reach_module.asyncio, "create_subprocess_exec", _fake_exec
    )

    tool = AgentReachTool()
    result = await tool(AgentReachArgs(action="transcribe", target=PAGE_URL))

    assert result.ok is False
    assert result.error is not None
    assert result.error.code == "transcribe_failed"
    # No fabricated transcript on failure.
    assert "transcript" not in result.data
