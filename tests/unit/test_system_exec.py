"""Unit tests for the system-automation tools.

Fully offline and side-effect-free: ``asyncio.create_subprocess_exec`` is
monkeypatched (no real process is ever spawned) and file search runs against a
``tmp_path`` root. The tests pin the SECURITY contract:

* commands run via an argv LIST only (``create_subprocess_exec`` — never a shell);
* output is capped and a timeout enforced;
* the optional allow-list blocks a disallowed command basename;
* ``find_files`` stays inside ``system_automation_root`` (path-traversal rejected);
* ``open_app`` rejects a ``-``-leading target and invokes the OS opener as argv.

Settings are injected by monkeypatching the module-level ``get_settings`` name the
tools import, following the repo's tool-test convention (see ``test_home_tool``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import friday.tools.system_exec as system_exec_mod
from friday.config import Settings
from friday.tools.base import ToolResult
from friday.tools.system_exec import (
    FindFilesArgs,
    FindFilesTool,
    OpenAppArgs,
    OpenAppTool,
    RunCommandArgs,
    RunCommandTool,
)


def _patch_settings(
    monkeypatch: pytest.MonkeyPatch,
    *,
    root: str = ".",
    allowlist: list[str] | None = None,
    timeout: float = 30.0,
) -> None:
    def _settings() -> Settings:
        return Settings(
            _env_file=None,
            system_automation_root=root,
            system_exec_allowlist=allowlist or [],
            system_exec_timeout=timeout,
        )

    monkeypatch.setattr(system_exec_mod, "get_settings", _settings)


class _FakeProcess:
    """A stand-in for the object ``create_subprocess_exec`` returns."""

    def __init__(self, *, stdout: bytes, stderr: bytes, returncode: int) -> None:
        self._stdout = stdout
        self._stderr = stderr
        self.returncode = returncode
        self.killed = False

    async def communicate(self) -> tuple[bytes, bytes]:
        return self._stdout, self._stderr

    def kill(self) -> None:  # pragma: no cover - only hit on the timeout path
        self.killed = True

    async def wait(self) -> int:  # pragma: no cover - only hit on the timeout path
        return self.returncode


# -- attributes ---------------------------------------------------------- #


def test_run_command_tool_attrs() -> None:
    tool = RunCommandTool()
    assert tool.name == "run_command"
    assert tool.side_effecting is True
    assert tool.idempotent is False
    assert tool.required_permission == "system"
    assert tool.args_model is RunCommandArgs


def test_find_files_tool_attrs() -> None:
    tool = FindFilesTool()
    assert tool.name == "find_files"
    assert tool.side_effecting is False
    assert tool.idempotent is True
    assert tool.required_permission == "system"
    assert tool.args_model is FindFilesArgs


def test_open_app_tool_attrs() -> None:
    tool = OpenAppTool()
    assert tool.name == "open_app"
    assert tool.side_effecting is True
    assert tool.idempotent is False
    assert tool.required_permission == "system"
    assert tool.args_model is OpenAppArgs


# -- args validation ----------------------------------------------------- #


def test_run_command_args_rejects_empty_command() -> None:
    with pytest.raises(ValueError):
        RunCommandArgs(command="")


def test_run_command_args_defaults_empty_arg_list() -> None:
    args = RunCommandArgs(command="echo")
    assert args.args == []


def test_open_app_args_rejects_empty_target() -> None:
    with pytest.raises(ValueError):
        OpenAppArgs(target="")


@pytest.mark.parametrize("bad", ["-rf", "--help", "-x"])
def test_open_app_args_rejects_option_like_target(bad: str) -> None:
    with pytest.raises(ValueError):
        OpenAppArgs(target=bad)


# -- run_command --------------------------------------------------------- #


async def test_run_command_returns_stdout_and_returncode_via_argv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_settings(monkeypatch)

    captured: dict[str, Any] = {}

    async def _fake_exec(*cmd: str, **_kwargs: Any) -> _FakeProcess:
        captured["cmd"] = cmd
        return _FakeProcess(stdout=b"hello\n", stderr=b"", returncode=0)

    monkeypatch.setattr(
        system_exec_mod.asyncio, "create_subprocess_exec", _fake_exec
    )

    tool = RunCommandTool()
    result = await tool(RunCommandArgs(command="echo", args=["hello"]))

    assert isinstance(result, ToolResult)
    assert result.ok is True
    assert result.error is None
    assert result.data["stdout"] == "hello\n"
    assert result.data["stderr"] == ""
    assert result.data["returncode"] == 0
    # SECURITY: the command is passed as an argv LIST (program + args), never a
    # shell string. The first positional is the program; the rest are its args.
    assert captured["cmd"] == ("echo", "hello")


async def test_run_command_does_not_use_shell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_settings(monkeypatch)

    # If the tool ever reached for a shell, this guard would trip the test.
    def _boom_shell(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("system_exec must never use create_subprocess_shell")

    monkeypatch.setattr(
        system_exec_mod.asyncio, "create_subprocess_shell", _boom_shell, raising=False
    )

    async def _fake_exec(*_cmd: str, **_kwargs: Any) -> _FakeProcess:
        return _FakeProcess(stdout=b"ok", stderr=b"", returncode=0)

    monkeypatch.setattr(
        system_exec_mod.asyncio, "create_subprocess_exec", _fake_exec
    )

    tool = RunCommandTool()
    result = await tool(RunCommandArgs(command="true"))
    assert result.ok is True


async def test_run_command_surfaces_nonzero_returncode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_settings(monkeypatch)

    async def _fake_exec(*_cmd: str, **_kwargs: Any) -> _FakeProcess:
        return _FakeProcess(stdout=b"", stderr=b"boom", returncode=2)

    monkeypatch.setattr(
        system_exec_mod.asyncio, "create_subprocess_exec", _fake_exec
    )

    tool = RunCommandTool()
    result = await tool(RunCommandArgs(command="false"))

    # A non-zero exit is reported honestly: ok=True (it ran) but returncode != 0
    # and stderr surfaced, so the caller sees the real outcome.
    assert result.ok is True
    assert result.data["returncode"] == 2
    assert result.data["stderr"] == "boom"


async def test_run_command_truncates_large_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_settings(monkeypatch)

    big = b"x" * (system_exec_mod.MAX_OUTPUT_CHARS + 5000)

    async def _fake_exec(*_cmd: str, **_kwargs: Any) -> _FakeProcess:
        return _FakeProcess(stdout=big, stderr=big, returncode=0)

    monkeypatch.setattr(
        system_exec_mod.asyncio, "create_subprocess_exec", _fake_exec
    )

    tool = RunCommandTool()
    result = await tool(RunCommandArgs(command="cat"))

    assert len(result.data["stdout"]) <= system_exec_mod.MAX_OUTPUT_CHARS
    assert len(result.data["stderr"]) <= system_exec_mod.MAX_OUTPUT_CHARS


async def test_run_command_spawn_error_returns_tool_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_settings(monkeypatch)

    async def _fake_exec(*_cmd: str, **_kwargs: Any) -> _FakeProcess:
        raise FileNotFoundError("no such file or directory: nope")

    monkeypatch.setattr(
        system_exec_mod.asyncio, "create_subprocess_exec", _fake_exec
    )

    tool = RunCommandTool()
    result = await tool(RunCommandArgs(command="nope"))

    assert result.ok is False
    assert result.error is not None
    assert result.error.code == "spawn_failed"
    assert "stdout" not in result.data


async def test_run_command_timeout_returns_tool_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_settings(monkeypatch, timeout=0.01)

    proc = _FakeProcess(stdout=b"", stderr=b"", returncode=0)

    async def _fake_exec(*_cmd: str, **_kwargs: Any) -> _FakeProcess:
        return proc

    async def _fake_wait_for(awaitable: Any, timeout: float) -> Any:
        # Close the underlying coroutine so it is not flagged as never-awaited,
        # then surface the timeout exactly as ``asyncio.wait_for`` would.
        if hasattr(awaitable, "close"):
            awaitable.close()
        raise TimeoutError("timed out")

    monkeypatch.setattr(
        system_exec_mod.asyncio, "create_subprocess_exec", _fake_exec
    )
    monkeypatch.setattr(system_exec_mod.asyncio, "wait_for", _fake_wait_for)

    tool = RunCommandTool()
    result = await tool(RunCommandArgs(command="sleep", args=["100"]))

    assert result.ok is False
    assert result.error is not None
    assert result.error.code == "timeout"
    # The runaway process is killed rather than left dangling.
    assert proc.killed is True


async def test_run_command_allowlist_blocks_disallowed_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_settings(monkeypatch, allowlist=["echo"])

    async def _boom(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("a disallowed command must not be spawned")

    monkeypatch.setattr(
        system_exec_mod.asyncio, "create_subprocess_exec", _boom
    )

    tool = RunCommandTool()
    result = await tool(RunCommandArgs(command="rm", args=["-rf", "/"]))

    assert result.ok is False
    assert result.error is not None
    assert result.error.code == "command_not_allowed"


async def test_run_command_allowlist_permits_listed_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The allow-list matches on the BASENAME, so an absolute path to an allowed
    # program still passes.
    _patch_settings(monkeypatch, allowlist=["echo"])

    captured: dict[str, Any] = {}

    async def _fake_exec(*cmd: str, **_kwargs: Any) -> _FakeProcess:
        captured["cmd"] = cmd
        return _FakeProcess(stdout=b"hi", stderr=b"", returncode=0)

    monkeypatch.setattr(
        system_exec_mod.asyncio, "create_subprocess_exec", _fake_exec
    )

    tool = RunCommandTool()
    result = await tool(RunCommandArgs(command="/bin/echo", args=["hi"]))

    assert result.ok is True
    assert captured["cmd"] == ("/bin/echo", "hi")


# -- find_files ---------------------------------------------------------- #


async def test_find_files_finds_files_under_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "b.txt").write_text("b")
    (tmp_path / "c.log").write_text("c")
    _patch_settings(monkeypatch, root=str(tmp_path))

    tool = FindFilesTool()
    result = await tool(FindFilesArgs(pattern="*.txt"))

    assert result.ok is True
    assert result.error is None
    found = {Path(p).name for p in result.data["matches"]}
    assert found == {"a.txt", "b.txt"}


async def test_find_files_recursive_glob_under_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sub = tmp_path / "nested" / "deep"
    sub.mkdir(parents=True)
    (sub / "found.py").write_text("x")
    _patch_settings(monkeypatch, root=str(tmp_path))

    tool = FindFilesTool()
    result = await tool(FindFilesArgs(pattern="**/*.py"))

    assert result.ok is True
    names = {Path(p).name for p in result.data["matches"]}
    assert "found.py" in names


async def test_find_files_rejects_traversal_pattern(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_settings(monkeypatch, root=str(tmp_path))

    tool = FindFilesTool()
    result = await tool(FindFilesArgs(pattern="../etc/*"))

    assert result.ok is False
    assert result.error is not None
    assert result.error.code == "path_not_allowed"
    assert "matches" not in result.data


async def test_find_files_rejects_absolute_traversal_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_settings(monkeypatch, root=str(tmp_path))

    tool = FindFilesTool()
    # An explicit root pointing outside the allowlisted root must be rejected.
    result = await tool(FindFilesArgs(pattern="*", root="/etc"))

    assert result.ok is False
    assert result.error is not None
    assert result.error.code == "path_not_allowed"


async def test_find_files_caps_match_count(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    for i in range(system_exec_mod.MAX_MATCHES + 25):
        (tmp_path / f"f{i}.dat").write_text("x")
    _patch_settings(monkeypatch, root=str(tmp_path))

    tool = FindFilesTool()
    result = await tool(FindFilesArgs(pattern="*.dat"))

    assert result.ok is True
    assert len(result.data["matches"]) <= system_exec_mod.MAX_MATCHES


# -- open_app ------------------------------------------------------------ #


async def test_open_app_invokes_opener_as_argv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_settings(monkeypatch)
    monkeypatch.setattr(system_exec_mod.sys, "platform", "linux")

    captured: dict[str, Any] = {}

    async def _fake_exec(*cmd: str, **_kwargs: Any) -> _FakeProcess:
        captured["cmd"] = cmd
        return _FakeProcess(stdout=b"", stderr=b"", returncode=0)

    monkeypatch.setattr(
        system_exec_mod.asyncio, "create_subprocess_exec", _fake_exec
    )

    tool = OpenAppTool()
    result = await tool(OpenAppArgs(target="https://example.com"))

    assert result.ok is True
    # SECURITY: opener invoked as an argv list (opener + target), never a shell.
    assert captured["cmd"][0] == "xdg-open"
    assert captured["cmd"][-1] == "https://example.com"


async def test_open_app_uses_platform_opener(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_settings(monkeypatch)
    monkeypatch.setattr(system_exec_mod.sys, "platform", "darwin")

    captured: dict[str, Any] = {}

    async def _fake_exec(*cmd: str, **_kwargs: Any) -> _FakeProcess:
        captured["cmd"] = cmd
        return _FakeProcess(stdout=b"", stderr=b"", returncode=0)

    monkeypatch.setattr(
        system_exec_mod.asyncio, "create_subprocess_exec", _fake_exec
    )

    tool = OpenAppTool()
    result = await tool(OpenAppArgs(target="Calculator"))

    assert result.ok is True
    assert captured["cmd"][0] == "open"


async def test_open_app_rejects_option_like_target_at_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Defence in depth: even if a bad target somehow reaches __call__, a
    # leading-dash target is refused before any subprocess is spawned.
    _patch_settings(monkeypatch)

    async def _boom(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("an option-like target must not be opened")

    monkeypatch.setattr(
        system_exec_mod.asyncio, "create_subprocess_exec", _boom
    )

    tool = OpenAppTool()
    result = await tool(OpenAppArgs.model_construct(target="-malicious"))

    assert result.ok is False
    assert result.error is not None
    assert result.error.code == "bad_target"


async def test_open_app_spawn_error_returns_tool_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_settings(monkeypatch)
    monkeypatch.setattr(system_exec_mod.sys, "platform", "linux")

    async def _fake_exec(*_cmd: str, **_kwargs: Any) -> _FakeProcess:
        raise FileNotFoundError("xdg-open not installed")

    monkeypatch.setattr(
        system_exec_mod.asyncio, "create_subprocess_exec", _fake_exec
    )

    tool = OpenAppTool()
    result = await tool(OpenAppArgs(target="thing"))

    assert result.ok is False
    assert result.error is not None
    assert result.error.code == "open_failed"
