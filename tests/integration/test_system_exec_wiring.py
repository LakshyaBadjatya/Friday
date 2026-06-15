"""Integration tests for the system-automation tools' wiring in :mod:`friday.app`.

Fully offline (``llm_provider="fake"``, ``memory_db_path=":memory:"``): builds the
real runtime graph via :func:`friday.app.build_runtime` with the flag forced
on/off and asserts the registration + allow-list contract directly off the shared
registry / agent registry. The registry confirm-step is exercised end-to-end with a
monkeypatched subprocess (no real process spawned).

Covered:
* Flag OFF (the default): none of the three tools are registered, and the
  Automation agent's ``allowed_tools`` does not include them.
* Flag ON: all three are registered, and the Automation agent's ``allowed_tools``
  includes ``run_command`` / ``find_files`` / ``open_app``.
* Through the registry: ``run_command`` / ``open_app`` are side-effecting +
  non-idempotent, so ``execute`` without ``confirmed=True`` returns
  ``needs_confirmation`` and runs NO subprocess; ``confirmed=True`` runs it.
"""

from __future__ import annotations

from typing import Any

import pytest

import friday.tools.system_exec as system_exec_mod
from friday.app import build_runtime
from friday.config import Settings
from friday.tools.system_exec import FindFilesTool, OpenAppTool, RunCommandTool

_TOOL_NAMES = ("run_command", "find_files", "open_app")


def _settings(*, enable: bool) -> Settings:
    # ":memory:" keeps every runtime's stores ephemeral and isolated so tests
    # never touch the developer's real data/ files or each other.
    return Settings(
        _env_file=None,
        enable_system_automation=enable,
        llm_provider="fake",
        memory_db_path=":memory:",
    )


class _FakeProcess:
    def __init__(self, *, stdout: bytes, stderr: bytes, returncode: int) -> None:
        self._stdout = stdout
        self._stderr = stderr
        self.returncode = returncode

    async def communicate(self) -> tuple[bytes, bytes]:
        return self._stdout, self._stderr


def test_system_tools_unregistered_when_flag_off() -> None:
    runtime = build_runtime(_settings(enable=False))

    for name in _TOOL_NAMES:
        with pytest.raises(KeyError):
            runtime.registry.get(name)

    agents = runtime.orchestrator._agents  # noqa: SLF001
    assert agents is not None
    automation_tools = agents.get("automation").allowed_tools
    for name in _TOOL_NAMES:
        assert name not in automation_tools


def test_system_tools_registered_and_allowed_when_flag_on() -> None:
    runtime = build_runtime(_settings(enable=True))

    assert isinstance(runtime.registry.get("run_command"), RunCommandTool)
    assert isinstance(runtime.registry.get("find_files"), FindFilesTool)
    assert isinstance(runtime.registry.get("open_app"), OpenAppTool)

    agents = runtime.orchestrator._agents  # noqa: SLF001
    assert agents is not None
    automation_tools = agents.get("automation").allowed_tools
    for name in _TOOL_NAMES:
        assert name in automation_tools


async def test_run_command_needs_confirmation_through_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = build_runtime(_settings(enable=True))

    # If the tool body ran, this would spawn — assert it does NOT without confirm.
    async def _boom(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("no subprocess may run without confirmation")

    monkeypatch.setattr(
        system_exec_mod.asyncio, "create_subprocess_exec", _boom
    )

    allowed = {"run_command"}
    result = await runtime.registry.execute(
        "run_command", {"command": "echo", "args": ["hi"]}, allowed
    )

    assert result.ok is False
    assert result.error is not None
    assert result.error.code == "confirmation_required"
    assert result.data["needs_confirmation"] is True
    assert result.data["tool"] == "run_command"


async def test_run_command_runs_through_registry_when_confirmed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = build_runtime(_settings(enable=True))

    captured: dict[str, Any] = {}

    async def _fake_exec(*cmd: str, **_kwargs: Any) -> _FakeProcess:
        captured["cmd"] = cmd
        return _FakeProcess(stdout=b"hi\n", stderr=b"", returncode=0)

    monkeypatch.setattr(
        system_exec_mod.asyncio, "create_subprocess_exec", _fake_exec
    )

    allowed = {"run_command"}
    result = await runtime.registry.execute(
        "run_command", {"command": "echo", "args": ["hi"]}, allowed, confirmed=True
    )

    assert result.ok is True
    assert result.data["stdout"] == "hi\n"
    assert result.data["returncode"] == 0
    # SECURITY: argv list, no shell.
    assert captured["cmd"] == ("echo", "hi")


async def test_open_app_needs_confirmation_through_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = build_runtime(_settings(enable=True))

    async def _boom(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("no subprocess may run without confirmation")

    monkeypatch.setattr(
        system_exec_mod.asyncio, "create_subprocess_exec", _boom
    )

    allowed = {"open_app"}
    result = await runtime.registry.execute(
        "open_app", {"target": "https://example.com"}, allowed
    )

    assert result.ok is False
    assert result.error is not None
    assert result.error.code == "confirmation_required"


async def test_find_files_runs_without_confirmation_through_registry(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    # find_files is read-only, so it skips the confirm-step (no confirmed=True).
    (tmp_path / "note.txt").write_text("hi")
    monkeypatch.setenv("FRIDAY_SYSTEM_AUTOMATION_ROOT", str(tmp_path))

    runtime = build_runtime(_settings(enable=True))

    # Point the tool's settings at the tmp root via the patched module getter.
    def _settings_with_root() -> Settings:
        return Settings(
            _env_file=None,
            enable_system_automation=True,
            system_automation_root=str(tmp_path),
            memory_db_path=":memory:",
        )

    monkeypatch.setattr(system_exec_mod, "get_settings", _settings_with_root)

    allowed = {"find_files"}
    result = await runtime.registry.execute(
        "find_files", {"pattern": "*.txt"}, allowed
    )

    assert result.ok is True
    names = {p.split("/")[-1] for p in result.data["matches"]}
    assert "note.txt" in names
