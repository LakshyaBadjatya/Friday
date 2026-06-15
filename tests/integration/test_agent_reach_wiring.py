"""Integration tests for the ``agent_reach`` tool wiring in :mod:`friday.app`.

Fully offline (``llm_provider="fake"``, ``memory_db_path=":memory:"``): builds the
real runtime graph via :func:`friday.app.build_runtime` with the flag forced
on/off and asserts the registration + allow-list contract directly off the
shared registry / agent registry. No network, no subprocess.

Covered:
* Flag OFF (the default): ``agent_reach`` is NOT registered, and neither the
  Research (analysis) nor the Knowledge agent's ``allowed_tools`` includes it.
* Flag ON: ``agent_reach`` IS registered, and both the Research and Knowledge
  agents' ``allowed_tools`` include ``"agent_reach"``.
"""

from __future__ import annotations

import pytest

from friday.app import build_runtime
from friday.config import Settings
from friday.tools.agent_reach import AgentReachTool


def _settings(*, enable_agent_reach: bool) -> Settings:
    # ":memory:" keeps every runtime's stores ephemeral and isolated so tests
    # never touch the developer's real data/ files or each other.
    return Settings(
        _env_file=None,
        enable_agent_reach=enable_agent_reach,
        llm_provider="fake",
        memory_db_path=":memory:",
    )


def test_agent_reach_unregistered_when_flag_off() -> None:
    runtime = build_runtime(_settings(enable_agent_reach=False))

    # The tool must be absent from the shared registry.
    with pytest.raises(KeyError):
        runtime.registry.get("agent_reach")

    # And neither agent's allow-list may mention it.
    agents = runtime.orchestrator._agents  # noqa: SLF001
    assert agents is not None
    assert "agent_reach" not in agents.get("analysis").allowed_tools
    assert "agent_reach" not in agents.get("knowledge").allowed_tools


def test_agent_reach_registered_and_allowed_when_flag_on() -> None:
    runtime = build_runtime(_settings(enable_agent_reach=True))

    # The tool is registered as the read-only AgentReachTool.
    tool = runtime.registry.get("agent_reach")
    assert isinstance(tool, AgentReachTool)
    assert tool.side_effecting is False
    assert tool.idempotent is True

    # Both the Research (analysis) and Knowledge agents may reach it.
    agents = runtime.orchestrator._agents  # noqa: SLF001
    assert agents is not None
    assert "agent_reach" in agents.get("analysis").allowed_tools
    assert "agent_reach" in agents.get("knowledge").allowed_tools
