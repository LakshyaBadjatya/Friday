"""Integration tests for the Stage-2 idea-batch tool wiring in :mod:`friday.app`.

Fully offline (``llm_provider="fake"``, ``memory_db_path=":memory:"``): builds the
real runtime graph via :func:`friday.app.build_runtime` with the flags forced
on/off and asserts the registration + allow-list + confirm-step contracts directly
off the shared registry / agent registry. No network, no subprocess.

Covered:
* ``enable_extra_tools`` ON (the default): the read-only idea-batch tools
  (capabilities / ask_user / entity_dossier / infofeed / browser) are registered,
  read-only, and callable through the registry; they are added to the fitting
  agents' allow-lists (capabilities/ask_user to all; entity_dossier to knowledge;
  infofeed/browser to research).
* ``enable_extra_tools`` OFF: none of those tools are registered and the agents'
  allow-lists are unchanged.
* The side-effecting idea-batch tools (downloads_butler / media) are registered
  ONLY behind their own readiness flags and require the confirm-step
  (downloads_butler is side-effecting + non-idempotent).
"""

from __future__ import annotations

import pytest

from friday.app import build_runtime
from friday.config import Settings
from friday.tools.ask_user import AskUserTool
from friday.tools.browser_tool import BrowserTool
from friday.tools.capabilities import CapabilitiesTool
from friday.tools.dossier import DossierTool
from friday.tools.downloads_butler import DownloadsButlerTool
from friday.tools.infofeed import InfofeedTool
from friday.tools.media import MediaTool

_READ_ONLY_NAMES = ("capabilities", "ask_user", "entity_dossier", "infofeed", "browser")


def _settings(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "_env_file": None,
        "llm_provider": "fake",
        "memory_db_path": ":memory:",
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Read-only idea-batch tools — registered when ON (default)
# --------------------------------------------------------------------------- #
def test_read_only_tools_registered_by_default() -> None:
    runtime = build_runtime(_settings())
    reg = runtime.registry
    assert isinstance(reg.get("capabilities"), CapabilitiesTool)
    assert isinstance(reg.get("ask_user"), AskUserTool)
    assert isinstance(reg.get("entity_dossier"), DossierTool)
    assert isinstance(reg.get("infofeed"), InfofeedTool)
    assert isinstance(reg.get("browser"), BrowserTool)
    # All read-only — no real-world action.
    for name in _READ_ONLY_NAMES:
        tool = reg.get(name)
        assert tool.side_effecting is False


def test_read_only_tools_added_to_agent_allow_lists() -> None:
    runtime = build_runtime(_settings())
    agents = runtime.orchestrator._agents  # noqa: SLF001
    assert agents is not None

    analysis = agents.get("analysis").allowed_tools
    knowledge = agents.get("knowledge").allowed_tools
    automation = agents.get("automation").allowed_tools

    # capabilities + ask_user reach every agent.
    for tools in (analysis, knowledge, automation):
        assert "capabilities" in tools
        assert "ask_user" in tools

    # entity_dossier -> knowledge (memory persona).
    assert "entity_dossier" in knowledge
    assert "entity_dossier" not in analysis

    # infofeed + browser -> research (analysis).
    assert "infofeed" in analysis
    assert "browser" in analysis
    assert "infofeed" not in knowledge
    assert "browser" not in knowledge


async def test_capabilities_callable_via_registry() -> None:
    runtime = build_runtime(_settings())
    result = await runtime.registry.execute(
        "capabilities", {}, allowed_tools={"capabilities"}
    )
    assert result.ok is True
    names = {entry["name"] for entry in result.data["tools"]}
    # The map reflects the shared registry — including the idea-batch tools and
    # the built-ins (web_search is always registered).
    assert "capabilities" in names
    assert "web_search" in names
    assert "infofeed" in names


async def test_ask_user_callable_via_registry() -> None:
    runtime = build_runtime(_settings())
    result = await runtime.registry.execute(
        "ask_user",
        {"question": "Which file?", "options": ["a", "b"]},
        allowed_tools={"ask_user"},
    )
    assert result.ok is True
    assert result.data["needs_input"] is True
    assert result.data["question"] == "Which file?"
    assert result.data["options"] == ["a", "b"]


async def test_dossier_callable_via_registry() -> None:
    # The dossier reads the shared graph store + long-term store; with nothing
    # stored it returns a grounded "nothing on file" dossier (never fabricates).
    runtime = build_runtime(_settings())
    result = await runtime.registry.execute(
        "entity_dossier", {"name": "Acme Corp"}, allowed_tools={"entity_dossier"}
    )
    assert result.ok is True
    assert result.data["entity"] is None
    assert result.data["facts"] == []
    assert "Acme Corp" in result.data["summary"]


# --------------------------------------------------------------------------- #
# Read-only idea-batch tools — absent when OFF
# --------------------------------------------------------------------------- #
def test_read_only_tools_absent_when_flag_off() -> None:
    runtime = build_runtime(_settings(enable_extra_tools=False))
    reg = runtime.registry
    for name in _READ_ONLY_NAMES:
        with pytest.raises(KeyError):
            reg.get(name)

    agents = runtime.orchestrator._agents  # noqa: SLF001
    assert agents is not None
    for agent_name in ("analysis", "knowledge", "automation"):
        tools = agents.get(agent_name).allowed_tools
        for tool_name in _READ_ONLY_NAMES:
            assert tool_name not in tools


# --------------------------------------------------------------------------- #
# Side-effecting idea-batch tools — own readiness + confirm-step
# --------------------------------------------------------------------------- #
def test_side_effecting_tools_absent_by_default() -> None:
    runtime = build_runtime(_settings())
    reg = runtime.registry
    with pytest.raises(KeyError):
        reg.get("downloads_butler")
    with pytest.raises(KeyError):
        reg.get("media")


def test_downloads_butler_registered_behind_flag() -> None:
    runtime = build_runtime(_settings(enable_downloads_butler=True))
    tool = runtime.registry.get("downloads_butler")
    assert isinstance(tool, DownloadsButlerTool)
    # Side-effecting + non-idempotent -> the registry confirm-step gates it.
    assert tool.side_effecting is True
    assert tool.idempotent is False


def test_media_registered_behind_flag() -> None:
    runtime = build_runtime(_settings(enable_media_control=True))
    tool = runtime.registry.get("media")
    assert isinstance(tool, MediaTool)
    assert tool.side_effecting is True


async def test_downloads_butler_requires_confirm(tmp_path: object) -> None:
    runtime = build_runtime(_settings(enable_downloads_butler=True))
    # A real (non-dry-run) move is side-effecting + non-idempotent, so an
    # unconfirmed call is held at the confirm-step (the tool body never runs).
    result = await runtime.registry.execute(
        "downloads_butler",
        {"root": str(tmp_path), "dry_run": False},
        allowed_tools={"downloads_butler"},
    )
    assert result.ok is False
    assert result.error is not None
    assert result.error.code == "confirmation_required"
    assert result.data.get("needs_confirmation") is True


async def test_media_requires_confirm_only_when_non_idempotent() -> None:
    # Media is side-effecting but IDEMPOTENT, so it does NOT trip the confirm-step;
    # it dispatches straight through (play has no destructive, non-repeatable effect).
    runtime = build_runtime(_settings(enable_media_control=True))
    result = await runtime.registry.execute(
        "media", {"action": "play"}, allowed_tools={"media"}
    )
    assert result.ok is True
    assert result.data["action"] == "play"
