"""Unit tests for :mod:`friday.tools.media`.

Fully offline: the tool is driven through :class:`FakeMedia` (no hardware), and
the :class:`SystemMedia` adapter's ``pynput`` backend is exercised purely via a
monkeypatched fake keyboard / forced ImportError — no real package or key press.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from friday.tools.base import ToolResult
from friday.tools.media import (
    FakeMedia,
    MediaArgs,
    MediaBackendUnavailable,
    MediaController,
    MediaTool,
    SystemMedia,
)

# -- attributes / args --------------------------------------------------- #


def test_media_tool_attrs() -> None:
    tool = MediaTool()
    assert tool.name == "media"
    assert tool.side_effecting is True
    # Transport commands are idempotent so the confirm-step does not trip.
    assert tool.idempotent is True
    assert tool.required_permission == "media"
    assert tool.args_model is MediaArgs


def test_fake_media_satisfies_controller_protocol() -> None:
    assert isinstance(FakeMedia(), MediaController)
    assert isinstance(SystemMedia(), MediaController)


def test_volume_args_clamped_by_validation() -> None:
    with pytest.raises(ValueError):
        MediaArgs(action="volume", level=150)
    with pytest.raises(ValueError):
        MediaArgs(action="volume", level=-1)


# -- FakeMedia + MediaTool happy paths ----------------------------------- #


async def test_play_and_pause_track_state() -> None:
    fake = FakeMedia()
    tool = MediaTool(controller=fake)

    res_play = await tool(MediaArgs(action="play"))
    assert isinstance(res_play, ToolResult)
    assert res_play.ok is True
    assert res_play.data == {"action": "play"}
    assert fake.playing is True

    res_pause = await tool(MediaArgs(action="pause"))
    assert res_pause.ok is True
    assert fake.playing is False

    assert fake.commands == ["play", "pause"]


async def test_next_and_prev_recorded() -> None:
    fake = FakeMedia()
    tool = MediaTool(controller=fake)
    await tool(MediaArgs(action="next"))
    await tool(MediaArgs(action="prev"))
    assert fake.commands == ["next", "prev"]


async def test_volume_sets_clamped_level() -> None:
    fake = FakeMedia()
    tool = MediaTool(controller=fake)
    result = await tool(MediaArgs(action="volume", level=80))
    assert result.ok is True
    assert result.data == {"action": "volume", "level": 80}
    assert fake.level == 80
    assert fake.commands == ["volume:80"]


async def test_volume_without_level_is_rejected() -> None:
    fake = FakeMedia()
    tool = MediaTool(controller=fake)
    result = await tool(MediaArgs(action="volume"))
    assert result.ok is False
    assert result.error is not None
    assert result.error.code == "missing_level"
    # The controller was never touched.
    assert fake.commands == []


async def test_tool_defaults_to_fake_controller() -> None:
    tool = MediaTool()
    result = await tool(MediaArgs(action="play"))
    assert result.ok is True


# -- SystemMedia (lazy pynput) ------------------------------------------- #


class _FakeKey:
    media_play_pause = "play_pause"
    media_next = "next"
    media_previous = "previous"
    media_volume_up = "vol_up"
    media_volume_down = "vol_down"


class _FakeController:
    def __init__(self) -> None:
        self.events: list[str] = []

    def press(self, key: str) -> None:
        self.events.append(f"press:{key}")

    def release(self, key: str) -> None:
        self.events.append(f"release:{key}")


def _install_fake_pynput(monkeypatch: pytest.MonkeyPatch) -> _FakeController:
    """Install a fake ``pynput.keyboard`` module and return its controller."""
    controller = _FakeController()

    keyboard_mod = types.ModuleType("pynput.keyboard")
    keyboard_mod.Controller = lambda: controller  # type: ignore[attr-defined]
    keyboard_mod.Key = _FakeKey  # type: ignore[attr-defined]

    pynput_mod = types.ModuleType("pynput")
    pynput_mod.keyboard = keyboard_mod  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "pynput", pynput_mod)
    monkeypatch.setitem(sys.modules, "pynput.keyboard", keyboard_mod)
    return controller


async def test_system_media_emits_media_keys_via_fake_pynput(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _install_fake_pynput(monkeypatch)
    media = SystemMedia()
    tool = MediaTool(controller=media)

    result = await tool(MediaArgs(action="next"))
    assert result.ok is True
    assert controller.events == ["press:next", "release:next"]


async def test_system_media_volume_steps_toward_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _install_fake_pynput(monkeypatch)
    # baseline 50, step 5 -> raising to 70 is (70-50)/5 = 4 up-taps.
    media = SystemMedia(baseline=50, step=5)
    tool = MediaTool(controller=media)

    result = await tool(MediaArgs(action="volume", level=70))
    assert result.ok is True
    up_presses = [e for e in controller.events if e == "press:vol_up"]
    assert len(up_presses) == 4


async def test_system_media_missing_pynput_returns_typed_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Force the lazy import to fail as though pynput is not installed.
    monkeypatch.setitem(sys.modules, "pynput", None)
    media = SystemMedia()
    tool = MediaTool(controller=media)

    result = await tool(MediaArgs(action="play"))
    assert result.ok is False
    assert result.error is not None
    assert result.error.code == "media_backend_unavailable"
    assert result.error.retriable is False


def test_system_media_keyboard_raises_unavailable_on_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "pynput", None)
    with pytest.raises(MediaBackendUnavailable):
        SystemMedia()._keyboard()


def _is_controller(obj: Any) -> bool:
    return isinstance(obj, MediaController)
