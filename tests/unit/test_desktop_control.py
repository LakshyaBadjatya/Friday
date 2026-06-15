"""Unit tests for the desktop-control boundary, fakes, audit wrapper, and adapter.

Pins the desktop contract:

* importing the module requires NO heavy library (``pyautogui`` is optional);
* :class:`FakeDesktop` records every action deterministically with no real
  mouse/keyboard/display;
* :class:`AuditedDesktop` records EVERY action to the injected sink *before*
  executing it on the wrapped controller (frictionless but fully audited);
* :class:`PyAutoGuiDesktop` lazy-imports ``pyautogui`` and raises a clear
  ``pip install pyautogui`` / ``make install-perception`` error when it is absent
  (the import is monkeypatched to fail).

No real screen, no real input devices, no network.
"""

from __future__ import annotations

import builtins
import importlib
import sys
from typing import Any

import pytest

from friday.desktop.control import (
    AuditedDesktop,
    DesktopAction,
    DesktopController,
    FakeDesktop,
    PyAutoGuiDesktop,
)
from friday.errors import ProviderError


def _fail_import_of(*modules: str) -> Any:
    """Build a fake ``__import__`` that raises ImportError for ``modules``."""
    real_import = builtins.__import__
    blocked = tuple(modules)

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name in blocked or any(name.startswith(f"{m}.") for m in blocked):
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    return fake_import


# --------------------------------------------------------------------------- #
# Importing the module requires NO heavy library
# --------------------------------------------------------------------------- #
def test_module_import_requires_no_heavy_lib() -> None:
    importlib.import_module("friday.desktop.control")
    assert "pyautogui" not in sys.modules


# --------------------------------------------------------------------------- #
# Protocol membership
# --------------------------------------------------------------------------- #
def test_fakes_satisfy_protocol() -> None:
    assert isinstance(FakeDesktop(), DesktopController)
    assert isinstance(AuditedDesktop(FakeDesktop(), []), DesktopController)


# --------------------------------------------------------------------------- #
# FakeDesktop records every action
# --------------------------------------------------------------------------- #
def test_fake_desktop_records_every_action() -> None:
    desk = FakeDesktop()
    desk.move_to(10, 20)
    desk.click(30, 40)
    desk.type_text("hi")
    desk.hotkey("ctrl", "c")
    shot = desk.screenshot()

    assert isinstance(shot, bytes)
    assert [a.kind for a in desk.actions] == [
        "move_to",
        "click",
        "type_text",
        "hotkey",
        "screenshot",
    ]
    assert desk.actions[0].args == {"x": 10, "y": 20}
    assert desk.actions[1].args == {"x": 30, "y": 40}
    assert desk.actions[2].args == {"text": "hi"}
    assert desk.actions[3].args == {"keys": ["ctrl", "c"]}
    assert desk.actions[4].args == {}


def test_fake_desktop_screenshot_returns_configured_payload() -> None:
    desk = FakeDesktop(screenshot_payload=b"PNGDATA")
    assert desk.screenshot() == b"PNGDATA"


# --------------------------------------------------------------------------- #
# DesktopAction model
# --------------------------------------------------------------------------- #
def test_desktop_action_constructs() -> None:
    action = DesktopAction(kind="click", args={"x": 1, "y": 2})
    assert action.kind == "click"
    assert action.args == {"x": 1, "y": 2}


# --------------------------------------------------------------------------- #
# AuditedDesktop: audit BEFORE execute, frictionless (no prompt)
# --------------------------------------------------------------------------- #
def test_audited_desktop_audits_then_executes_each_action() -> None:
    inner = FakeDesktop()
    sink: list[DesktopAction] = []
    audited = AuditedDesktop(inner, sink)

    audited.move_to(1, 2)
    audited.click(3, 4)
    audited.type_text("password")
    audited.hotkey("alt", "tab")
    shot = audited.screenshot()

    assert isinstance(shot, bytes)
    # Every action reached the audit sink, in order, with full args.
    assert [a.kind for a in sink] == [
        "move_to",
        "click",
        "type_text",
        "hotkey",
        "screenshot",
    ]
    assert sink[2].args == {"text": "password"}
    assert sink[3].args == {"keys": ["alt", "tab"]}
    # And every action reached the wrapped controller too.
    assert [a.kind for a in inner.actions] == [
        "move_to",
        "click",
        "type_text",
        "hotkey",
        "screenshot",
    ]


def test_audited_desktop_records_before_executing() -> None:
    """The sink entry is written BEFORE the wrapped controller runs the action."""
    order: list[str] = []

    class _RecordingSink:
        def append(self, action: DesktopAction) -> None:
            order.append(f"audit:{action.kind}")

    class _RecordingInner:
        def move_to(self, x: int, y: int) -> None:
            order.append("exec:move_to")

        def click(self, x: int, y: int) -> None:
            order.append("exec:click")

        def type_text(self, text: str) -> None:
            order.append("exec:type_text")

        def hotkey(self, *keys: str) -> None:
            order.append("exec:hotkey")

        def screenshot(self) -> bytes:
            order.append("exec:screenshot")
            return b""

    audited = AuditedDesktop(_RecordingInner(), _RecordingSink())
    audited.click(5, 6)

    assert order == ["audit:click", "exec:click"]


def test_audited_desktop_audits_even_when_execution_raises() -> None:
    """A runaway/failed action is still recorded before the failure surfaces."""
    sink: list[DesktopAction] = []

    class _Boom:
        def click(self, x: int, y: int) -> None:
            raise RuntimeError("fail-safe triggered")

        def move_to(self, x: int, y: int) -> None: ...
        def type_text(self, text: str) -> None: ...
        def hotkey(self, *keys: str) -> None: ...
        def screenshot(self) -> bytes:
            return b""

    audited = AuditedDesktop(_Boom(), sink)
    with pytest.raises(RuntimeError):
        audited.click(1, 1)
    # The audit happened before the executing controller blew up.
    assert [a.kind for a in sink] == ["click"]
    assert sink[0].args == {"x": 1, "y": 1}


def test_audited_desktop_accepts_callable_sink() -> None:
    """The sink may be any object exposing ``append`` (e.g. a plain list)."""
    recorded: list[DesktopAction] = []
    audited = AuditedDesktop(FakeDesktop(), recorded)
    audited.type_text("abc")
    assert recorded[0].args == {"text": "abc"}


# --------------------------------------------------------------------------- #
# PyAutoGuiDesktop: lazy import, FAILSAFE, helpful error when backend missing
# --------------------------------------------------------------------------- #
def test_pyautogui_desktop_construction_is_import_light() -> None:
    # Constructing the adapter must not import pyautogui.
    PyAutoGuiDesktop()
    assert "pyautogui" not in sys.modules


@pytest.mark.parametrize(
    "invoke",
    [
        lambda d: d.move_to(1, 2),
        lambda d: d.click(1, 2),
        lambda d: d.type_text("x"),
        lambda d: d.hotkey("ctrl", "c"),
        lambda d: d.screenshot(),
    ],
)
def test_pyautogui_missing_backend_raises_helpful_error(
    invoke: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    desk = PyAutoGuiDesktop()
    monkeypatch.setattr(builtins, "__import__", _fail_import_of("pyautogui"))
    with pytest.raises(ProviderError) as exc:
        invoke(desk)
    message = str(exc.value)
    assert "pyautogui" in message
    assert "install-perception" in message or "pip install pyautogui" in message


def test_pyautogui_sets_failsafe_and_dispatches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real adapter enables FAILSAFE and forwards calls to pyautogui."""
    calls: list[tuple[str, tuple[Any, ...]]] = []

    class _FakePyAutoGui:
        FAILSAFE = False

        def moveTo(self, x: int, y: int) -> None:
            calls.append(("moveTo", (x, y)))

        def click(self, x: int, y: int) -> None:
            calls.append(("click", (x, y)))

        def typewrite(self, text: str) -> None:
            calls.append(("typewrite", (text,)))

        def hotkey(self, *keys: str) -> None:
            calls.append(("hotkey", keys))

        def screenshot(self) -> Any:
            calls.append(("screenshot", ()))
            return _FakeImage()

    class _FakeImage:
        def save(self, buffer: Any, format: str) -> None:
            buffer.write(b"FAKEPNG")

    fake = _FakePyAutoGui()
    monkeypatch.setitem(sys.modules, "pyautogui", fake)

    desk = PyAutoGuiDesktop()
    desk.move_to(7, 8)
    desk.click(9, 10)
    desk.type_text("hello")
    desk.hotkey("ctrl", "s")
    data = desk.screenshot()

    # FAILSAFE must be armed: slamming the cursor to a corner aborts a runaway.
    assert fake.FAILSAFE is True
    assert data == b"FAKEPNG"
    assert calls == [
        ("moveTo", (7, 8)),
        ("click", (9, 10)),
        ("typewrite", ("hello",)),
        ("hotkey", ("ctrl", "s")),
        ("screenshot", ()),
    ]
