"""Desktop-control subsystem: drive the mouse, keyboard, and screenshots.

This package owns the typed boundary for FRIDAY's desktop control plus a
deterministic fake, a fully-audited wrapper, and a lazily-imported real adapter:

* :class:`~friday.desktop.control.DesktopController` — the runtime-checkable
  ``move_to`` / ``click`` / ``type_text`` / ``hotkey`` / ``screenshot`` protocol.
* :class:`~friday.desktop.control.FakeDesktop` — records every action into a
  list, so tests run with no real input devices or display.
* :class:`~friday.desktop.control.AuditedDesktop` — wraps any controller so
  EVERY action is recorded to an injected audit sink **before** it executes
  (frictionless, no per-action prompt, but fully audited).
* :class:`~friday.desktop.control.PyAutoGuiDesktop` — the real adapter that
  lazy-imports ``pyautogui``, arms ``pyautogui.FAILSAFE = True`` (the
  slam-to-corner runaway abort), and raises a clear install error when the
  backend is missing.

Desktop control is **side-effecting** (it can move your mouse and type for you),
so the real adapter lazy-imports its backend *inside* methods and raises a clear
``pip install pyautogui`` / ``make install-perception`` hint when it is absent —
importing this package never requires ``pyautogui`` and ``uv sync`` stays
unaffected. No LLM SDK is imported anywhere in this package (architecture guard).
"""

from __future__ import annotations

from friday.desktop.control import (
    AuditedDesktop,
    AuditSink,
    DesktopAction,
    DesktopController,
    FakeDesktop,
    PyAutoGuiDesktop,
)

__all__ = [
    "AuditSink",
    "AuditedDesktop",
    "DesktopAction",
    "DesktopController",
    "FakeDesktop",
    "PyAutoGuiDesktop",
]
