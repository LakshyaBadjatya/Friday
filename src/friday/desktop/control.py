"""Desktop-control boundary, recording fake, audit wrapper, and lazy real adapter.

FRIDAY drives the desktop (mouse, keyboard, screenshots) the same way it drives
every other side-effecting capability: through a small, runtime-checkable
protocol with a deterministic fake for tests and a lazily-imported real adapter.

* :class:`DesktopController` — the runtime-checkable contract:
  ``move_to`` / ``click`` / ``type_text`` / ``hotkey`` / ``screenshot``.
* :class:`DesktopAction` — a typed record of one desktop action (used by the
  fake and the audit wrapper).
* :class:`FakeDesktop` — an in-memory controller that records every action into
  a list, so tests run with no real mouse/keyboard/display.
* :class:`AuditedDesktop` — wraps any controller so EVERY action is recorded to
  an injected audit sink **before** it is executed. This is *frictionless* (no
  per-action confirmation prompt) but *fully audited*; the runaway abort is the
  fail-safe of the underlying adapter (slam the cursor into a screen corner),
  not a synchronous prompt.
* :class:`PyAutoGuiDesktop` — the real adapter that lazy-imports ``pyautogui``
  inside its methods, arms ``pyautogui.FAILSAFE = True`` (the slam-to-corner
  abort), and raises a clear install error (``pip install pyautogui`` /
  ``make install-perception``) when the backend is absent.

No heavy library is imported at module top level, so importing this module never
requires ``pyautogui`` and ``uv sync`` stays unaffected. No LLM SDK is imported
anywhere in this module (architecture guard).
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from friday.errors import ProviderError

_INSTALL_HINT = (
    "pyautogui is not installed. Desktop control is an optional extra and is "
    "excluded from the uv lock; install it with `make install-perception` "
    "(or `pip install pyautogui`)."
)


@runtime_checkable
class DesktopController(Protocol):
    """Contract for driving the desktop: pointer, keyboard, and screenshots.

    Coordinates are absolute screen pixels. ``type_text`` types a literal string;
    ``hotkey`` presses a chord (e.g. ``hotkey("ctrl", "c")``); ``screenshot``
    returns encoded image bytes (e.g. PNG).
    """

    def move_to(self, x: int, y: int) -> None:
        """Move the pointer to absolute screen coordinates ``(x, y)``."""
        ...

    def click(self, x: int, y: int) -> None:
        """Click at absolute screen coordinates ``(x, y)``."""
        ...

    def type_text(self, text: str) -> None:
        """Type ``text`` as a sequence of literal key presses."""
        ...

    def hotkey(self, *keys: str) -> None:
        """Press ``keys`` together as a chord (e.g. ``"ctrl", "c"``)."""
        ...

    def screenshot(self) -> bytes:
        """Capture the screen and return encoded image bytes."""
        ...


class DesktopAction(BaseModel):
    """A typed record of a single desktop action.

    Attributes:
        kind: The action name (``"move_to"``, ``"click"``, ``"type_text"``,
            ``"hotkey"``, or ``"screenshot"``).
        args: The action's arguments, keyed by parameter name. A ``screenshot``
            carries no arguments, so ``args`` is empty for it.
    """

    kind: str
    args: dict[str, Any] = Field(default_factory=dict)


@runtime_checkable
class AuditSink(Protocol):
    """A sink that records desktop actions for audit.

    A plain ``list`` satisfies this protocol (``list.append``), so tests and the
    integration pass can inject ``[]`` directly; a durable sink (file/db) just
    needs an ``append`` taking a :class:`DesktopAction`.
    """

    def append(self, action: DesktopAction) -> None:
        """Record one audited desktop action."""
        ...


class FakeDesktop:
    """An in-memory :class:`DesktopController` that records every action.

    Each call appends a :class:`DesktopAction` to :attr:`actions`, so tests can
    assert exactly what would have been done with no real mouse, keyboard, or
    display. :meth:`screenshot` returns a fixed payload.
    """

    def __init__(self, screenshot_payload: bytes = b"FRIDAY_FAKE_SCREENSHOT") -> None:
        """Create the fake desktop.

        Args:
            screenshot_payload: The bytes returned by :meth:`screenshot`.
        """
        self.screenshot_payload = screenshot_payload
        self.actions: list[DesktopAction] = []

    def move_to(self, x: int, y: int) -> None:
        self.actions.append(DesktopAction(kind="move_to", args={"x": x, "y": y}))

    def click(self, x: int, y: int) -> None:
        self.actions.append(DesktopAction(kind="click", args={"x": x, "y": y}))

    def type_text(self, text: str) -> None:
        self.actions.append(DesktopAction(kind="type_text", args={"text": text}))

    def hotkey(self, *keys: str) -> None:
        self.actions.append(DesktopAction(kind="hotkey", args={"keys": list(keys)}))

    def screenshot(self) -> bytes:
        self.actions.append(DesktopAction(kind="screenshot"))
        return self.screenshot_payload


class AuditedDesktop:
    """Wrap a :class:`DesktopController` to audit EVERY action before executing.

    For each call, an :class:`DesktopAction` describing the call is appended to
    the injected ``audit_sink`` **first**, and only then is the wrapped
    controller's method invoked. This makes desktop control *frictionless* (no
    per-action confirmation prompt blocks the agent) while remaining *fully
    audited* — the audit trail is written even if the executing action later
    raises (e.g. the slam-to-corner fail-safe aborts a runaway).

    Both the wrapped controller and the sink are injected, so the audited path
    is composed entirely from in-memory fakes in tests.
    """

    def __init__(self, inner: DesktopController, audit_sink: AuditSink) -> None:
        """Assemble the audited controller.

        Args:
            inner: The controller that actually performs each action.
            audit_sink: The sink each action is recorded to before execution
                (a plain ``list`` works).
        """
        self.inner = inner
        self.audit_sink = audit_sink

    def _audit(self, kind: str, args: dict[str, Any]) -> None:
        """Record ``kind``/``args`` to the sink before the action executes."""
        self.audit_sink.append(DesktopAction(kind=kind, args=args))

    def move_to(self, x: int, y: int) -> None:
        self._audit("move_to", {"x": x, "y": y})
        self.inner.move_to(x, y)

    def click(self, x: int, y: int) -> None:
        self._audit("click", {"x": x, "y": y})
        self.inner.click(x, y)

    def type_text(self, text: str) -> None:
        self._audit("type_text", {"text": text})
        self.inner.type_text(text)

    def hotkey(self, *keys: str) -> None:
        self._audit("hotkey", {"keys": list(keys)})
        self.inner.hotkey(*keys)

    def screenshot(self) -> bytes:
        self._audit("screenshot", {})
        return self.inner.screenshot()


class PyAutoGuiDesktop:
    """Real :class:`DesktopController` backed by ``pyautogui`` (lazy).

    The ``pyautogui`` import happens inside the methods, so importing this module
    never requires the backend. On every entry the adapter arms
    ``pyautogui.FAILSAFE = True`` — slamming the cursor into a screen corner
    raises and aborts a runaway sequence. When the backend is missing, a
    :class:`friday.errors.ProviderError` is raised with a clear
    ``pip install pyautogui`` / ``make install-perception`` hint.
    """

    def _pyautogui(self) -> Any:
        """Lazy-import ``pyautogui``, arm FAILSAFE, and return the module.

        Raises:
            ProviderError: If ``pyautogui`` is not installed.
        """
        try:
            # Optional desktop-control backend: excluded from the uv lock, so
            # mypy has no stub for it; lazily imported here and guarded by the
            # ImportError. Typed as ``Any`` so the dynamic attribute access below
            # stays clean under --strict.
            import pyautogui  # type: ignore[import-untyped, import-not-found, unused-ignore]  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
            raise ProviderError(_INSTALL_HINT) from exc
        # Slam-cursor-to-corner aborts a runaway action sequence.
        pyautogui.FAILSAFE = True
        return pyautogui

    def move_to(self, x: int, y: int) -> None:
        self._pyautogui().moveTo(x, y)

    def click(self, x: int, y: int) -> None:
        self._pyautogui().click(x, y)

    def type_text(self, text: str) -> None:
        self._pyautogui().typewrite(text)

    def hotkey(self, *keys: str) -> None:
        self._pyautogui().hotkey(*keys)

    def screenshot(self) -> bytes:
        """Capture the screen and return PNG-encoded bytes."""
        from io import BytesIO

        image = self._pyautogui().screenshot()
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()
