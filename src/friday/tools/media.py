"""Media-control tool: play / pause / next / prev / volume over a controller seam.

The transport is abstracted behind the :class:`MediaController` protocol so the
tool never reaches for a backend directly:

* :class:`FakeMedia` is the in-memory default — it records every command and
  tracks a play/pause state plus a clamped volume, so tests (and an offline
  desktop) drive the full surface without touching hardware.
* :class:`SystemMedia` is the real adapter. It emits OS media keys via the
  optional ``pynput`` keyboard backend, which is **lazy-imported inside the
  method that needs it** — so importing this module never pulls in ``pynput``
  and the package stays an optional, not a hard, dependency. When the backend is
  unavailable the adapter raises :class:`MediaBackendUnavailable`, which the tool
  surfaces as a typed ``media_backend_unavailable`` failure.

:class:`MediaTool` is ``side_effecting=True`` but ``idempotent=True`` — toggling
playback or stepping volume is a transport command with no destructive,
non-repeatable effect, so it does not trip the registry confirm-step. The
controller is injected (dependency injection); nothing here imports app config.
"""

from __future__ import annotations

import logging
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from friday.tools.base import ToolError, ToolResult

logger = logging.getLogger("friday.tools.media")

#: The transport commands the tool/controller understand.
MediaAction = Literal["play", "pause", "next", "prev", "volume"]


class MediaBackendUnavailable(RuntimeError):
    """Raised by a real controller when its OS media backend cannot be used."""


class MediaArgs(BaseModel):
    """Arguments for :class:`MediaTool`.

    ``action`` selects the transport command. ``level`` is required only for the
    ``volume`` action and is a 0-100 target; it is ignored for the others.
    """

    action: MediaAction
    level: int | None = Field(default=None, ge=0, le=100)


@runtime_checkable
class MediaController(Protocol):
    """The seam a concrete media backend implements.

    Each method performs (or, for the fake, records) one transport command.
    ``volume`` receives a clamped 0-100 target. Implementations may raise
    :class:`MediaBackendUnavailable` when their backend is not usable.
    """

    async def play(self) -> None:
        """Resume / start playback."""
        ...

    async def pause(self) -> None:
        """Pause playback."""
        ...

    async def next(self) -> None:
        """Skip to the next track."""
        ...

    async def prev(self) -> None:
        """Skip to the previous track."""
        ...

    async def volume(self, level: int) -> None:
        """Set the volume to ``level`` (0-100)."""
        ...


class FakeMedia:
    """In-memory media controller: records commands, tracks state + volume.

    ``commands`` is the ordered log of every command received (the ``volume``
    entry is recorded as ``"volume:<level>"``). ``playing`` reflects the last
    play/pause command and ``level`` the last volume set (clamped to 0-100).
    """

    def __init__(self, *, level: int = 50) -> None:
        self.commands: list[str] = []
        self.playing: bool = False
        self.level: int = max(0, min(100, level))

    async def play(self) -> None:
        self.commands.append("play")
        self.playing = True

    async def pause(self) -> None:
        self.commands.append("pause")
        self.playing = False

    async def next(self) -> None:
        self.commands.append("next")

    async def prev(self) -> None:
        self.commands.append("prev")

    async def volume(self, level: int) -> None:
        clamped = max(0, min(100, level))
        self.commands.append(f"volume:{clamped}")
        self.level = clamped


class SystemMedia:
    """Real media controller emitting OS media keys via ``pynput`` (lazy).

    The ``pynput`` keyboard backend is imported *inside* the helper that taps a
    key, so simply constructing or importing this class never requires the
    package. ``volume`` steps the system volume up/down toward ``level`` from a
    tracked baseline using the volume-up/down media keys (most desktops expose no
    absolute "set volume" key).

    Raises:
        MediaBackendUnavailable: when ``pynput`` is not installed or the OS key
            cannot be sent.
    """

    def __init__(self, *, baseline: int = 50, step: int = 5) -> None:
        self._baseline = max(0, min(100, baseline))
        self._step = max(1, step)

    def _keyboard(self) -> Any:
        """Lazily import ``pynput`` and return a keyboard Controller + keys.

        Importing here (not at module top) keeps ``pynput`` an optional dep.
        """
        try:
            from pynput.keyboard import (  # type: ignore[import-untyped]
                Controller,
                Key,
            )
        except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
            raise MediaBackendUnavailable(
                "media key backend 'pynput' is not installed"
            ) from exc
        return Controller(), Key

    def _tap(self, key: Any) -> None:
        """Press+release one media key, mapping backend errors to our type."""
        controller, _ = self._keyboard()
        try:
            controller.press(key)
            controller.release(key)
        except Exception as exc:  # pragma: no cover - backend specific
            raise MediaBackendUnavailable(f"failed to send media key: {exc}") from exc

    async def play(self) -> None:
        _, key = self._keyboard()
        self._tap(key.media_play_pause)

    async def pause(self) -> None:
        _, key = self._keyboard()
        self._tap(key.media_play_pause)

    async def next(self) -> None:
        _, key = self._keyboard()
        self._tap(key.media_next)

    async def prev(self) -> None:
        _, key = self._keyboard()
        self._tap(key.media_previous)

    async def volume(self, level: int) -> None:
        _, key = self._keyboard()
        target = max(0, min(100, level))
        delta = target - self._baseline
        taps = abs(delta) // self._step
        media_key = key.media_volume_up if delta >= 0 else key.media_volume_down
        for _ in range(taps):
            self._tap(media_key)
        self._baseline = target


class MediaTool:
    """Drive a :class:`MediaController` (play/pause/next/prev/volume).

    Args:
        controller: The transport backend. Defaults to a fresh :class:`FakeMedia`
            so the tool is usable (and testable) out of the box; inject a
            :class:`SystemMedia` for real media keys.
    """

    name = "media"
    description = "Control media playback: play, pause, next, prev, or set volume."
    args_model = MediaArgs
    required_permission = "media"
    idempotent = True
    side_effecting = True

    def __init__(self, controller: MediaController | None = None) -> None:
        self._controller: MediaController = controller or FakeMedia()

    async def __call__(self, args: Any) -> ToolResult:
        """Dispatch ``action`` to the controller, surfacing backend failures."""
        # ``args`` arrives validated from the registry; coerce defensively.
        if not isinstance(args, MediaArgs):
            args = MediaArgs.model_validate(args)

        if args.action == "volume" and args.level is None:
            return ToolResult(
                ok=False,
                data={},
                error=ToolError(
                    code="missing_level",
                    message="the 'volume' action requires a 'level' (0-100)",
                    retriable=False,
                ),
            )

        try:
            await self._dispatch(args)
        except MediaBackendUnavailable as exc:
            logger.warning("media backend unavailable: %s", exc)
            return ToolResult(
                ok=False,
                data={},
                error=ToolError(
                    code="media_backend_unavailable",
                    message=str(exc),
                    retriable=False,
                ),
            )

        data: dict[str, Any] = {"action": args.action}
        if args.action == "volume":
            data["level"] = args.level
        return ToolResult(ok=True, data=data, error=None)

    async def _dispatch(self, args: MediaArgs) -> None:
        """Route the validated ``action`` to the controller method."""
        if args.action == "play":
            await self._controller.play()
        elif args.action == "pause":
            await self._controller.pause()
        elif args.action == "next":
            await self._controller.next()
        elif args.action == "prev":
            await self._controller.prev()
        else:  # "volume" — level guaranteed non-None by the caller's guard.
            assert args.level is not None
            await self._controller.volume(args.level)
