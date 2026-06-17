# © Lakshya Badjatya — Author
"""System-tray surface — a small icon to open the HUD / notify / quit.

Mirrors the desktop-control seam: a runtime-checkable :class:`TrayController`
protocol, a deterministic :class:`FakeTray` for tests / headless builds, and a
lazily-imported real :class:`PyStrayTray` adapter that touches ``pystray`` + PIL
only inside its methods — so importing this module never needs the GUI backend
and the offline build constructs no tray.

The tray's event loop *blocks*, so it is launched out-of-band by ``friday tray``
rather than inside the ASGI lifespan. :func:`build_tray` returns ``None`` unless
``enable_tray`` is on, keeping wiring a one-liner.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from friday.config import Settings

logger = logging.getLogger("friday.desktop.tray")


@runtime_checkable
class TrayController(Protocol):
    """A system-tray icon: notify the user and run its (blocking) event loop."""

    def notify(self, title: str, message: str) -> None:
        """Show a desktop notification from the tray."""
        ...

    def run(self) -> None:
        """Run the tray's event loop (blocks until the user quits)."""
        ...


class FakeTray:
    """An in-memory tray that records notifications; ``run`` is a no-op.

    Lets the surface be exercised headlessly (and is what a build without
    ``pystray`` degrades to). ``ran`` flips ``True`` so a caller can assert the
    loop was entered without actually blocking on a GUI.
    """

    def __init__(self, title: str = "FRIDAY", hud_url: str = "") -> None:
        self.title = title
        self.hud_url = hud_url
        self.notifications: list[tuple[str, str]] = []
        self.ran = False

    def notify(self, title: str, message: str) -> None:
        """Record the notification in-memory."""
        self.notifications.append((title, message))

    def run(self) -> None:
        """No-op loop: mark that it was entered and return immediately."""
        self.ran = True


class PyStrayTray:
    """The real tray adapter — lazy-imports ``pystray`` + PIL inside its methods.

    A menu with "Open HUD" (launches the default browser at ``hud_url``) and
    "Quit". Construction never imports the backend; the first :meth:`run` does, so
    a missing dependency surfaces as a clear, catchable :class:`ImportError`.
    """

    def __init__(self, title: str = "FRIDAY", hud_url: str = "") -> None:
        self.title = title
        self.hud_url = hud_url
        self._icon: Any = None

    def _build_icon(self) -> Any:
        import webbrowser  # noqa: PLC0415

        import pystray  # type: ignore[import-not-found]  # noqa: PLC0415
        from PIL import Image, ImageDraw  # type: ignore[import-not-found]  # noqa: PLC0415

        image = Image.new("RGB", (64, 64), (6, 9, 16))
        draw = ImageDraw.Draw(image)
        draw.ellipse((12, 12, 52, 52), outline=(79, 227, 255), width=4)

        def _open_hud(_icon: object, _item: object) -> None:
            if self.hud_url:
                webbrowser.open(self.hud_url)

        menu = pystray.Menu(
            pystray.MenuItem("Open HUD", _open_hud),
            pystray.MenuItem("Quit", lambda icon, _item: icon.stop()),
        )
        return pystray.Icon(self.title, image, self.title, menu)

    def notify(self, title: str, message: str) -> None:
        """Show a notification via the running icon (best-effort)."""
        if self._icon is not None:
            try:
                self._icon.notify(message, title)
            except Exception as exc:  # noqa: BLE001 - notification is best-effort
                logger.warning("tray notify failed: %s", exc)

    def run(self) -> None:
        """Build the icon and run pystray's blocking loop."""
        self._icon = self._build_icon()
        self._icon.run()


def build_tray(settings: Settings) -> TrayController | None:
    """Return a real tray when ``enable_tray`` is on, else ``None``.

    Falls back to :class:`FakeTray` (logged) if the GUI backend can't be loaded,
    so ``friday tray`` on a headless box degrades instead of crashing.
    """
    if not settings.enable_tray:
        return None
    tray = PyStrayTray(settings.tray_title, settings.tray_hud_url)
    try:
        tray._build_icon()  # probe the backend up front
    except ImportError:
        logger.warning(
            "enable_tray is set but pystray/PIL is not installed; using a no-op "
            "tray (install the desktop extras for a real tray icon)"
        )
        return FakeTray(settings.tray_title, settings.tray_hud_url)
    return tray
