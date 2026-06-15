"""BLE proximity scanning boundary: model, protocol, fixture fake, lazy adapter.

* :class:`Device` — a strict pydantic v2 model of a single BLE advertisement
  (``address`` always present; ``name`` / ``rssi`` optional, as a backend may not
  surface either).
* :class:`PresenceScanner` — the runtime-checkable ``async scan()`` protocol the
  service reads through: the real :class:`BleakScanner` in production, the
  :class:`FakePresenceScanner` in tests.
* :class:`FakePresenceScanner` — replays scripted scans (then repeats the last),
  so the whole presence path runs offline with no Bluetooth hardware.
* :class:`BleakScanner` — the real adapter that **lazy-imports** ``bleak`` inside
  :meth:`scan` and raises a clear :class:`RuntimeError` (with an install hint)
  when the backend is absent.

``bleak`` is intentionally **NOT** in the uv lock (it needs a Bluetooth stack and
platform-specific wheels); it is lazy-imported here so importing this module — and
therefore the whole app — never requires it and ``uv sync`` stays unaffected.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

# ``bleak`` is deliberately excluded from the uv lock: it pulls a platform BLE
# stack and is only needed for a live hardware scan. It is lazy-imported inside
# ``BleakScanner.scan`` (guarded by the ImportError below), so this module imports
# cleanly without it. Install it out-of-band (e.g. ``uv pip install bleak``) on a
# host with Bluetooth when a real scan is wanted.
_INSTALL_HINT = (
    "bleak is not installed. The BLE backend is excluded from the uv lock; "
    "install it out-of-band on a Bluetooth-capable host (e.g. `uv pip install "
    "bleak`) to enable live presence scanning."
)


class Device(BaseModel):
    """A single BLE advertisement seen during a scan.

    ``address`` is the device MAC (always present); ``name`` and ``rssi`` are
    optional because a backend may advertise neither. Strict/extra-forbidding so
    a malformed backend payload fails loudly rather than smuggling stray fields.
    """

    model_config = ConfigDict(extra="forbid")

    address: str
    name: str | None = None
    rssi: int | None = None


@runtime_checkable
class PresenceScanner(Protocol):
    """Anything that can return the BLE devices currently in range.

    Structural (``Protocol``) so a test fake needs only an ``async scan()``
    method, no inheritance.
    """

    async def scan(self) -> list[Device]:
        """Return the devices visible right now (empty when nothing is in range)."""
        ...


class FakePresenceScanner:
    """A deterministic :class:`PresenceScanner` replaying scripted scans.

    Each :meth:`scan` call returns the next scripted scan; once the script is
    exhausted it repeats the last one (so a service can be ticked indefinitely),
    and an empty script always returns ``[]``. :attr:`calls` records how many
    times :meth:`scan` was invoked.
    """

    def __init__(self, scans: list[list[Device]]) -> None:
        """Create the fake.

        Args:
            scans: The scripted scans, returned one per :meth:`scan` call in
                order. After the last, that last scan repeats.
        """
        self._scans = scans
        self.calls = 0

    async def scan(self) -> list[Device]:
        index = self.calls
        self.calls += 1
        if not self._scans:
            return []
        if index >= len(self._scans):
            return list(self._scans[-1])
        return list(self._scans[index])


class BleakScanner:
    """The real :class:`PresenceScanner` backed by ``bleak`` (lazy).

    The heavy ``bleak`` import happens inside :meth:`scan`, so importing this
    module never requires the backend. When ``bleak`` is missing a
    :class:`RuntimeError` is raised with an install hint; otherwise the backend's
    discovered devices are mapped to :class:`Device`.
    """

    def __init__(self, *, timeout: float = 5.0) -> None:
        """Construct the adapter.

        Args:
            timeout: Per-scan discovery budget (seconds) passed to ``bleak``.
        """
        self.timeout = timeout

    async def scan(self) -> list[Device]:
        """Discover BLE devices in range and map them to :class:`Device`."""
        try:
            # Optional BLE backend: excluded from the uv lock, so mypy has no stub
            # for it; lazily imported as a plain module (keeps the whole-statement
            # ``# type: ignore`` on one line ruff won't wrap) and guarded below.
            import bleak  # type: ignore[import-not-found, import-untyped, unused-ignore]  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
            raise RuntimeError(_INSTALL_HINT) from exc

        found = await bleak.BleakScanner.discover(timeout=self.timeout)
        return [
            Device(
                address=str(dev.address),
                name=getattr(dev, "name", None),
                rssi=getattr(dev, "rssi", None),
            )
            for dev in found
        ]
