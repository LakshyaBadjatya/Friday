"""Unit tests for the presence feature (Tier 3): scanner + service.

All offline: the :class:`~friday.presence.scanner.FakePresenceScanner` returns
scripted scans, so no Bluetooth backend (``bleak``) is ever touched. The
:class:`~friday.presence.service.PresenceService` is driven across those scripted
scans and its arrival/departure transitions asserted; a fixed clock is injected
so the ``ts`` on each result is deterministic.

Covered:
* :class:`Device` is a strict pydantic v2 model.
* ``FakePresenceScanner`` yields its scripted scans in order, then repeats the
  last, and records call count.
* ``PresenceService.update`` maps seen MAC addresses to known names and reports
  the current present/absent split.
* Arrival is detected on the scan a known device first appears.
* Departure is detected on the scan a known device first disappears.
* Steady state (no change) yields empty arrived/departed.
* The real :class:`BleakScanner` adapter raises a clear error when ``bleak`` is
  missing (lazy import), and otherwise maps backend devices to :class:`Device`.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from friday.presence.scanner import (
    BleakScanner,
    Device,
    FakePresenceScanner,
    PresenceScanner,
)
from friday.presence.service import PresenceService, PresenceUpdate


# --------------------------------------------------------------------------- #
# Device model
# --------------------------------------------------------------------------- #
def test_device_is_strict_model() -> None:
    dev = Device(address="AA:BB:CC:DD:EE:FF", name="Phone", rssi=-40)
    assert dev.address == "AA:BB:CC:DD:EE:FF"
    assert dev.name == "Phone"
    assert dev.rssi == -40
    # name/rssi are optional (a backend may not surface them).
    bare = Device(address="11:22:33:44:55:66")
    assert bare.name is None
    assert bare.rssi is None


# --------------------------------------------------------------------------- #
# FakePresenceScanner
# --------------------------------------------------------------------------- #
def _clock() -> datetime:
    return datetime(2026, 6, 15, 12, 0, 0, tzinfo=UTC)


async def test_fake_scanner_yields_scripted_scans_in_order() -> None:
    phone = Device(address="AA:BB:CC:DD:EE:FF", name="Phone", rssi=-50)
    laptop = Device(address="11:22:33:44:55:66", name="Laptop", rssi=-60)
    scanner = FakePresenceScanner(scans=[[phone], [phone, laptop]])

    assert isinstance(scanner, PresenceScanner)
    assert await scanner.scan() == [phone]
    assert await scanner.scan() == [phone, laptop]
    # Exhausted: repeats the last scripted scan rather than raising.
    assert await scanner.scan() == [phone, laptop]
    assert scanner.calls == 3


async def test_fake_scanner_empty_script_returns_empty() -> None:
    scanner = FakePresenceScanner(scans=[])
    assert await scanner.scan() == []


# --------------------------------------------------------------------------- #
# PresenceService: current present/absent split
# --------------------------------------------------------------------------- #
async def test_update_maps_known_devices_to_names() -> None:
    phone = Device(address="AA:BB:CC:DD:EE:FF", name="bt-phone", rssi=-50)
    stranger = Device(address="99:99:99:99:99:99", name="unknown", rssi=-70)
    scanner = FakePresenceScanner(scans=[[phone, stranger]])
    known = {"AA:BB:CC:DD:EE:FF": "Phone", "11:22:33:44:55:66": "Laptop"}
    service = PresenceService(scanner, known, clock=_clock)

    result = await service.update()

    assert isinstance(result, PresenceUpdate)
    # Present is the known names currently seen; absent the known names not seen.
    assert result.present == ["Phone"]
    assert result.absent == ["Laptop"]
    # An unknown device on the air is ignored (only KNOWN devices are tracked).
    assert "unknown" not in result.present
    assert result.ts == _clock().isoformat()


async def test_update_address_match_is_case_insensitive() -> None:
    phone = Device(address="aa:bb:cc:dd:ee:ff", name="Phone")
    scanner = FakePresenceScanner(scans=[[phone]])
    known = {"AA:BB:CC:DD:EE:FF": "Phone"}
    service = PresenceService(scanner, known, clock=_clock)

    result = await service.update()
    assert result.present == ["Phone"]


# --------------------------------------------------------------------------- #
# Arrival / departure detection across scans
# --------------------------------------------------------------------------- #
async def test_arrival_detected_when_known_device_appears() -> None:
    phone = Device(address="AA:BB:CC:DD:EE:FF", name="Phone")
    # First scan: empty (nobody home). Second scan: phone appears.
    scanner = FakePresenceScanner(scans=[[], [phone]])
    known = {"AA:BB:CC:DD:EE:FF": "Phone"}
    service = PresenceService(scanner, known, clock=_clock)

    first = await service.update()
    assert first.arrived == []
    assert first.departed == []
    assert first.present == []

    second = await service.update()
    assert second.arrived == ["Phone"]
    assert second.departed == []
    assert second.present == ["Phone"]


async def test_departure_detected_when_known_device_disappears() -> None:
    phone = Device(address="AA:BB:CC:DD:EE:FF", name="Phone")
    # First scan: phone present. Second scan: phone gone.
    scanner = FakePresenceScanner(scans=[[phone], []])
    known = {"AA:BB:CC:DD:EE:FF": "Phone"}
    service = PresenceService(scanner, known, clock=_clock)

    first = await service.update()
    assert first.arrived == ["Phone"]
    assert first.present == ["Phone"]

    second = await service.update()
    assert second.arrived == []
    assert second.departed == ["Phone"]
    assert second.present == []
    assert second.absent == ["Phone"]


async def test_steady_state_reports_no_transitions() -> None:
    phone = Device(address="AA:BB:CC:DD:EE:FF", name="Phone")
    scanner = FakePresenceScanner(scans=[[phone], [phone]])
    known = {"AA:BB:CC:DD:EE:FF": "Phone"}
    service = PresenceService(scanner, known, clock=_clock)

    await service.update()
    second = await service.update()
    assert second.arrived == []
    assert second.departed == []
    assert second.present == ["Phone"]


async def test_present_and_absent_are_sorted_and_stable() -> None:
    a = Device(address="AA:AA:AA:AA:AA:AA", name="A")
    c = Device(address="CC:CC:CC:CC:CC:CC", name="C")
    scanner = FakePresenceScanner(scans=[[c, a]])
    known = {
        "AA:AA:AA:AA:AA:AA": "Alpha",
        "BB:BB:BB:BB:BB:BB": "Bravo",
        "CC:CC:CC:CC:CC:CC": "Charlie",
    }
    service = PresenceService(scanner, known, clock=_clock)

    result = await service.update()
    # Deterministic, name-sorted regardless of scan order.
    assert result.present == ["Alpha", "Charlie"]
    assert result.absent == ["Bravo"]


# --------------------------------------------------------------------------- #
# BleakScanner adapter (lazy import; missing backend -> clear error)
# --------------------------------------------------------------------------- #
async def test_bleak_scanner_missing_backend_raises_clear_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import builtins

    real_import = builtins.__import__

    def _no_bleak(name: str, *args: object, **kwargs: object) -> object:
        if name == "bleak" or name.startswith("bleak."):
            raise ImportError("No module named 'bleak'")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", _no_bleak)

    scanner = BleakScanner()
    with pytest.raises(RuntimeError) as exc:
        await scanner.scan()
    # The error names the missing backend and how to install it.
    assert "bleak" in str(exc.value).lower()


async def test_bleak_scanner_maps_backend_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fake ``bleak`` module is injected; the adapter maps it to ``Device``."""
    import sys
    import types

    class _BackendDevice:
        def __init__(self, address: str, name: str | None, rssi: int | None) -> None:
            self.address = address
            self.name = name
            self.rssi = rssi

    discovered = [
        _BackendDevice("AA:BB:CC:DD:EE:FF", "Phone", -55),
        _BackendDevice("11:22:33:44:55:66", None, None),
    ]

    fake_bleak = types.ModuleType("bleak")

    class _FakeBleakScanner:
        @staticmethod
        async def discover(
            *, return_adv: bool = False, timeout: float = 5.0
        ) -> list[_BackendDevice]:
            return discovered

    fake_bleak.BleakScanner = _FakeBleakScanner  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "bleak", fake_bleak)

    scanner = BleakScanner(timeout=1.0)
    devices = await scanner.scan()

    assert isinstance(scanner, PresenceScanner)
    assert devices == [
        Device(address="AA:BB:CC:DD:EE:FF", name="Phone", rssi=-55),
        Device(address="11:22:33:44:55:66", name=None, rssi=None),
    ]
