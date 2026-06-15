"""Presence detection (Tier 3): which known devices are currently nearby via BLE.

This package owns FRIDAY's presence feature — a BLE proximity scan mapped against
a ``MAC -> friendly name`` map of known devices, surfaced as a present/absent
split with arrival/departure transitions across scans. It is off by default
behind ``FRIDAY_ENABLE_PRESENCE``, so the offline build runs no scan and exposes
no ``/presence`` route (-> 404).

The Bluetooth backend (``bleak``) is intentionally **NOT** in the uv lock and is
lazy-imported by :class:`~friday.presence.scanner.BleakScanner`, so importing this
package never requires a Bluetooth stack; the default offline path uses the
:class:`~friday.presence.scanner.FakePresenceScanner`.

The public surface is the typed :class:`~friday.presence.scanner.Device` model,
the :class:`~friday.presence.scanner.PresenceScanner` protocol and its fake/real
implementations, the :class:`~friday.presence.service.PresenceService` (plus its
:class:`~friday.presence.service.PresenceUpdate` result), and the flagged
``/presence`` :data:`router` (re-exported for the integration agent to wire).
"""

from __future__ import annotations

from friday.api.routes_presence import router
from friday.presence.scanner import (
    BleakScanner,
    Device,
    FakePresenceScanner,
    PresenceScanner,
)
from friday.presence.service import PresenceService, PresenceUpdate

__all__ = [
    "BleakScanner",
    "Device",
    "FakePresenceScanner",
    "PresenceScanner",
    "PresenceService",
    "PresenceUpdate",
    "router",
]
