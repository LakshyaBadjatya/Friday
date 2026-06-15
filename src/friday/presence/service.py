"""Presence tracking: turn a stream of BLE scans into arrivals/departures.

:class:`PresenceService` wraps a :class:`~friday.presence.scanner.PresenceScanner`
and a ``known`` map of MAC -> friendly name. Each :meth:`update` runs one scan,
maps the seen MAC addresses to the known names, and reports:

* ``present`` — known names currently in range,
* ``absent``  — known names not in range,
* ``arrived`` — known names that became present *since the previous update*,
* ``departed``— known names that became absent *since the previous update*.

The service remembers the previous present-set between updates, so transitions
are computed by set difference. All name lists are sorted for deterministic
output. MAC matching is case-insensitive. ``clock`` is injected (default UTC
:func:`datetime.now`) so the ``ts`` on each :class:`PresenceUpdate` is pinnable in
tests; the service touches no wall clock of its own.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import UTC, datetime

from pydantic import BaseModel

from friday.presence.scanner import PresenceScanner


class PresenceUpdate(BaseModel):
    """The result of one :meth:`PresenceService.update` tick.

    ``present`` / ``absent`` are the full current split of *known* device names;
    ``arrived`` / ``departed`` are the transitions since the previous update
    (empty on a steady state). All four are name-sorted. ``ts`` is the ISO-8601
    instant the update was computed (from the injected clock).
    """

    present: list[str]
    absent: list[str]
    arrived: list[str]
    departed: list[str]
    ts: str


def _utcnow() -> datetime:
    """Default clock: the current instant in UTC."""
    return datetime.now(UTC)


class PresenceService:
    """Track known BLE devices across scans, reporting arrivals/departures.

    Only devices in ``known`` are tracked; an unknown advertisement on the air is
    ignored. The previous present-set is remembered between :meth:`update` calls
    so the first update never reports a departure and reports an arrival for every
    known device already in range.
    """

    def __init__(
        self,
        scanner: PresenceScanner,
        known: Mapping[str, str],
        *,
        clock: Callable[[], datetime] = _utcnow,
    ) -> None:
        """Assemble the service.

        Args:
            scanner: The BLE scan source.
            known: A ``MAC -> friendly name`` map of the devices to track. MAC
                keys are matched case-insensitively against scanned addresses.
            clock: Injectable clock returning the current instant; defaults to
                UTC :func:`datetime.now`.
        """
        self._scanner = scanner
        # Normalise the known map to lower-case MACs for case-insensitive match.
        self._known: dict[str, str] = {
            mac.lower(): name for mac, name in known.items()
        }
        self._clock = clock
        # Names known to be present as of the previous update (None == no prior).
        self._previous: set[str] | None = None

    async def update(self) -> PresenceUpdate:
        """Run one scan and report the present split plus the transitions."""
        devices = await self._scanner.scan()
        seen_macs = {dev.address.lower() for dev in devices}

        present_names = {
            name for mac, name in self._known.items() if mac in seen_macs
        }
        all_names = set(self._known.values())
        absent_names = all_names - present_names

        previous = self._previous if self._previous is not None else set()
        arrived = present_names - previous
        departed = previous - present_names
        self._previous = present_names

        return PresenceUpdate(
            present=sorted(present_names),
            absent=sorted(absent_names),
            arrived=sorted(arrived),
            departed=sorted(departed),
            ts=self._clock().isoformat(),
        )
