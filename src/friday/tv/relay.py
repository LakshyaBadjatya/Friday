"""In-memory phone→TV command relay: pair a TV, enqueue actions, drain/await them.

State is per-process and intentionally ephemeral (a personal, single-user surface).
Each device keeps a bounded FIFO queue; an :class:`asyncio.Event` lets the WebSocket
await the next command without polling.
"""

from __future__ import annotations

import asyncio
import secrets
from collections import deque
from dataclasses import dataclass, field

from friday.tv.models import TVAction


@dataclass
class _Device:
    device_id: str
    name: str
    queue: deque[TVAction]
    event: asyncio.Event = field(default_factory=asyncio.Event)


class TVRelay:
    """Registry of paired TVs and their pending command queues."""

    def __init__(self, max_queue: int = 32) -> None:
        self._devices: dict[str, _Device] = {}
        self._max = max_queue

    def pair(self, name: str = "") -> str:
        """Register a new TV and return its opaque device id."""
        device_id = secrets.token_urlsafe(8)
        self._devices[device_id] = _Device(
            device_id=device_id, name=name, queue=deque(maxlen=self._max)
        )
        return device_id

    def devices(self) -> list[str]:
        return list(self._devices)

    def default_device(self) -> str | None:
        """The sole paired device, or ``None`` when zero or many are paired."""
        return next(iter(self._devices)) if len(self._devices) == 1 else None

    def enqueue(self, device_id: str, action: TVAction) -> bool:
        """Queue ``action`` for ``device_id``; ``False`` if the device is unknown."""
        device = self._devices.get(device_id)
        if device is None:
            return False
        device.queue.append(action)
        device.event.set()
        return True

    def drain(self, device_id: str) -> list[TVAction]:
        """Return and clear all pending actions for ``device_id`` (poll fallback)."""
        device = self._devices.get(device_id)
        if device is None:
            return []
        items = list(device.queue)
        device.queue.clear()
        return items

    async def wait(self, device_id: str) -> TVAction:
        """Await and pop the next action for ``device_id`` (WebSocket path).

        Raises ``KeyError`` if the device is not registered.
        """
        device = self._devices[device_id]
        while not device.queue:
            device.event.clear()
            await device.event.wait()
        return device.queue.popleft()
