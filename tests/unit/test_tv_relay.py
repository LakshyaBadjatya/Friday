from __future__ import annotations

from friday.tv.models import TVAction, TVActionType
from friday.tv.relay import TVRelay


def _action(app: str = "youtube") -> TVAction:
    return TVAction(type=TVActionType.OPEN_APP, app=app, speak=f"Opening {app}.")


def test_pair_returns_unique_ids() -> None:
    relay = TVRelay()
    a = relay.pair("Living Room")
    b = relay.pair("Bedroom")
    assert a and b and a != b
    assert set(relay.devices()) == {a, b}


def test_enqueue_then_drain_is_fifo() -> None:
    relay = TVRelay()
    dev = relay.pair("TV")
    assert relay.enqueue(dev, _action("youtube")) is True
    assert relay.enqueue(dev, _action("netflix")) is True
    drained = relay.drain(dev)
    assert [a.app for a in drained] == ["youtube", "netflix"]
    assert relay.drain(dev) == []  # queue emptied by the first drain


def test_enqueue_unknown_device_returns_false() -> None:
    relay = TVRelay()
    assert relay.enqueue("nope", _action()) is False


def test_default_device_is_sole_paired_else_none() -> None:
    relay = TVRelay()
    assert relay.default_device() is None
    dev = relay.pair("TV")
    assert relay.default_device() == dev
    relay.pair("Second TV")
    assert relay.default_device() is None  # ambiguous → caller must target by id


def test_queue_is_bounded_drops_oldest() -> None:
    relay = TVRelay(max_queue=2)
    dev = relay.pair("TV")
    for app in ("a", "b", "c"):
        relay.enqueue(dev, _action(app))
    assert [a.app for a in relay.drain(dev)] == ["b", "c"]
