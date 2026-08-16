"""``WS /link/connect`` — where a machine dials in and stays.

The relay opens this socket from the owner's side and holds it. Nothing is
listening on his laptop; the connection is outbound, which is the only shape
that works from behind a home router and the only one that does not add an
attack surface to a personal machine.

Authentication is a shared token, checked before the socket is accepted. It is
compared with :func:`secrets.compare_digest` because a token check that returns
early on the first wrong byte tells an attacker how much of the token was right,
and this is the one door in the system that leads somewhere real.

The endpoint is deliberately thin. It authenticates, registers the machine,
pumps frames between the socket and the registry, and deregisters on the way
out. Everything about *what may be asked* lives on the relay, on the machine
itself, where the owner can read it — not here, where a model's output could
reach it.
"""

from __future__ import annotations

import json
import secrets
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from friday.link.protocol import Hello, Result
from friday.link.registry import Link, registry_of
from friday.logging import get_logger

logger = get_logger("friday.api.link")

router = APIRouter(tags=["link"])


def _expected_token(websocket: WebSocket) -> str:
    settings = getattr(websocket.app.state, "settings", None)
    secret = getattr(settings, "link_token", None)
    if secret is None:
        return ""
    getter = getattr(secret, "get_secret_value", None)
    return str(getter() if callable(getter) else secret or "")


@router.websocket("/link/connect")
async def link_connect(websocket: WebSocket) -> None:
    """Accept a relay, then carry jobs to it until the socket dies."""
    expected = _expected_token(websocket)
    offered = websocket.query_params.get("token", "")
    if not expected:
        # Refused rather than left open. An unset token would otherwise mean
        # anyone who finds this URL gets a socket that runs jobs on a laptop.
        logger.warning("link: refused a relay — FRIDAY_LINK_TOKEN is not set")
        await websocket.close(code=1008)
        return
    if not secrets.compare_digest(offered, expected):
        logger.warning("link: refused a relay with a bad token")
        await websocket.close(code=1008)
        return

    await websocket.accept()
    link: Link | None = None
    machine = ""
    try:
        # The first frame identifies the machine. Anything else is a relay that
        # does not speak this protocol, and it is dropped rather than guessed at.
        raw = await websocket.receive_text()
        hello = Hello.model_validate_json(raw)
        machine = hello.machine.strip()[:40] or "unknown"
        link = Link(machine, websocket, hello.commands, hello.platform)
        registry_of(websocket.app).attach(link)

        while True:
            frame = await websocket.receive_text()
            _deliver(link, frame)
    except WebSocketDisconnect:
        pass
    except Exception as exc:  # noqa: BLE001 - a bad relay is not a server error
        logger.warning("link: relay %s dropped: %s", machine or "?", exc)
    finally:
        if link is not None:
            registry_of(websocket.app).detach(machine, link)


def _deliver(link: Link, frame: str) -> None:
    """Route one inbound frame to whoever is waiting for it.

    Heartbeats arrive on the same socket and are simply the absence of a job id;
    they keep the connection warm through NAT timeouts and are otherwise of no
    interest.
    """
    try:
        payload: Any = json.loads(frame)
    except ValueError:
        return
    if not isinstance(payload, dict) or not payload.get("id"):
        return
    try:
        link.deliver(Result.model_validate(payload))
    except Exception:  # noqa: BLE001 - a malformed result is not a crash
        logger.warning("link: %s sent a result that did not parse", link.machine)
