"""``/ws/voice`` — a minimal websocket scaffold for streaming + barge-in (Phase 3).

This is intentionally small: the full streaming UX (live capture, partial STT,
duplex barge-in) lands in a later tier. For now the endpoint:

* refuses the connection with a ``1008`` policy-violation close when voice is
  disabled (``FRIDAY_ENABLE_VOICE`` off), so the socket simply isn't usable until
  the flag is set;
* otherwise accepts the connection and sends a single ``{"type": "ready"}`` frame
  announcing the barge-in-capable channel, then echoes any control frames a
  client sends (e.g. a ``{"type": "bargein"}`` signal) until the client
  disconnects.

No heavy voice library is imported here; this module only wires the transport.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from friday.logging import get_logger

logger = get_logger("friday.api.ws")

router = APIRouter()

# Close code for "policy violation" — used to refuse the socket when voice is off.
_POLICY_VIOLATION = 1008


def _voice_enabled(websocket: WebSocket) -> bool:
    """Whether voice is enabled, read off the startup settings on ``app.state``."""
    settings = getattr(websocket.app.state, "settings", None)
    return bool(getattr(settings, "enable_voice", False))


def _wakeword_enabled(websocket: WebSocket) -> bool:
    """Whether the wake word is enabled, read off the startup settings."""
    settings = getattr(websocket.app.state, "settings", None)
    return bool(getattr(settings, "enable_wakeword", False))


@router.websocket("/ws/voice")
async def ws_voice(websocket: WebSocket) -> None:
    """Accept a voice websocket, announce readiness, and echo control frames.

    Guarded by ``FRIDAY_ENABLE_VOICE``: when voice is disabled the connection is
    accepted only to be immediately closed with a policy-violation code, so a
    client gets a clean, explicit refusal rather than a silent hang.
    """
    if not _voice_enabled(websocket):
        await websocket.accept()
        await websocket.close(code=_POLICY_VIOLATION, reason="voice disabled")
        return

    await websocket.accept()
    await websocket.send_json({"type": "ready", "bargein": True})

    try:
        while True:
            message: dict[str, Any] = await websocket.receive_json()
            # Echo control frames back (barge-in signaling scaffold); the full
            # duplex streaming UX is a later tier.
            await websocket.send_json({"type": "echo", "received": message})
    except WebSocketDisconnect:  # pragma: no cover - exercised via client close
        logger.info("voice websocket disconnected")
    finally:
        if websocket.application_state != WebSocketState.DISCONNECTED:
            await websocket.close()


@router.websocket("/ws/wake")
async def ws_wake(websocket: WebSocket) -> None:
    """Wake/summon channel: the client streams transcripts, the server replies with
    :class:`~friday.voice.wake_service.WakeEvent`s.

    The HUD (or a server STT runner) sends ``{"transcript": "..."}`` frames; the
    shared :class:`~friday.voice.wake_service.WakeService` on ``app.state`` parses
    each and, on a "Hey FRIDAY" / "FRIDAY summon &lt;op&gt;" match, the server sends
    back a wake/summon event ``{"type", "operator", "greeting"}`` so the HUD reveals
    the cockpit and speaks the greeting in that operator's voice. Non-commands get
    no reply. Guarded by ``FRIDAY_ENABLE_WAKEWORD``: when off, the socket is refused
    with a policy-violation close.
    """
    if not _wakeword_enabled(websocket):
        await websocket.accept()
        await websocket.close(code=_POLICY_VIOLATION, reason="wake word disabled")
        return

    service = getattr(websocket.app.state, "wake_service", None)
    await websocket.accept()
    await websocket.send_json({"type": "ready"})

    try:
        while True:
            message: dict[str, Any] = await websocket.receive_json()
            transcript = message.get("transcript", "") if isinstance(message, dict) else ""
            event = service.handle_transcript(transcript) if service is not None else None
            if event is not None:
                await websocket.send_json(event.model_dump())
    except WebSocketDisconnect:  # pragma: no cover - exercised via client close
        logger.info("wake websocket disconnected")
    finally:
        if websocket.application_state != WebSocketState.DISCONNECTED:
            await websocket.close()
