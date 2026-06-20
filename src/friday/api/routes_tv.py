"""``/tv`` — the Android TV surface (parsing + phone→TV relay). Flag: enable_tv.

Mirrors ``/siri/ask``: off by default (routes 404), behind ``AuthMiddleware`` + the
rate limiter when on. ``/tv/ask`` parses spoken text into a :class:`TVAction`; when
the text is not a command it falls back to the same orchestrator as ``/chat`` and
returns ``action: null``. The relay routes "…on the TV" phone commands to a paired
TV, which drains them over ``/tv/poll`` or the ``/tv/stream`` WebSocket.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from friday.api.routes_siri import _read_query
from friday.core.state import GraphState
from friday.errors import FridayError
from friday.logging import get_logger
from friday.siri.speech import for_speech
from friday.tv.intents import parse_tv_command
from friday.tv.relay import TVRelay

logger = get_logger("friday.api.routes_tv")

router = APIRouter()

_FALLBACK_SPEECH = "Sorry, I didn't catch that. Could you try again?"
_MAX_QUERY = 8000


def _enabled(request: Request) -> bool:
    settings = getattr(request.app.state, "settings", None)
    return bool(getattr(settings, "enable_tv", False))


def _disabled() -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": "tv disabled"})


def _relay(request: Request) -> TVRelay | None:
    return getattr(request.app.state, "tv_relay", None)


async def _spoken_answer(request: Request, query: str) -> str:
    """Run a non-command query through the orchestrator; never raise to the client."""
    orchestrator = getattr(request.app.state, "orchestrator", None)
    if orchestrator is None or not hasattr(orchestrator, "handle"):
        return _FALLBACK_SPEECH
    state = GraphState(session_id="tv", user_input=query)
    try:
        result = await orchestrator.handle(state)
    except FridayError:
        logger.warning("tv ask: orchestrator raised FridayError; speaking a fallback")
        return _FALLBACK_SPEECH
    except Exception:  # noqa: BLE001 - TV must never read a raw 500 to the user
        logger.exception("tv ask: unexpected error; speaking a fallback")
        return _FALLBACK_SPEECH
    return for_speech(getattr(result, "response", "") or "") or _FALLBACK_SPEECH


@router.post("/tv/ask", response_model=None)
async def tv_ask(request: Request) -> Any:
    """Parse one spoken query into ``{speak, text, mode, action}``."""
    if not _enabled(request):
        return _disabled()
    query = await _read_query(request)
    if query is None:
        return JSONResponse(status_code=400, content={"detail": "missing query 'q'"})
    query = query[:_MAX_QUERY]

    action = parse_tv_command(query)
    if action is not None:
        return JSONResponse(
            status_code=200,
            content={
                "speak": action.speak,
                "text": "",
                "mode": "tv",
                "action": action.model_dump(),
            },
        )

    speech = await _spoken_answer(request, query)
    return JSONResponse(
        status_code=200,
        content={"speak": speech, "text": speech, "mode": "tv", "action": None},
    )


@router.post("/tv/pair", response_model=None)
async def tv_pair(request: Request) -> Any:
    """Register a TV and return its opaque device id."""
    if not _enabled(request):
        return _disabled()
    relay = _relay(request)
    if relay is None:
        return JSONResponse(status_code=503, content={"detail": "relay unavailable"})
    try:
        body = await request.json()
    except ValueError:
        body = {}
    name = str(body.get("name", "")) if isinstance(body, dict) else ""
    device_id = relay.pair(name)
    return JSONResponse(status_code=200, content={"device_id": device_id, "name": name})


@router.post("/tv/command", response_model=None)
async def tv_command(request: Request) -> Any:
    """Parse ``text`` and enqueue the resulting action for a paired ``device_id``."""
    if not _enabled(request):
        return _disabled()
    relay = _relay(request)
    if relay is None:
        return JSONResponse(status_code=503, content={"detail": "relay unavailable"})
    try:
        body = await request.json()
    except ValueError:
        body = {}
    device_id = str(body.get("device_id", "")) if isinstance(body, dict) else ""
    text = str(body.get("text", "")) if isinstance(body, dict) else ""
    action = parse_tv_command(text)
    if action is None:
        return JSONResponse(
            status_code=200, content={"queued": False, "reason": "not a command"}
        )
    queued = relay.enqueue(device_id, action)
    return JSONResponse(
        status_code=200, content={"queued": queued, "action": action.model_dump()}
    )


@router.get("/tv/poll", response_model=None)
async def tv_poll(request: Request) -> Any:
    """Drain pending actions for ``device_id`` (WebSocket-free fallback)."""
    if not _enabled(request):
        return _disabled()
    relay = _relay(request)
    device_id = request.query_params.get("device_id", "")
    actions = [a.model_dump() for a in relay.drain(device_id)] if relay else []
    return JSONResponse(status_code=200, content={"actions": actions})


@router.websocket("/tv/stream")
async def tv_stream(websocket: WebSocket) -> None:
    """Hold a connection per TV and push each enqueued action as JSON."""
    settings = getattr(websocket.app.state, "settings", None)
    relay = getattr(websocket.app.state, "tv_relay", None)
    device_id = websocket.query_params.get("device_id", "")
    if not bool(getattr(settings, "enable_tv", False)) or relay is None or not device_id:
        await websocket.close(code=1008)
        return
    await websocket.accept()
    try:
        while True:
            action = await relay.wait(device_id)
            await websocket.send_json(action.model_dump())
    except (WebSocketDisconnect, KeyError):
        return
