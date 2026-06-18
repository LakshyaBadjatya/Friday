"""``POST /siri/ask`` — the Siri Shortcuts front door into the core loop.

Flagged behind ``FRIDAY_ENABLE_SIRI`` (default off -> ``404``, mirroring ``/studio``
and ``/maps``); the feature simply does not exist until turned on. When on it sits
behind the gateway :class:`~friday.api.middleware.AuthMiddleware` (require a bearer
key) and the rate limiter, so a public/tunnelled deployment is gated by a token.

It runs the spoken query through the **same** :class:`~friday.core.orchestrator.Orchestrator`
that backs ``/chat`` (full power — nothing is blocked here) and returns a short,
markdown-stripped string for Siri's "Speak Text" action. Pass ``?format=json`` to
get ``{"speak", "text", "mode"}`` instead.

Input is read leniently so the Shortcut can send whichever is easiest: a ``?q=``
query param, a JSON body (``{"q"|"text"|"query": ...}``), a urlencoded form, or a
raw ``text/plain`` body. A domain :class:`~friday.errors.FridayError` is spoken as a
graceful apology (HTTP 200) so Siri never reads a stack trace; auth/rate-limit
rejections keep their honest 401/429 from the middleware.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any
from urllib.parse import parse_qs

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, PlainTextResponse

from friday.circle.intents import handle_intent, parse_intent
from friday.core.state import GraphState
from friday.errors import FridayError
from friday.logging import get_logger
from friday.siri.speech import for_speech

logger = get_logger("friday.api.routes_siri")

router = APIRouter()

#: Default session id so successive "Hey Siri, ask Friday…" turns share memory.
_DEFAULT_SESSION = "siri"
#: Spoken when the brain returns nothing / errors — Siri should never read silence.
_FALLBACK_SPEECH = "Sorry, I didn't catch that. Could you try again?"
#: Upper bound on the accepted query (parity with ``/chat``'s 8000-char input).
_MAX_QUERY = 8000

#: Literal Shortcuts placeholder labels. If the request body contains one of these
#: verbatim, the shortcut is mis-wired — it's sending the *name* of a variable
#: instead of its value (the user's actual words). We detect that exact case and
#: speak a fix-it hint rather than letting the brain ask to clarify every turn.
_PLACEHOLDER_LABELS = frozenset(
    {
        "dictated text",
        "dictate text",
        "spoken text",
        "spoken input",
        "provided input",
        "shortcut input",
        "ask each time",
        "text",
        "input",
    }
)
#: Spoken when a placeholder label is detected — actionable, not cryptic.
_PLACEHOLDER_HINT = (
    "It looks like your shortcut is sending a placeholder instead of your words. "
    "Open the Friday shortcut, and in the Get Contents of URL step, delete the typed "
    "text in the request body and insert the blue Dictated Text variable instead."
)


def _siri_enabled(request: Request) -> bool:
    """Whether the Siri surface is enabled, read off startup settings on app state."""
    settings = getattr(request.app.state, "settings", None)
    return bool(getattr(settings, "enable_siri", False))


def _disabled() -> JSONResponse:
    """The canonical ``siri disabled`` 404 response."""
    return JSONResponse(status_code=404, content={"detail": "siri disabled"})


async def _read_query(request: Request) -> str | None:
    """Pull the spoken query from ``?q=``, a JSON body, a form, or a raw body.

    Returns the trimmed query, or ``None`` when nothing usable was sent. Parsing is
    done on the raw bytes (rather than ``request.json()``/``request.form()``) so the
    various content types a Shortcut might send are handled uniformly.
    """
    q = request.query_params.get("q")
    if q and q.strip():
        return q.strip()

    raw = await request.body()
    if not raw:
        return None
    text = raw.decode("utf-8", errors="replace")
    ctype = request.headers.get("content-type", "")

    if "application/json" in ctype:
        try:
            data = json.loads(text)
        except ValueError:
            return None
        if isinstance(data, dict):
            for key in ("q", "text", "query"):
                value = data.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        return None

    if "application/x-www-form-urlencoded" in ctype:
        parsed = parse_qs(text)
        for key in ("q", "text", "query"):
            if parsed.get(key) and parsed[key][0].strip():
                return parsed[key][0].strip()
        return None

    # Fall back to a raw text/plain body.
    return text.strip() or None


def _respond(speech: str, *, raw: str, mode: str | None, want_json: bool) -> Any:
    """Render the spoken reply as plain text (default) or a JSON envelope."""
    if want_json:
        return JSONResponse(
            status_code=200,
            content={"speak": speech, "text": raw, "mode": mode},
        )
    return PlainTextResponse(content=speech, media_type="text/plain; charset=utf-8")


def _caller_uid(request: Request) -> str | None:
    """Resolve the bearer token to a circle uid via ``app.state.siri_identities``.

    The map (token -> uid) is wired at startup; absent it, the caller is anonymous
    and circle intents are skipped (the request falls through to the assistant).
    """
    identities = getattr(request.app.state, "siri_identities", None)
    if not isinstance(identities, dict):
        return None
    auth = request.headers.get("authorization", "")
    if not auth.lower().startswith("bearer "):
        return None
    uid = identities.get(auth[7:].strip())
    return uid if isinstance(uid, str) else None


def _try_circle(request: Request, query: str) -> str | None:
    """Handle a circle status intent if one is present and the caller is known.

    Returns the spoken reply, or ``None`` to fall through to the orchestrator
    (no circle wired, anonymous caller, non-circle phrasing, or an unknown name).
    """
    circle = getattr(request.app.state, "circle", None)
    status = getattr(request.app.state, "circle_status", None)
    if circle is None or status is None:
        return None
    caller_uid = _caller_uid(request)
    if caller_uid is None:
        return None
    intent = parse_intent(query)
    if intent is None:
        return None
    return handle_intent(circle, status, caller_uid, intent, now=datetime.now(UTC))


@router.post("/siri/ask", response_model=None)
async def siri_ask(request: Request) -> Any:
    """Answer one spoken query through the core loop; 404 when the flag is off."""
    if not _siri_enabled(request):
        return _disabled()

    query = await _read_query(request)
    if query is None:
        return JSONResponse(
            status_code=400, content={"detail": "missing query 'q'"}
        )
    query = query[:_MAX_QUERY]
    want_json = request.query_params.get("format", "").lower() == "json"
    session_id = request.query_params.get("session") or _DEFAULT_SESSION

    # Mis-wired shortcut guard: the body is a literal variable label (e.g. "Dictated
    # Text"), not the spoken words. Speak an actionable fix instead of clarifying.
    if query.lower() in _PLACEHOLDER_LABELS:
        return _respond(
            _PLACEHOLDER_HINT, raw=_PLACEHOLDER_HINT, mode="hint", want_json=want_json
        )

    # Circle status intents ("what's X doing", "set my status…") win when the
    # caller is known and the phrasing matches; otherwise fall through below.
    circle_reply = _try_circle(request, query)
    if circle_reply is not None:
        return _respond(
            for_speech(circle_reply), raw=circle_reply, mode="circle", want_json=want_json
        )

    orchestrator = getattr(request.app.state, "orchestrator", None)
    if orchestrator is None or not hasattr(orchestrator, "handle"):
        logger.error("siri ask: orchestrator missing on app.state")
        return _respond(_FALLBACK_SPEECH, raw="", mode=None, want_json=want_json)

    state = GraphState(session_id=session_id, user_input=query)
    try:
        result = await orchestrator.handle(state)
    except FridayError as exc:
        logger.warning(
            "siri ask raised FridayError",
            extra={"error_type": type(exc).__name__},
        )
        return _respond(_FALLBACK_SPEECH, raw="", mode=None, want_json=want_json)

    raw_text = getattr(result, "response", None) or ""
    speech = for_speech(raw_text) or _FALLBACK_SPEECH
    mode = getattr(getattr(result, "mode", None), "value", None)
    return _respond(speech, raw=raw_text, mode=mode, want_json=want_json)
