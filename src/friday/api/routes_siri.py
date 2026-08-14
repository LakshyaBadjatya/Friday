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
import secrets
from datetime import UTC, datetime
from typing import Any
from urllib.parse import parse_qs

import anyio
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, PlainTextResponse

from friday.circle.intents import handle_intent, parse_intent
from friday.core.state import GraphState
from friday.errors import FridayError
from friday.logging import get_logger
from friday.siri import context as siri_context
from friday.siri.arithmetic import arithmetic_reply
from friday.siri.speech import for_speech

logger = get_logger("friday.api.routes_siri")

router = APIRouter()

#: Default session id so successive "Hey Siri, ask Friday…" turns share memory.
_DEFAULT_SESSION = "siri"
#: Spoken when the brain returns nothing / errors — Siri should never read silence.
_FALLBACK_SPEECH = "Sorry, I didn't catch that. Could you try again?"
#: Spoken when the turn blows its wall-clock budget. Siri abandons a slow request
#: and reads nothing at all, so a fast honest line beats a perfect late one: the
#: user hears *something* while the voice session is still open.
_TIMEOUT_SPEECH = (
    "That one's taking me longer than Siri will wait, Boss. Ask me again in a moment."
)
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

#: "Who made you?" — answered instantly (no LLM) with a fixed name, varied wording.
_CREATOR_TRIGGERS = (
    "who made you",
    "who created you",
    "who built you",
    "who designed you",
    "who developed you",
    "who programmed you",
    "who coded you",
    "who is your maker",
    "who's your maker",
    "who is your creator",
    "who's your creator",
    "who is your master",
    "who's your master",
    "who is your owner",
    "who do you work for",
    "who do you belong to",
)
_CREATOR_LINES = (
    "My master is Lakshya Badjatya — he built me.",
    "I was created by Lakshya Badjatya, Boss.",
    "That'd be Lakshya Badjatya — my maker and master.",
    "Lakshya Badjatya made me. I answer to him.",
    "I'm Lakshya Badjatya's creation.",
    "Crafted by Lakshya Badjatya, my one and only master.",
    "Lakshya Badjatya is the mind behind me.",
)


def _creator_reply(query: str) -> str | None:
    """A fast, varied 'who made you' answer (same name, different wording)."""
    low = query.lower()
    if any(trigger in low for trigger in _CREATOR_TRIGGERS):
        return secrets.choice(_CREATOR_LINES)
    return None


#: Formula / theory / "explain" questions get a teach-me-simply instruction so the
#: model defines symbols, gives intuition, and flags uncertainty instead of guessing.
_TEACH_TRIGGERS = (
    "formula",
    "equation",
    "theorem",
    "theory",
    "law of",
    "principle",
    "derive",
    "derivation",
    "prove",
    "explain",
    "definition",
    "define ",
    "concept",
    "how does",
    "why does",
    "how do you calculate",
)
_TEACH_INSTR = (
    " (Answer accurately and simply, in plain spoken words a beginner follows. If "
    "there's a formula, state it naming each symbol in words like 'E equals m c "
    "squared', say what each symbol means, and give a one-line intuition. Be precise; "
    "if you're not certain, say so rather than guess.)"
)


def _augment_teaching(query: str) -> str:
    """Append the explain-simply-and-accurately instruction to teaching questions."""
    low = query.lower()
    if any(trigger in low for trigger in _TEACH_TRIGGERS):
        return query + _TEACH_INSTR
    return query


#: Appended to the persona for the fast voice path so replies are short and speakable.
_VOICE_RULES = (
    "\n\nYou are answering by VOICE through Siri. Reply in 1-4 short sentences of "
    "plain spoken text — no markdown, no bullet lists, no headings, no code blocks. "
    "Address the user as 'Boss'. Be accurate and direct; if you're unsure, say so "
    "briefly rather than guessing. For a formula or concept, state it, then give a "
    "one-line plain-English explanation."
)


async def _fast_answer(
    request: Request, query: str, history: list[Any], deadline: float
) -> str | None:
    """A single persona'd LLM call — the low-latency voice path.

    Skips the full orchestrator graph (routing, memory, optional critic re-pass,
    confidence scoring), which is several steps and sometimes a second model call.
    Reuses the live FRIDAY persona for voice consistency, falling back to a minimal
    one. ``history`` is the recalled context window, replayed between the persona
    and this turn so follow-ups ("and the last topic?") resolve against what was
    actually said rather than starting cold.

    Returns ``None`` on any provider failure so the caller drops to the
    orchestrator, but raises :class:`TimeoutError` when ``deadline`` expires —
    those are different situations: a failure is worth retrying through the slower
    path, whereas a timeout means there is no time left to retry anything.
    """
    llm = getattr(request.app.state, "llm", None)
    if llm is None:
        return None
    from friday.providers.llm import Message  # noqa: PLC0415

    orchestrator = getattr(request.app.state, "orchestrator", None)
    persona = "You are FRIDAY, a sharp, warm personal AI assistant."
    getter = getattr(orchestrator, "_persona_text", None)
    if callable(getter):
        try:
            persona = getter() or persona
        except Exception:  # noqa: BLE001 - fall back to the minimal persona
            pass
    try:
        with anyio.fail_after(deadline):
            resp = await llm.complete(
                [
                    Message(role="system", content=persona + _VOICE_RULES),
                    *history,
                    Message(role="user", content=_augment_teaching(query)),
                ]
            )
    except TimeoutError:
        logger.warning("siri fast path exceeded its %.1fs budget", deadline)
        raise
    except Exception:  # noqa: BLE001 - drop to the orchestrator
        return None
    return (getattr(resp, "text", "") or "").strip() or None


def _siri_enabled(request: Request) -> bool:
    """Whether the Siri surface is enabled, read off startup settings on app state."""
    settings = getattr(request.app.state, "settings", None)
    return bool(getattr(settings, "enable_siri", False))


def _disabled() -> JSONResponse:
    """The canonical ``siri disabled`` 404 response."""
    return JSONResponse(status_code=404, content={"detail": "siri disabled"})


def _memory(request: Request) -> Any:
    """The conversation buffer for the voice path.

    Deliberately the *same* :class:`~friday.memory.short_term.ShortTermMemory` the
    orchestrator writes to (wired at ``app.state.short_term``), so a spoken turn
    and a ``/chat`` turn on one ``session_id`` build a single shared thread rather
    than two blind ones. ``None`` when unwired — recall then degrades to no
    context instead of failing the request.
    """
    return getattr(request.app.state, "short_term", None)


def _deadline(request: Request) -> float:
    """Wall-clock seconds allowed for one spoken answer (see ``siri_timeout_seconds``)."""
    settings = getattr(request.app.state, "settings", None)
    try:
        deadline = float(getattr(settings, "siri_timeout_seconds", 12.0) or 12.0)
    except (TypeError, ValueError):
        return 12.0
    return deadline if deadline > 0 else 12.0


def _context_limit(request: Request) -> int:
    """How many past messages to replay (``0`` disables recall entirely)."""
    settings = getattr(request.app.state, "settings", None)
    try:
        limit = int(
            getattr(settings, "siri_context_messages", siri_context.DEFAULT_CONTEXT_MESSAGES)
        )
    except (TypeError, ValueError):
        return siri_context.DEFAULT_CONTEXT_MESSAGES
    return max(0, limit)


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


async def _read_coords(request: Request) -> tuple[str | None, str | None]:
    """GPS for "… near me": prefer ``?lat=&lon=``, else a JSON body's lat/lon.

    The Shortcut can pass the device location either in the URL or — far more
    reliably — as ``lat``/``lon`` fields next to ``q`` in the JSON body. Reading
    both means the same shortcut works whichever way the user wired it. The body
    is re-read here (Starlette caches it after ``_read_query``), so this is cheap.
    """
    lat = request.query_params.get("lat")
    lon = request.query_params.get("lon")
    if lat and lon:
        return lat, lon
    raw = await request.body()
    if not raw or "application/json" not in request.headers.get("content-type", ""):
        return None, None
    try:
        data = json.loads(raw.decode("utf-8", errors="replace"))
    except ValueError:
        return None, None
    if not isinstance(data, dict):
        return None, None
    blat = data.get("lat") or data.get("latitude")
    blon = data.get("lon") or data.get("lng") or data.get("longitude")
    if blat is None or blon is None:
        return None, None
    return str(blat), str(blon)


def _respond(
    speech: str,
    *,
    raw: str,
    mode: str | None,
    want_json: bool,
    action: dict[str, Any] | None = None,
) -> Any:
    """Render the spoken reply as plain text (default) or a JSON envelope."""
    if want_json:
        return JSONResponse(
            status_code=200,
            content={"speak": speech, "text": raw, "mode": mode, "action": action},
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


def _try_firestore_circle(request: Request, query: str) -> str | None:
    """Act on the app's real Firestore as the caller (presence, reminders, nudges…).

    Tried first when the bearer looks like a real Firebase credential (an ID-token
    JWT or a long refresh token); short dev tokens are skipped so offline tests never
    touch the network. ANY failure returns ``None`` so the request falls through to
    the in-memory circle / orchestrator — the live endpoint can never break.
    """
    auth = request.headers.get("authorization", "")
    if not auth.lower().startswith("bearer "):
        return None
    token = auth[7:].strip()
    if token.count(".") != 2 and len(token) < 100:
        return None  # not a plausible Firebase ID/refresh token (skip dev tokens)
    try:
        from friday.circle.siri_circle import handle as _handle  # noqa: PLC0415

        return _handle(token, query, datetime.now(UTC))
    except Exception:  # noqa: BLE001 - never break the live endpoint
        logger.warning("siri firestore-circle path failed", exc_info=False)
        return None


def _instagram_service(request: Request) -> Any:
    """Lazily build + cache the Instagram service on ``app.state.instagram``.

    Returns the cached service if present (so tests can inject a fake); else, when
    the feature is enabled and credentials exist, builds an ``InstagrapiClient`` +
    ``InstagramService`` from startup settings and caches it. Returns ``None`` when
    the flag is off or credentials are missing — the caller then falls through.
    instagrapi is NOT imported here: the client only imports it at login time.
    """
    cached = getattr(request.app.state, "instagram", None)
    if cached is not None:
        return cached
    settings = getattr(request.app.state, "settings", None)
    if settings is None or not getattr(settings, "enable_instagram_dms", False):
        return None
    username = getattr(settings, "instagram_username", "") or ""
    pw_secret = getattr(settings, "instagram_password", None)
    password = pw_secret.get_secret_value() if pw_secret is not None else ""
    if not username or not password:
        return None
    from friday.instagram.client import InstagrapiClient  # noqa: PLC0415
    from friday.instagram.service import InstagramService  # noqa: PLC0415
    from friday.instagram.session import parse_session  # noqa: PLC0415

    sess_secret = getattr(settings, "instagram_session_json", None)
    raw_session = sess_secret.get_secret_value() if sess_secret is not None else None
    client = InstagrapiClient(username, password, parse_session(raw_session))
    limit = int(getattr(settings, "instagram_read_aloud_limit", 5) or 5)
    service = InstagramService(client, read_aloud_limit=limit)
    request.app.state.instagram = service
    return service


def _try_instagram(request: Request, query: str) -> str | None:
    """Handle an Instagram DM intent (count / read-aloud / reply); ``None`` to skip.

    Regex-classifies first (no network) inside ``siri_instagram.handle``; only an
    Instagram phrase constructs the client / hits the API. ANY error returns
    ``None`` so the request falls through and the live endpoint never breaks. A
    successful Instagram turn stamps ``app.state._ig_marker`` so a bare "read them
    aloud" right after counts as an Instagram follow-up.
    """
    service = _instagram_service(request)
    if service is None:
        return None
    from friday.instagram.siri_instagram import handle as instagram_handle  # noqa: PLC0415

    now = datetime.now(UTC)
    marker = getattr(request.app.state, "_ig_marker", None)
    try:
        reply = instagram_handle(service, query, now, marker=marker)
    except Exception:  # noqa: BLE001 - never break the live endpoint
        logger.warning("siri instagram path failed", exc_info=False)
        return None
    if reply is not None:
        request.app.state._ig_marker = now
    return reply


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
    memory = _memory(request)
    deadline = _deadline(request)

    def _reply(
        speech: str,
        *,
        raw: str,
        mode: str | None,
        action: dict[str, Any] | None = None,
        record: bool = True,
    ) -> Any:
        """Render the reply and, by default, record the turn in the context window.

        Every branch that actually answers goes through here, so the window holds
        the whole conversation — a distance answer, an Instagram read-out and an
        LLM reply are all equally "the last topic". Fallbacks pass
        ``record=False``: a turn that failed is not something to recall later.
        """
        if record:
            siri_context.remember(memory, session_id, query, raw or speech)
        return _respond(speech, raw=raw, mode=mode, want_json=want_json, action=action)

    # Mis-wired shortcut guard: the body is a literal variable label (e.g. "Dictated
    # Text"), not the spoken words. Speak an actionable fix instead of clarifying.
    if query.lower() in _PLACEHOLDER_LABELS:
        return _reply(
            _PLACEHOLDER_HINT, raw=_PLACEHOLDER_HINT, mode="hint", record=False
        )

    # The conversation so far, replayed into whichever path answers below.
    limit = _context_limit(request)
    history = (
        siri_context.recall(memory, session_id, max_messages=limit) if limit else []
    )

    # "What were we just talking about?" with an empty window: say so plainly
    # instead of letting the model invent a plausible-sounding earlier topic.
    if not history and siri_context.is_recall_question(query):
        return _reply(
            siri_context.NO_HISTORY_REPLY,
            raw=siri_context.NO_HISTORY_REPLY,
            mode="recall",
            record=False,
        )

    # "<command> on the TV" → parse and hand to the paired TV (the phone is the mic).
    relay = getattr(request.app.state, "tv_relay", None)
    if relay is not None:
        from friday.tv.intents import parse_tv_command, strip_tv_suffix  # noqa: PLC0415

        bare = strip_tv_suffix(query)
        if bare is not None:
            tv_action = parse_tv_command(bare)
            if tv_action is not None:
                device = relay.default_device()
                if device is None:
                    msg = "No TV is paired yet. Open Friday on the TV to pair it."
                    return _reply(msg, raw=msg, mode="tv", record=False)
                relay.enqueue(device, tv_action)
                return _reply(
                    tv_action.speak,
                    raw=tv_action.speak,
                    mode="tv",
                    action=tv_action.model_dump(),
                )

    # "Who made you?" — answered instantly (no model, no network).
    creator = _creator_reply(query)
    if creator is not None:
        return _reply(creator, raw=creator, mode="identity")

    # Pure arithmetic is *computed*, never predicted. An LLM asked for "17 percent
    # of 2,480" returns whatever tokens usually follow that phrasing, which is how
    # a confidently wrong number gets spoken aloud. Anything that is not
    # unambiguously a sum returns None here and still reaches the model.
    maths = arithmetic_reply(query)
    if maths is not None:
        return _reply(maths, raw=maths, mode="math")

    # Distance queries — geocoded + routed via OpenStreetMap (computed, not guessed).
    # Run in a worker thread: it is blocking urllib, and on the event loop it stalls
    # every other in-flight request (including this one's siblings) until it returns.
    from friday.maps.distance import distance_reply  # noqa: PLC0415

    dist = await anyio.to_thread.run_sync(distance_reply, query)
    if dist is not None:
        return _reply(dist, raw=dist, mode="distance")

    # "… near me" — use the exact GPS the shortcut sent (URL ?lat=&lon= or body).
    # The fast heuristic catches obvious phrasings; the AI classifier below catches
    # anything else. The richer share text (with a map link) goes in `text` so the
    # shortcut can push it to Telegram or the iOS share sheet.
    lat, lon = await _read_coords(request)
    if lat and lon:
        from friday.maps.nearby import classify_nearby, nearby_from_filter, nearby_reply

        try:
            flat, flon = float(lat), float(lon)
        except (TypeError, ValueError):
            flat = flon = None  # type: ignore[assignment]
        if flat is not None and flon is not None:
            near = await anyio.to_thread.run_sync(nearby_reply, query, flat, flon)
            if near is None:
                # AI auto-guess: let the model decide if this is a nearby-places
                # ask and infer the category, so no fixed phrase is required.
                llm = getattr(request.app.state, "llm", None)
                inferred = await classify_nearby(llm, query)
                if inferred is not None:
                    near = await anyio.to_thread.run_sync(
                        nearby_from_filter, inferred[0], inferred[1], flat, flon
                    )
            if near is not None:
                spoken, share = near
                return _reply(spoken, raw=share, mode="nearby")

    # Firestore-linked circle (acts on the app's real data as the caller) wins first
    # when a real token is present; then the in-memory circle; else the orchestrator.
    # Threaded for the same reason as distance: it is blocking urllib + Firestore.
    fs_reply = await anyio.to_thread.run_sync(_try_firestore_circle, request, query)
    if fs_reply is not None:
        return _reply(for_speech(fs_reply), raw=fs_reply, mode="circle")

    # Instagram DMs ("any instagram dms", "read my instagram messages", "reply to X
    # on instagram …") — only an Instagram phrase touches the API; else falls through.
    # instagrapi is entirely synchronous, so this must not run on the event loop.
    ig_reply = await anyio.to_thread.run_sync(_try_instagram, request, query)
    if ig_reply is not None:
        return _reply(for_speech(ig_reply), raw=ig_reply, mode="instagram")

    # Circle status intents ("what's X doing", "set my status…") win when the
    # caller is known and the phrasing matches; otherwise fall through below.
    circle_reply = _try_circle(request, query)
    if circle_reply is not None:
        return _reply(for_speech(circle_reply), raw=circle_reply, mode="circle")

    # Fast voice path: one persona'd LLM call instead of the full orchestrator graph
    # (routing, memory, optional critic re-pass, confidence) — much lower latency.
    # Falls through to the orchestrator below when it yields nothing. Kill-switch:
    # FRIDAY_SIRI_FAST_PATH=false.
    settings = getattr(request.app.state, "settings", None)
    if getattr(settings, "siri_fast_path", True):
        try:
            fast = await _fast_answer(request, query, history, deadline)
        except TimeoutError:
            # No budget left to also try the slower orchestrator — say so now,
            # while Siri is still listening, rather than answering into silence.
            return _reply(_TIMEOUT_SPEECH, raw="", mode="timeout", record=False)
        if fast:
            return _reply(for_speech(fast), raw=fast, mode="fast")

    orchestrator = getattr(request.app.state, "orchestrator", None)
    if orchestrator is None or not hasattr(orchestrator, "handle"):
        logger.error("siri ask: orchestrator missing on app.state")
        return _reply(_FALLBACK_SPEECH, raw="", mode=None, record=False)

    # The orchestrator keeps its own short-term memory under this same session id,
    # so it needs no history injected — it reloads the very buffer written above.
    state = GraphState(session_id=session_id, user_input=_augment_teaching(query))
    result = None
    try:
        with anyio.move_on_after(deadline):
            result = await orchestrator.handle(state)
    except FridayError as exc:
        logger.warning(
            "siri ask raised FridayError",
            extra={"error_type": type(exc).__name__},
        )
        return _reply(_FALLBACK_SPEECH, raw="", mode=None, record=False)
    except Exception:  # noqa: BLE001 - Siri must never read a raw 500 to the user
        logger.exception("siri ask: unexpected error; speaking a graceful fallback")
        return _reply(_FALLBACK_SPEECH, raw="", mode=None, record=False)
    if result is None:
        logger.warning("siri ask: orchestrator exceeded its %.1fs budget", deadline)
        return _reply(_TIMEOUT_SPEECH, raw="", mode="timeout", record=False)

    raw_text = getattr(result, "response", None) or ""
    speech = for_speech(raw_text) or _FALLBACK_SPEECH
    mode = getattr(getattr(result, "mode", None), "value", None)
    # The orchestrator already recorded this turn in the shared buffer; recording
    # it again here would double every orchestrator-answered exchange.
    return _reply(speech, raw=raw_text, mode=mode, record=False)


def _discover_chat_id(token: str) -> str:
    """Find the most recent chat that messaged the bot (so no chat_id env needed)."""
    import json  # noqa: PLC0415
    import urllib.request  # noqa: PLC0415

    url = f"https://api.telegram.org/bot{token}/getUpdates"
    try:
        with urllib.request.urlopen(url, timeout=8) as resp:  # noqa: S310
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:  # noqa: BLE001
        return ""
    for update in reversed(data.get("result", []) if isinstance(data, dict) else []):
        chat = (update.get("message") or update.get("channel_post") or {}).get("chat", {})
        if chat.get("id"):
            return str(chat["id"])
    return ""


def _send_telegram(request: Request, text: str) -> bool:
    """Send ``text`` to the configured (or auto-discovered) Telegram chat."""
    settings = getattr(request.app.state, "settings", None)
    secret = getattr(settings, "telegram_bot_token", None)
    chat_id = getattr(settings, "telegram_chat_id", "") or ""
    token = secret.get_secret_value() if secret is not None else ""
    if not token:
        return False
    if not chat_id:
        chat_id = _discover_chat_id(token)  # whoever last messaged the bot
    if not chat_id:
        return False
    import urllib.parse  # noqa: PLC0415
    import urllib.request  # noqa: PLC0415

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    body = urllib.parse.urlencode({"chat_id": chat_id, "text": text}).encode()
    try:
        with urllib.request.urlopen(  # noqa: S310
            urllib.request.Request(url, data=body), timeout=8
        ) as resp:
            return bool(200 <= resp.status < 300)
    except Exception:  # noqa: BLE001 - never raise to the caller
        return False


@router.post("/siri/telegram", response_model=None)
async def siri_telegram(request: Request) -> Any:
    """Smart-share to Telegram: the model extracts only the key content (never the
    whole transcript). When the request is too vague it speaks a question instead
    of sending, so the user can clarify."""
    if not _siri_enabled(request):
        return _disabled()
    text = await _read_query(request)
    if not text:
        return JSONResponse(status_code=400, content={"detail": "missing text"})
    want_json = request.query_params.get("format", "").lower() == "json"

    from friday.notify import smart_share  # noqa: PLC0415

    llm = getattr(request.app.state, "llm", None)
    message, question = await smart_share(llm, text)
    if message is None:
        ask = question or "What should I send, Boss?"
        return _respond(ask, raw=ask, mode="telegram_ask", want_json=want_json)

    ok = await anyio.to_thread.run_sync(_send_telegram, request, message)
    msg = (
        "Shared on Telegram, Boss."
        if ok
        else "Telegram isn't set up yet — add the bot token and chat id."
    )
    return _respond(msg, raw=message, mode="telegram", want_json=want_json)


@router.api_route("/siri/digest", methods=["GET", "POST"], response_model=None)
async def siri_digest(request: Request) -> Any:
    """Build the daily brief (weather forecast + news) and push it to Telegram.

    Designed for a free external cron (cron-job.org / UptimeRobot) to hit at 6 AM:
    ``GET /siri/digest?key=…&lat=…&lon=…``. ``key`` must match ``digest_key`` when
    that setting is non-empty (open otherwise); ``lat``/``lon`` set the forecast
    location, defaulting to ``digest_lat``/``digest_lon``.
    """
    if not _siri_enabled(request):
        return _disabled()
    settings = getattr(request.app.state, "settings", None)
    secret = getattr(settings, "digest_key", "") or ""
    if secret and request.query_params.get("key", "") != secret:
        return JSONResponse(status_code=403, content={"detail": "forbidden"})

    lat = request.query_params.get("lat") or getattr(settings, "digest_lat", "") or ""
    lon = request.query_params.get("lon") or getattr(settings, "digest_lon", "") or ""

    from friday.notify import build_digest  # noqa: PLC0415

    digest = await build_digest(str(lat), str(lon))
    sent = await anyio.to_thread.run_sync(_send_telegram, request, digest)
    return JSONResponse(status_code=200, content={"sent": sent, "digest": digest})
