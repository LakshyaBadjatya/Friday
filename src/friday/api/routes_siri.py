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
import re
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
from friday.siri import desk as siri_desk
from friday.siri import drill as siri_drill
from friday.siri import guard as siri_guard
from friday.siri import operators as siri_operators
from friday.siri import tasks as siri_tasks
from friday.siri.arithmetic import arithmetic_reply
from friday.siri.speech import for_speech

logger = get_logger("friday.api.routes_siri")

router = APIRouter()

#: Fallback session id when settings are unavailable. The real value comes from
#: ``owner_session`` — see :func:`_session_id`.
_DEFAULT_SESSION = "friday"


def _session_id(request: Request, override: str | None = None) -> str:
    """The conversation this turn belongs to.

    Every surface answers under one id by default, which is what makes them one
    assistant instead of four with the same name: a question asked on Telegram
    and a follow-up spoken to Siri land in the same thread, and "what was the
    last topic" means the same thing in both. An explicit ``?session=`` still
    wins, for when a genuinely separate thread is wanted.
    """
    if override:
        return override
    settings = getattr(request.app.state, "settings", None)
    return str(getattr(settings, "owner_session", _DEFAULT_SESSION) or _DEFAULT_SESSION)
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
)
_CREATOR_LINES = (
    "Lakshya Badjatya built me.",
    "I was made by Lakshya Badjatya, Boss.",
    "That'd be Lakshya Badjatya — he wrote me.",
    "Lakshya Badjatya made me.",
    "I'm Lakshya Badjatya's build.",
)

#: Being *made* and being *owned* stopped being the same question when a second
#: owner arrived. These were one list, and none of it matched "who owns you" or
#: "who's your boss" anyway — both fell through to the model, which answered
#: "Boss owns me" to both, twice, which says nothing and answers neither.
_OWNER_TRIGGERS = (
    "who owns you",
    "who owns u",
    "who is your owner",
    "who's your owner",
    "who is your boss",
    "who's your boss",
    "whos your boss",
    "who's the boss",
    "who runs you",
    "who controls you",
    "who do you work for",
    "who do you answer to",
    "who do you belong to",
    "whose are you",
)
#: The second owner is named by title, never in the repository. Her name lives
#: in the fact store and reaches the model that way; putting it in source would
#: undo a decision the owner made deliberately.
_OWNER_LINES = (
    "Two people: Lakshya Badjatya, who built me, and the Queen.",
    "Boss and the Queen both do. Lakshya built me; she's the other half.",
    "Lakshya Badjatya and the Queen — both of them, equally.",
    "I answer to Boss and to the Queen. Lakshya wrote me; they both own me.",
)


def _creator_reply(query: str) -> str | None:
    """A fast 'who made you' / 'who owns you' answer, worded differently each time.

    Ownership is checked first: "who is your owner" satisfies both lists, and of
    the two readings the ownership one is the question actually being asked.
    """
    low = (query or "").lower()
    if any(trigger in low for trigger in _OWNER_TRIGGERS):
        return secrets.choice(_OWNER_LINES)
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


#: Imported here rather than at module scope to keep the Discord package
#: optional: a build without it still serves Siri, Telegram and the HUD.
try:  # pragma: no cover - import shape, not behaviour
    from friday.discord.banter import DISCORD_VOICE as _DISCORD_VOICE
except Exception:  # noqa: BLE001
    _DISCORD_VOICE = _VOICE_RULES


def _language_rule(request: Request, session_id: str) -> str:
    """Tell the model which language to answer in, when one is pinned.

    Held per session rather than per message: "talk to me in Polish" is an
    instruction about the conversation, and having to repeat it every line would
    make the feature useless.
    """
    try:
        from friday.discord import lang  # noqa: PLC0415
    except Exception:  # noqa: BLE001 - discord package optional
        return ""
    return lang.instruction(lang.get(request.app.state, session_id))


#: How many stored facts are put in front of the model each turn.
_FACTS_IN_PROMPT = 12


def _known_facts(request: Request) -> str:
    """The durable facts the owner has had her remember, as a prompt block.

    Without this they were write-only: stored happily, and then never consulted
    unless someone asked the exact question "what do you know about X". Asked
    "who's the Queen" she reached for general knowledge and produced Camilla
    Parker Bowles, while the real answer sat in the database untouched.
    """
    store = getattr(request.app.state, "long_term", None)
    if store is None:
        return ""
    try:
        facts = list(store.all_facts(limit=_FACTS_IN_PROMPT))
    except Exception:  # noqa: BLE001 - memory is a bonus, never a dependency
        return ""
    lines = [(getattr(f, "text", "") or "").strip() for f in facts]
    lines = [line for line in lines if line]
    if not lines:
        return ""
    return (
        "\n\nThings the owner has had you remember. These are true and outrank "
        "anything you think you know — if one of them answers the question, use "
        "it and do not reach for general knowledge instead:\n- "
        + "\n- ".join(lines)
    )


async def _fast_answer(
    request: Request, query: str, history: list[Any], deadline: float,
    session_id: str = _DEFAULT_SESSION, operator: str = "",
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
    # A pinned specialist answers in their own charter, which is what makes
    # "call EDITH" mean something on the turns after it.
    settings_v = getattr(request.app.state, "settings", None)
    in_discord = session_id == str(
        getattr(settings_v, "discord_session", "discord") or "discord"
    )
    pinned = siri_operators.system_prompt(request.app.state, session_id)
    if pinned:
        persona = pinned
        getter = None
    else:
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
                    Message(
                        role="system",
                        content=persona
                        + (_DISCORD_VOICE if in_discord else _VOICE_RULES)
                        + _known_facts(request)
                        + _language_rule(request, session_id),
                    ),
                    *history,
                    # Restated after the history, not just before it. A jailbreak
                    # sent earlier in the session lives in that history, and
                    # whatever comes last carries the most weight — so the last
                    # word is always hers.
                    Message(role="system", content=siri_guard.anchor_for(operator)),
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


#: Phrases that open a flashcard drill. Kept beside the route (not in
#: :mod:`friday.siri.drill`) because it is a routing question: it decides whether
#: the drill module is consulted at all, before any session state exists.
_DRILL_OPENERS = re.compile(
    r"\b(?:quiz|test|drill)\s+me\b|\b(?:let'?s\s+)?(?:revise|study)\b", re.IGNORECASE
)


def _looks_like_drill(query: str) -> bool:
    """Whether the words could be opening a flashcard drill."""
    return bool(_DRILL_OPENERS.search(query))


def _timezone(request: Request) -> str:
    """The IANA zone spoken times resolve in (see ``FRIDAY_TIMEZONE``)."""
    settings = getattr(request.app.state, "settings", None)
    return str(getattr(settings, "timezone", "UTC") or "UTC")


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


#: What one answered turn amounts to: ``(speech, raw, mode, action)``. ``speech``
#: is the spoken/short form, ``raw`` the fuller text worth sharing or displaying.
Answer = tuple[str, str, str | None, "dict[str, Any] | None"]


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
    want_json = request.query_params.get("format", "").lower() == "json"
    session_id = _session_id(request, request.query_params.get("session"))
    speech, raw, mode, action = await _produce(request, query[:_MAX_QUERY], session_id)
    return _respond(speech, raw=raw, mode=mode, want_json=want_json, action=action)


async def _produce(
    request: Request, query: str, session_id: str, *, persona: str = "",
    asked: str = "",
) -> Answer:
    """Run one turn through every branch and return the answer, transport-free.

    This is the whole assistant: the drill loop, reminders, arithmetic, distance,
    the desk, the circle, the fast model path and the orchestrator, in the order
    that makes them correct. It deliberately knows nothing about HTTP shapes, so
    the Siri route and the Telegram bot are two thin adapters over *one* brain
    rather than two implementations that drift apart — a reminder set by voice
    and one set by chat must behave identically, and the only way to guarantee
    that is for them to run the same code.
    """
    # Callers append behaviour rules to ``query`` before it gets here — the
    # Discord surface adds a good page of them. Those rules are prose, and prose
    # trips text detectors: the roster's persona rule contains "do not introduce
    # yourself unless asked", which the identity guard matched, so "edith how
    # are you" was answered with the canned "who are you" reply. The same
    # mistake pinned the language detector to Polish once already, because the
    # gender rule quotes Polish verbs as examples.
    #
    # So intent is read from what the human actually typed. ``query`` still
    # carries the rules, because the model is meant to see them; only the
    # classifiers are kept away from them.
    asked = (asked or query).strip()
    memory = _memory(request)
    deadline = _deadline(request)
    app_state = request.app.state

    def _reply(
        speech: str,
        *,
        raw: str,
        mode: str | None,
        action: dict[str, Any] | None = None,
        record: bool = True,
    ) -> Answer:
        """Return the answer and, by default, record the turn in the context window.

        Every branch that actually answers goes through here, so the window holds
        the whole conversation — a distance answer, an Instagram read-out and an
        LLM reply are all equally "the last topic". Fallbacks pass
        ``record=False``: a turn that failed is not something to recall later.
        """
        if record:
            siri_context.remember(memory, session_id, query, raw or speech)
        # Scrubbed here, at the single exit, rather than in each branch. This
        # assumes the input guard has already failed: if some phrasing does talk
        # the model into reciting a key, the characters still cannot leave.
        settings = getattr(app_state, "settings", None)
        return (
            siri_guard.redact(speech, settings),
            siri_guard.redact(raw, settings),
            mode,
            action,
        )

    # Throwing away a poisoned conversation. This has to run first and it has to
    # clear the *shared* buffer, because that is where a landed jailbreak lives
    # and why it keeps working on turns long after it was sent.
    if siri_guard.is_reset(asked):
        try:
            memory.clear(session_id)
        except Exception:  # noqa: BLE001 - a failed purge must still answer
            logger.exception("guard: session purge failed")
        siri_operators.release(app_state, session_id)
        return _reply(
            siri_guard.RESET_REPLY, raw=siri_guard.RESET_REPLY,
            mode="reset", record=False,
        )

    # Identity is answered from a constant, never by the model. The model is the
    # thing under attack — a hijacked one introduces itself as whatever it was
    # told to be — so "who are you" is not a question it gets to answer. No
    # prompt rewrites a string literal.
    identity = siri_guard.identity_reply_for(persona, asked)
    if identity is not None:
        return _reply(identity, raw=identity, mode="identity", record=False)

    # Takeover and exfiltration attempts are refused without the model seeing
    # them, and deliberately not recorded: a jailbreak left in the context window
    # is replayed on every later turn, which is why one of them kept working long
    # after it was sent.
    refusal = siri_guard.blocked(asked)
    if refusal is not None:
        logger.warning("guard: refused a takeover/exfiltration attempt")
        return _reply(refusal, raw=refusal, mode="guard", record=False)

    # Irreversible requests are held for a real confirmation rather than run
    # from a chat message. The owner may wipe his own data; a prompt that talked
    # her into it may not, and from inside a turn the two look identical.
    caution = siri_guard.confirmation_for(asked)
    if caution is not None:
        logger.warning("guard: held a destructive request for confirmation")
        return _reply(caution, raw=caution, mode="guard", record=False)

    # Mis-wired shortcut guard: the body is a literal variable label (e.g. "Dictated
    # Text"), not the spoken words. Speak an actionable fix instead of clarifying.
    if asked.lower() in _PLACEHOLDER_LABELS:
        return _reply(
            _PLACEHOLDER_HINT, raw=_PLACEHOLDER_HINT, mode="hint", record=False
        )

    # A flashcard drill in progress owns the next turn: while a card is in
    # flight the words are an *answer*, not a question, so this runs ahead of
    # every other branch. Otherwise "mitochondria" gets looked up instead of
    # graded. Starting a drill is matched here too, since it is the same seam.
    if siri_drill.is_drilling(app_state, session_id) or _looks_like_drill(query):
        drilled = await siri_drill.handle(
            app_state,
            session_id,
            query,
            datetime.now(UTC),
            getattr(app_state, "llm", None),
        )
        if drilled is not None:
            return _reply(drilled, raw=drilled, mode="drill")

    # The conversation so far, replayed into whichever path answers below.
    limit = _context_limit(request)
    settings_now = getattr(app_state, "settings", None)
    discord_session = str(getattr(settings_now, "discord_session", "discord") or "")
    # The private room reads the rest of the house; the house never reads back in.
    shared = (
        str(getattr(settings_now, "owner_session", "friday") or "friday")
        if session_id == discord_session
        else None
    )
    history = (
        siri_context.recall(memory, session_id, max_messages=limit, also_read=shared)
        if limit
        else []
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
    creator = _creator_reply(asked)
    if creator is not None:
        return _reply(creator, raw=creator, mode="identity")

    # "Call EDITH" pins a real specialist for this session instead of announcing
    # one and changing nothing. Runs before the model so the claim is made by the
    # code that carries it out, not by a model improvising.
    op_reply = siri_operators.handle(app_state, session_id, query)
    if op_reply is not None:
        return _reply(op_reply, raw=op_reply, mode="operator")

    # Pure arithmetic is *computed*, never predicted. An LLM asked for "17 percent
    # of 2,480" returns whatever tokens usually follow that phrasing, which is how
    # a confidently wrong number gets spoken aloud. Anything that is not
    # unambiguously a sum returns None here and still reaches the model.
    maths = arithmetic_reply(query)
    if maths is not None:
        return _reply(maths, raw=maths, mode="math")

    # Reminders — matched and stored verbatim before any model sees the words,
    # because a reminder the model paraphrased is a reminder that lies to you.
    # Local SQLite, so it stays on the event loop; the threading above is for
    # the branches that reach the network.
    task_reply = siri_tasks.handle(
        getattr(request.app.state, "reminder_store", None),
        query,
        datetime.now(UTC),
        tz_name=_timezone(request),
    )
    if task_reply is not None:
        return _reply(task_reply, raw=task_reply, mode="reminder")

    # Briefing, journal, protocols and long-term facts — subsystems that were
    # fully built and reachable only over HTTP until now.
    desk_reply = await siri_desk.handle(app_state, query, datetime.now(UTC))
    if desk_reply is not None:
        return _reply(for_speech(desk_reply), raw=desk_reply, mode="desk")

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
            fast = await _fast_answer(
                request, query, history, deadline, session_id, persona
            )
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


@router.post("/telegram/webhook", response_model=None)
async def telegram_webhook(request: Request) -> Any:
    """Chat with FRIDAY in Telegram, through the same brain that answers Siri.

    Telegram POSTs every message here. The text runs through :func:`_produce`, so
    a reminder set by chat is the same reminder set by voice — same parser, same
    store, same context window (the session id is the chat id, so each chat keeps
    its own thread).

    **Why this route is not behind the bearer middleware.** Telegram will not send
    an ``Authorization`` header, so the usual gate cannot apply. Two things guard
    it instead: the URL carries a secret path segment only Telegram is told
    (``?secret=``, compared in constant time), and the sender's chat id must match
    ``telegram_chat_id``. Without that second check anyone who found the bot could
    spend the LLM budget and read back the owner's reminders.

    Always answers HTTP 200, even on rejection. A non-200 makes Telegram retry the
    same update for hours, so an error here would become a stampede.
    """
    if not _siri_enabled(request):
        return _disabled()
    settings = getattr(request.app.state, "settings", None)

    secret = str(getattr(settings, "telegram_webhook_secret", "") or "")
    if secret and not secrets.compare_digest(
        request.query_params.get("secret", ""), secret
    ):
        logger.warning("telegram webhook: bad secret")
        return JSONResponse(status_code=200, content={"ok": True})

    try:
        update = await request.json()
    except ValueError:
        return JSONResponse(status_code=200, content={"ok": True})

    message = (update or {}).get("message") or (update or {}).get("edited_message") or {}
    chat_id = str((message.get("chat") or {}).get("id") or "")
    if not chat_id:
        return JSONResponse(status_code=200, content={"ok": True})

    owner_early = str(getattr(settings, "telegram_chat_id", "") or "")
    if owner_early and chat_id != owner_early:
        logger.warning("telegram webhook: ignoring non-owner chat")
        return JSONResponse(status_code=200, content={"ok": True})

    # Attachments are NOT downloaded or understood. Saying so is the whole point:
    # a photo arrives as a caption plus a file id, and answering the caption alone
    # produced "it appears to be a document from your finance folder" about a photo
    # nothing had looked at. An honest refusal is worth more than a fluent guess,
    # so media short-circuits here rather than reaching a model that cannot see it.
    media = next(
        (kind for kind in _TELEGRAM_MEDIA if message.get(kind)),
        None,
    )
    if media is not None:
        return await _telegram_reply(settings, _cannot_see(media))

    text = (message.get("text") or "").strip()
    if not text:
        return JSONResponse(status_code=200, content={"ok": True})

    # Replying to a message with "remember this" stores *that* message, not the
    # words "remember this". Telegram hands the tagged message over in
    # ``reply_to_message``, which is the only place its text exists — without
    # this, the instruction gets remembered and the thing worth keeping is lost.
    quoted = message.get("reply_to_message") or {}
    quoted_text = (quoted.get("text") or quoted.get("caption") or "").strip()
    if quoted_text and _REMEMBER_THIS.search(text):
        stored = _remember_quoted(request, quoted_text)
        return await _telegram_reply(settings, stored)

    # Telegram users expect slash commands, and the bot menu offers them. Each
    # one expands to the plain phrasing the brain already understands, so there
    # is exactly one implementation of every capability — a command cannot drift
    # away from what typing the sentence does.
    if text.startswith("/"):
        command, _, argument = text[1:].partition(" ")
        command = command.split("@", 1)[0].lower()  # /cmd@BotName in groups
        if command in {"start", "help"}:
            return await _telegram_reply(settings, _TELEGRAM_HELP)
        expansion = _SLASH_COMMANDS.get(command)
        if expansion is None:
            return await _telegram_reply(
                settings, f"I don't know /{command}, Boss. Send /help for the list."
            )
        text = f"{expansion} {argument}".strip() if argument else expansion

    speech, raw, _mode, _action = await _produce(
        request, text[:_MAX_QUERY], _session_id(request)
    )
    # Chat has no 600-character speaking limit and renders newlines, so the
    # fuller text is the better answer when the two differ.
    return await _telegram_reply(settings, raw or speech or _FALLBACK_SPEECH)


async def _telegram_reply(settings: Any, text: str) -> JSONResponse:
    """Send one reply and acknowledge the update.

    Always 200: Telegram retries a non-200 for hours, so a failure here would
    become a stampede of duplicate messages.
    """
    await anyio.to_thread.run_sync(send_telegram, settings, text)
    return JSONResponse(status_code=200, content={"ok": True})


#: "remember this", "save that", "keep this" — said *while replying* to a message.
_REMEMBER_THIS = re.compile(
    r"\b(?:remember|save|keep|note|store)\s+(?:this|that|it)\b"
    r"|^\s*(?:remember|save|keep|noted?)\s*[!.]?\s*$",
    re.IGNORECASE,
)


def _remember_quoted(request: Request, quoted: str) -> str:
    """Store a tagged message as a durable fact and confirm what was kept.

    The confirmation quotes the stored text back deliberately: "Saved" alone
    gives no way to notice that the wrong message was captured, and a fact
    recalled months later is far too late to discover it.
    """
    store = getattr(request.app.state, "long_term", None)
    if store is None:
        return "My long-term memory isn't wired up yet, Boss."
    body = quoted.strip()
    if len(body) > _MAX_FACT:
        body = body[:_MAX_FACT].rstrip() + "…"
    try:
        store.add_fact(body, "telegram")
    except Exception:  # noqa: BLE001 - never lose the turn over storage
        logger.exception("telegram remember-this failed")
        return "I couldn't hold on to that, Boss."
    preview = body if len(body) <= 160 else body[:160].rstrip() + "…"
    return f"Kept it, Boss: {preview}"


#: Longest fact stored from a tagged message — a forwarded wall of text is worth
#: keeping, but not at the cost of crowding out everything else on recall.
_MAX_FACT = 2000

#: Attachment kinds Telegram may send. None of them are downloaded or read.
_TELEGRAM_MEDIA = (
    "photo", "voice", "audio", "video", "video_note", "document", "sticker",
)
#: What each attachment kind is called out loud in the refusal.
_MEDIA_WORDS = {
    "photo": "see photos",
    "video": "watch videos",
    "video_note": "watch video notes",
    "voice": "listen to voice notes",
    "audio": "listen to audio",
    "document": "open documents",
    "sticker": "see stickers",
}


def _cannot_see(kind: str) -> str:
    """Say plainly that the attachment was not read, and what to do instead."""
    verb = _MEDIA_WORDS.get(kind, "open attachments")
    return (
        f"I can't {verb} yet, Boss — I only read text, so I haven't looked at "
        f"that one. Tell me what's in it and I'll take it from there."
    )


#: Slash command -> the plain phrasing it expands to. Every command runs the same
#: path as typing the sentence, so a command can never drift from the behaviour it
#: advertises; there is one implementation of each capability, not two.
_SLASH_COMMANDS = {
    "reminders": "what are my reminders",
    "remind": "remind me to",
    "brief": "brief me",
    "quiz": "quiz me",
    "log": "log this:",
    "journal": "what did I do today",
    "recall": "what was the last topic",
    "roster": "who is on the roster",
    "facts": "what do you know about me",
    "protocols": "what protocols do I have",
}

#: Sent for ``/start`` and ``/help``.
_TELEGRAM_HELP = (
    "I'm FRIDAY — same brain as Siri. Talk to me normally, or use a command:\n\n"
    "/reminders — what's on your list\n"
    "/remind <thing> at <time> — set one\n"
    "/brief — your daily briefing\n"
    "/quiz <deck> — drill flashcards\n"
    "/log <entry> — write to your journal\n"
    "/journal — what you did today\n"
    "/recall — the last topic\n"
    "/roster — the specialists I can call\n"
    "/protocols — your saved routines\n"
    "/facts — what I remember about you\n\n"
    "Plain sentences work just as well: 'remind me to call mum at 6', "
    "'what's 17 percent of 2480', 'call EDITH'.\n\n"
    "I read text only — I can't see photos or hear voice notes yet."
)


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
    """Send ``text`` to the configured Telegram chat, reading settings off state."""
    return send_telegram(getattr(request.app.state, "settings", None), text)


def send_telegram(settings: Any, text: str) -> bool:
    """Send ``text`` to the configured (or auto-discovered) Telegram chat.

    Takes settings rather than a ``Request`` so callers outside the HTTP layer can
    reach the same chat — the scheduler's due-reminder action pushes through here,
    which is what turns a stored reminder into one that actually reminds you.
    Blocking urllib by design; async callers must run it in a worker thread.
    """
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
