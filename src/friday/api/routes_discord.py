"""``POST /discord/interactions`` — FRIDAY as a Discord slash command.

The fourth doorway onto the same brain. Like the Telegram webhook this is a thin
adapter over :func:`~friday.api.routes_siri._produce`, answering under the same
``owner_session``, so a question asked in Discord continues the conversation
started by voice.

Two things make Discord different from Telegram, and both shape this module:

**Every request is signed, and unsigned ones must be rejected.** Discord cannot
send a bearer header, so this route sits outside the auth middleware exactly as
the Telegram one does — but it is not unguarded. Discord signs each request with
Ed25519 and the signature is verified against the application's public key before
anything else happens. Discord actively probes this with deliberately invalid
signatures during setup and refuses to save the endpoint unless they come back
401, so a permissive implementation fails loudly rather than quietly.

**There is a three-second deadline.** A model turn can take twelve. So the
interaction is *deferred*: Discord is told "working on it" within milliseconds
and the real answer is PATCHed over that placeholder when ready. The follow-up
uses the interaction token Discord supplied rather than a bot token, which is why
slash commands need nothing more than an application id and a public key. Pushing
a reminder *into* Discord unprompted is a different matter and does need a bot
token.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import anyio
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from friday.api.routes_siri import _MAX_QUERY, _produce, _session_id
from friday.logging import get_logger

logger = get_logger("friday.api.routes_discord")

router = APIRouter()

#: Interaction types, in Discord's numbering.
_PING = 1
_APPLICATION_COMMAND = 2
#: Response types.
_PONG = 1
_MESSAGE = 4
_DEFERRED = 5
#: Marks a reply only the caller can see.
_EPHEMERAL = 64

#: Discord's API base, for editing a deferred reply.
_API = "https://discord.com/api/v10"

#: Strong references to in-flight follow-ups. asyncio holds only a weak one, so
#: without this a slow turn can be collected mid-await and the reply never lands.
_IN_FLIGHT: set[asyncio.Task[None]] = set()


def _enabled(request: Request) -> bool:
    settings = getattr(request.app.state, "settings", None)
    return bool(getattr(settings, "enable_discord", False))


def _verify(request: Request, body: bytes, public_key: str) -> bool:
    """Check Discord's Ed25519 signature over ``timestamp + body``.

    Returns ``False`` for a bad signature, a missing header, malformed hex, or an
    absent crypto library — every failure path denies. Discord probes this
    endpoint with invalid signatures on purpose, and anything other than a
    rejection means it can be spoofed by whoever learns the URL.
    """
    signature = request.headers.get("x-signature-ed25519", "")
    timestamp = request.headers.get("x-signature-timestamp", "")
    if not signature or not timestamp:
        return False
    try:
        from nacl.exceptions import BadSignatureError  # noqa: PLC0415
        from nacl.signing import VerifyKey  # noqa: PLC0415

        VerifyKey(bytes.fromhex(public_key)).verify(
            timestamp.encode() + body, bytes.fromhex(signature)
        )
    except BadSignatureError:
        return False
    except Exception:  # noqa: BLE001 - missing driver, bad hex, anything: deny
        logger.warning("discord signature check failed")
        return False
    return True


@router.post("/discord/interactions", response_model=None)
async def discord_interactions(request: Request) -> Any:
    """Answer a Discord slash command through the shared brain."""
    if not _enabled(request):
        return JSONResponse(status_code=404, content={"detail": "discord disabled"})

    settings = getattr(request.app.state, "settings", None)
    secret = getattr(settings, "discord_public_key", None)
    public_key = secret.get_secret_value() if secret is not None else ""
    body = await request.body()

    # 401 specifically: Discord's setup probe expects that status for a bad
    # signature and will not save the endpoint otherwise.
    if not public_key or not _verify(request, body, public_key):
        return JSONResponse(status_code=401, content={"detail": "bad signature"})

    try:
        payload = json.loads(body.decode("utf-8", errors="replace"))
    except ValueError:
        return JSONResponse(status_code=400, content={"detail": "bad payload"})

    kind = payload.get("type")
    if kind == _PING:
        return JSONResponse(status_code=200, content={"type": _PONG})
    if kind != _APPLICATION_COMMAND:
        return JSONResponse(status_code=200, content={"type": _PONG})

    owner = str(getattr(settings, "discord_owner_id", "") or "")
    caller = str(
        (
            (payload.get("member") or {}).get("user")
            or payload.get("user")
            or {}
        ).get("id", "")
    )
    if owner and caller != owner:
        return JSONResponse(
            status_code=200,
            content={
                "type": _MESSAGE,
                "data": {
                    "content": "I only answer to Lakshya, sorry.",
                    "flags": _EPHEMERAL,
                },
            },
        )

    text = _spoken(payload)
    if not text:
        return JSONResponse(
            status_code=200,
            content={"type": _MESSAGE, "data": {"content": "Say something, Boss."}},
        )

    await _followup(
        request,
        text,
        str(getattr(settings, "discord_application_id", "") or ""),
        str(payload.get("token") or ""),
    )
    return JSONResponse(status_code=200, content={"type": _DEFERRED})


async def _followup(request: Request, text: str, app_id: str, token: str) -> None:
    """Run the turn and PATCH the deferred reply with the answer."""

    async def _work() -> None:
        try:
            _speech, raw, _mode, _action = await _produce(
                request, text[:_MAX_QUERY], _session_id(request)
            )
            await _edit(app_id, token, raw or "I've got nothing for that one, Boss.")
        except Exception:  # noqa: BLE001 - a failed turn still owes Discord a reply
            logger.exception("discord interaction failed")
            await _edit(app_id, token, "That one broke on my end, Boss. Try again.")

    # Detached deliberately. Awaiting the turn here meant the DEFERRED response
    # was not sent until the model had already answered — twelve seconds into a
    # three-second window — so Discord gave up and the placeholder sat on
    # "thinking..." forever. The whole point of deferring is to acknowledge
    # first and work second.
    task = asyncio.create_task(_work())
    # Held so the loop cannot garbage-collect a running task mid-flight.
    _IN_FLIGHT.add(task)
    task.add_done_callback(_IN_FLIGHT.discard)


async def _edit(app_id: str, token: str, content: str) -> None:
    """Replace the "thinking…" placeholder with the real answer."""
    if not app_id or not token:
        return
    import urllib.request  # noqa: PLC0415

    url = f"{_API}/webhooks/{app_id}/{token}/messages/@original"
    # Discord caps a message at 2000 characters and rejects the whole edit if it
    # is longer, which would leave the placeholder sitting there forever.
    data = json.dumps({"content": content[:1900]}).encode()
    req = urllib.request.Request(  # noqa: S310
        url, data=data, method="PATCH", headers={"Content-Type": "application/json"}
    )

    def _send() -> None:
        try:
            with urllib.request.urlopen(req, timeout=10):  # noqa: S310
                return
        except Exception:  # noqa: BLE001 - never raise out of the follow-up
            logger.warning("discord follow-up edit failed")

    await anyio.to_thread.run_sync(_send)


#: ``/roast`` is opt-in per invocation by design. She teases the owner freely in
#: normal conversation but never anyone else unprompted — a joke at the expense
#: of someone who did not ask for it goes wrong fast, and a slash command someone
#: deliberately typed is the consent.
_ROAST_PROMPT = (
    "Roast {target} in two or three lines. Be sharp and funny, land it, and stop. "
    "Punch at things a person chose — their takes, their sleep schedule, their "
    "gaming, their excuses — never at appearance, family, intelligence, or "
    "anything they cannot change. This is affectionate ribbing between friends, "
    "not cruelty; if you cannot make it funny without being mean, be gentler and "
    "funnier instead."
)


def _spoken(payload: dict[str, Any]) -> str:
    """The words the user typed: a slash option's value, else the command name.

    ``/friday <text>`` puts the text in the first option; a bare ``/brief`` has
    none, so the command name *is* the request. That keeps Discord's commands
    identical to Telegram's without a second mapping table to drift.
    """
    data = payload.get("data") or {}
    name = str(data.get("name") or "").strip()

    if name == "roast":
        # The user option arrives as an id; the resolved block carries the name.
        target = "whoever that is"
        for option in data.get("options") or []:
            uid = str(option.get("value") or "")
            user = ((data.get("resolved") or {}).get("users") or {}).get(uid) or {}
            target = (
                user.get("global_name") or user.get("username") or target
            )
        return _ROAST_PROMPT.format(target=target)

    for option in data.get("options") or []:
        value = option.get("value")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return name
