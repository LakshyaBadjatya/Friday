"""Actually doing things to the server, rather than saying she will.

Asked to create a role she answered "i can do that 👑" and then, one message
later, "nah, can't do that." Both were guesses: she had no server-management
ability at all and no way to know it, so the model filled the gap twice and
contradicted itself. This module is the ability, so the claim can be true.

Deliberately narrow: create a role, hand it to someone, and nothing else. No
deleting, banning, kicking or editing the server. A model deciding when to
remove things is a bad idea on a good day, and these are the operations that
were actually asked for.

Every failure reports what to fix. Discord refuses role changes for two very
common reasons that both surface as 403 and look identical to a bug: the bot
needs **Manage Roles**, and its own highest role must sit **above** the one being
granted. Being told that is the difference between a two-second fix and an hour
reading code that is fine.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

import anyio

from friday.logging import get_logger

logger = get_logger("friday.discord.admin")

_API = "https://discord.com/api/v10"

#: "create a queen role", "make a role called X". The name is captured before
#: the word "role" because that is how people say it out loud.
_CREATE = re.compile(
    r"\b(?:create|make|add|set\s+up)\s+(?:a\s+|an\s+|the\s+)?"
    r"(?:role\s+(?:called|named)\s+(?P<named>[\w' ]{1,32})"
    r"|(?P<name>[\w']{1,32}(?:\s[\w']{1,16})?)\s+role)\b",
    re.IGNORECASE,
)
#: "give it to the queen", "give the queen role to X", "assign it to @someone".
_GIVE = re.compile(
    r"\b(?:give|assign|hand)\s+(?:it|that|the\s+role)\s+to\s+(?P<who>[@\w' ]{1,40})"
    r"|\b(?:give|assign)\s+(?:the\s+)?(?P<role>[\w' ]{1,32}?)\s+role\s+to\s+"
    r"(?P<who2>[@\w' ]{1,40})",
    re.IGNORECASE,
)

#: Gold, for the only role anyone has actually asked for so far.
_DEFAULT_COLOUR = 0xF1C40F


def wants_role(text: str) -> tuple[str | None, str | None]:
    """Parse a role request into ``(role_to_create, person_to_give_it_to)``.

    Either half may be ``None``: "make a queen role" creates without assigning,
    "give it to the queen" assigns a role created a moment ago.
    """
    body = (text or "").strip()
    if not body:
        return None, None
    role = None
    created = _CREATE.search(body)
    if created is not None:
        role = (created.group("named") or created.group("name") or "").strip()
    given = _GIVE.search(body)
    who = None
    if given is not None:
        who = (given.group("who") or given.group("who2") or "").strip().lstrip("@")
        role = role or (given.group("role") or "").strip() or None
    return (role or None), (who or None)


async def create_role(token: str, guild: str, name: str) -> tuple[str | None, str]:
    """Create a role; returns ``(role_id, message)``, id ``None`` on failure."""
    body = json.dumps({
        "name": name[:32],
        "color": _DEFAULT_COLOUR,
        "hoist": True,        # listed separately in the member sidebar
        "mentionable": True,
    }).encode()
    ok, data = await _call("POST", f"/guilds/{guild}/roles", token, body)
    if not ok:
        return None, _explain(data, f"couldn't make the {name} role")
    return str(data.get("id") or ""), f"made the **{name}** role 👑"


async def assign_role(token: str, guild: str, user: str, role: str) -> tuple[bool, str]:
    """Give an existing role to a member."""
    ok, data = await _call(
        "PUT", f"/guilds/{guild}/members/{user}/roles/{role}", token, b""
    )
    if not ok:
        return False, _explain(data, "couldn't hand the role over")
    return True, "and it's theirs 👑"


async def find_role(token: str, guild: str, name: str) -> str | None:
    """An existing role id by name, so one is not created twice."""
    ok, data = await _call("GET", f"/guilds/{guild}/roles", token, None)
    if not ok or not isinstance(data, list):
        return None
    wanted = name.strip().lower()
    for role in data:
        if str(role.get("name", "")).strip().lower() == wanted:
            return str(role.get("id"))
    return None


async def find_member(token: str, guild: str, name: str) -> tuple[str, str] | None:
    """Look somebody up by whatever name is on screen.

    Nickname first, then display name, then username — people refer to each
    other by what they see, which is the nickname. A loose contains-match is the
    last resort so "the queen" can find "the second owner 👑" without an exact spelling.
    """
    query = urllib.parse.quote(name.strip()[:32])
    ok, data = await _call(
        "GET", f"/guilds/{guild}/members/search?query={query}&limit=10", token, None
    )
    if not ok or not isinstance(data, list) or not data:
        return None
    wanted = name.strip().lower()
    for member in data:
        user = member.get("user") or {}
        for shown in (member.get("nick"), user.get("global_name"),
                      user.get("username")):
            if shown and str(shown).lower() == wanted:
                return str(user.get("id")), str(shown)
    first = data[0]
    user = first.get("user") or {}
    shown = first.get("nick") or user.get("global_name") or user.get("username")
    return (str(user.get("id")), str(shown)) if user.get("id") else None


#: "call the second owner amster", "set the second owner's nickname to X", "nickname her Amster".
NICKNAME = re.compile(
    r"\b(?:set\s+)?(?:the\s+)?nick(?:name)?\s+(?:of\s+)?(?P<who>[@\w' ]{1,40}?)"
    r"\s+(?:to|as)\s+(?P<nick>.{1,32}?)\s*$"
    r"|\bcall\s+(?P<who2>[@\w' ]{1,40}?)\s+(?P<nick2>.{1,32}?)\s+from\s+now\b"
    r"|\bnickname\s+(?P<who3>[@\w' ]{1,40}?)\s+(?P<nick3>.{1,32}?)\s*$",
    re.IGNORECASE,
)


def wants_nickname(text: str) -> tuple[str | None, str | None]:
    """Parse "set X's nickname to Y" into ``(person, nickname)``."""
    match = NICKNAME.search((text or "").strip())
    if match is None:
        return None, None
    who = (match.group("who") or match.group("who2") or match.group("who3") or "")
    nick = (match.group("nick") or match.group("nick2") or match.group("nick3") or "")
    who = who.strip().lstrip("@").rstrip("'s").strip()
    return (who or None), (nick.strip().strip("\"'") or None)


async def set_nickname(
    token: str, guild: str, user: str, nick: str
) -> tuple[bool, str]:
    """Change a member's server nickname.

    Discord refuses this for the owner of the server no matter what permissions
    the bot has — a server owner's nickname can only be changed by themselves —
    so that case gets its own explanation rather than the generic one.
    """
    body = json.dumps({"nick": nick[:32]}).encode()
    ok, data = await _call(
        "PATCH", f"/guilds/{guild}/members/{user}", token, body
    )
    if ok:
        return True, f"done — they're **{nick}** now 🐹"
    code = (data or {}).get("code") if isinstance(data, dict) else None
    if code == 50013:
        return False, (
            "discord won't let me — i need **Manage Nicknames**, and nobody can "
            "rename the server owner but themselves 🧍"
        )
    return False, _explain(data, "couldn't change that nickname")


async def _call(
    method: str, path: str, token: str, body: bytes | None
) -> tuple[bool, Any]:
    """One Discord REST call, off the event loop."""
    headers = {
        "Authorization": f"Bot {token}",
        "User-Agent": "FRIDAY (https://friday.sukhma.in, 1.0)",
    }
    if body is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(  # noqa: S310
        f"{_API}{path}", data=body, method=method, headers=headers
    )

    def _send() -> tuple[bool, Any]:
        try:
            with urllib.request.urlopen(request, timeout=15) as resp:  # noqa: S310
                raw = resp.read()
            return True, (json.loads(raw) if raw else {})
        except urllib.error.HTTPError as exc:
            try:
                return False, json.loads(exc.read() or b"{}")
            except Exception:  # noqa: BLE001
                return False, {"code": exc.code}
        except Exception:  # noqa: BLE001
            logger.warning("discord admin call failed: %s %s", method, path)
            return False, {}

    return await anyio.to_thread.run_sync(_send)


def _explain(data: Any, prefix: str) -> str:
    """Turn a Discord error into something the owner can act on."""
    code = (data or {}).get("code") if isinstance(data, dict) else None
    message = (data or {}).get("message") if isinstance(data, dict) else ""
    if code == 50013 or message == "Missing Permissions":
        return (
            f"{prefix} — i need **Manage Roles**, and my own role has to sit "
            f"*above* the one i'm handing out. server settings → roles, drag "
            f"F.R.I.D.A.Y to the top 🧍"
        )
    if code == 50001:
        return f"{prefix} — i can't see that properly. check my invite scopes."
    return f"{prefix}. discord said no and didn't say why 🤷"
