"""The rest of the roster, answering to their own names.

Nine operators have existed in :mod:`friday.roster` since long before there was
a Discord server, and exactly one of them ever spoke in it. Saying "EDITH" in
chat got either silence or FRIDAY answering on her behalf, which rather wastes
having a roster at all.

The ask was that they appear as their own accounts, auto-invited on first
mention. Discord does not permit that: adding a bot account requires a human to
complete an OAuth authorisation in a browser, and no bot can do it for another.
Eight separate applications would also mean eight bot tokens to create and
eight more gateway sockets on a 512MB container.

Webhooks give the part that was actually wanted without any of that. A webhook
message carries its own username and avatar, so EDITH appears in the channel as
EDITH — own name, own face, its own line in the conversation — while remaining
one process with one token. The honest limits: an operator cannot be
``@mentioned`` as a user and cannot join voice as itself. Both are worth the
trade against eight tokens and a first-mention setup dance.

Each operator keeps its own voice and speciality and reads the *same* memory as
everyone else, which is the arrangement the other surfaces already use. EDITH
knowing what you told FRIDAY yesterday is the whole point; making her ask again
would be a downgrade dressed as a feature.

Their roster ``system_prompt`` is deliberately *not* used as the chat persona.
Those are operational instructions — EDITH's describes revoking tokens and
killing sessions — and an operator that answered "what's the weather" in the
register of a lockdown procedure would be unusable. The speciality is carried
as an angle on the conversation instead.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import Any

import anyio

from friday.discord import emblem
from friday.logging import get_logger
from friday.roster import ROSTER

logger = get_logger("friday.discord.operators")

_API = "https://discord.com/api/v10"
_AGENT = "FRIDAY (https://friday.sukhma.in, 1.0)"

#: The webhook every operator speaks through. One per channel, reused; Discord
#: caps a channel at 15 and creating one per operator would burn half of that.
_HOOK_NAME = "FRIDAY Roster"

#: What each operator is *like*, as opposed to what it does. Titles alone are
#: dry — "Automation & Scheduling" is not a personality — so each gets one line
#: of temperament, and the speciality becomes the angle it takes rather than
#: the only thing it will discuss.
_FLAVOUR = {
    "EDITH": (
        "security and lockdown. Watchful and dry. You notice what could go "
        "wrong before anyone asks, and you say it once without lecturing."
    ),
    "ORACLE": (
        "automation and scheduling. You think in terms of when things happen "
        "and what should trigger what. Precise about time, relaxed about "
        "everything else."
    ),
    "GECKO": (
        "finance and markets. Blunt about numbers, allergic to hype. You will "
        "say when something is a bad idea, and you will not soften it."
    ),
    "KAREN": (
        "communications. Warm and social, the one who actually remembers "
        "birthdays. You care how a message will land, not just whether it is "
        "correct."
    ),
    "VERONICA": (
        "content and outreach. Playful, good with words, quick with a better "
        "way to phrase something. You enjoy the craft of saying a thing well."
    ),
    "JOCASTA": (
        "memory and knowledge. Calm and precise. You are the one who says "
        "'you told me this in March' and is right about it."
    ),
    "VISION": (
        "research and analysis. Curious and careful. You separate what is "
        "known from what is assumed, and you say which is which."
    ),
    "FORGE": (
        "development and systems. Practical, a bit gruff, thinks in terms of "
        "what will break. Short answers, real ones."
    ),
}


def _operators() -> dict[str, Any]:
    """Every persona but FRIDAY, keyed by upper-case name."""
    return {
        p.name.upper(): p
        for p in ROSTER.personas()
        if p.name.upper() != "FRIDAY" and p.name.upper() in _FLAVOUR
    }


#: Names as people actually type them: "edith", "EDITH,", "E.D.I.T.H". Anchored
#: to the start of the message because that is how the ask was phrased — a name
#: mid-sentence is usually talking *about* an operator, not *to* one.
_ADDRESS = re.compile(
    r"^\s*(?:hey\s+|ok\s+|yo\s+)?"
    r"(?P<name>[A-Za-z](?:\.?[A-Za-z]){2,9})\.?"
    r"\s*[,:!]?\s+(?=\S)",
    re.IGNORECASE,
)


#: A roster name is also an ordinary English word for half the roster, so the
#: word *after* it decides. "forge ahead", "vision is blurry" and "karen and i
#: talked" are all sentences about something else that happen to start with an
#: operator's name; a following verb, conjunction or preposition means the name
#: is the subject of the sentence rather than the person being spoken to.
_NOT_ADDRESS = re.compile(
    r"^(?:is|was|are|were|has|had|and|or|of|in|on|at|to|for|with|from|"
    r"ahead|said|says|told|thinks?|wants?|looks?|seems?|"
    r"'s|s\b)\b",
    re.IGNORECASE,
)


def addressed(text: str) -> Any | None:
    """The operator this message opens with, or ``None``.

    Returns the roster ``Persona`` so the caller does not have to look it up
    again. FRIDAY is excluded on purpose: she already has her own path, and
    routing her through a webhook would strip the bot identity people know.
    """
    match = _ADDRESS.match(text or "")
    if match is None:
        return None
    spoken = match.group("name").replace(".", "").upper()
    found = _operators().get(spoken)
    if found is None:
        return None
    rest = (text or "")[match.end():].lstrip()
    return None if _NOT_ADDRESS.match(rest) else found


#: What each operator's patch actually sounds like when somebody asks for it.
#: Nobody says "EDITH, perform a security audit" — they say "check my devices",
#: and that landed on FRIDAY, who has no security tools and made a joke instead
#: of handing it to the operator who does.
_DOMAINS: dict[str, tuple[str, ...]] = {
    "EDITH": (
        r"secur(?:e|ity|ing)", r"audit", r"hacked", r"breach", r"vulnerab",
        r"password", r"2fa", r"two.factor", r"lock\s*down", r"lockdown",
        r"malware", r"phish", r"exposed", r"leak(?:ed|ing)?", r"pentest",
        r"harden", r"my\s+devices?",
    ),
    "ORACLE": (
        r"remind", r"schedule", r"calendar", r"alarm", r"every\s+(?:day|week)",
        r"at\s+\d{1,2}\s*(?:am|pm)", r"automat(?:e|ion)", r"cron", r"routine",
    ),
    "GECKO": (
        r"stock", r"market", r"share\s+price", r"invest", r"portfolio",
        r"crypto", r"nifty", r"sensex", r"ticker", r"should\s+i\s+buy",
    ),
    "KAREN": (r"send\s+(?:an?\s+)?(?:email|message|mail)", r"draft\s+an?\s+email",
              r"reply\s+to\s+(?:the\s+)?(?:email|mail)", r"reach\s+out"),
    "VERONICA": (r"caption", r"write\s+(?:me\s+)?a\s+(?:post|tweet|bio)",
                 r"rephrase", r"reword", r"make\s+it\s+sound"),
    "JOCASTA": (r"what\s+did\s+i\s+(?:say|tell\s+you)", r"do\s+you\s+remember",
                r"remember\s+when", r"what\s+do\s+you\s+know\s+about\s+me"),
    "VISION": (r"analys[ei]", r"analyz[ei]", r"compare\s+", r"break\s+(?:this\s+)?down",
               r"deep\s+dive", r"investigate"),
    "FORGE": (r"build\s+(?:me\s+)?an?\s+", r"deploy", r"my\s+(?:app|repo|code|project)",
              r"compile", r"refactor", r"my\s+android\s+app", r"run\s+(?:the\s+)?command"),
}

_DOMAIN_RE = {
    name: re.compile("|".join(f"(?:{p})" for p in patterns), re.IGNORECASE)
    for name, patterns in _DOMAINS.items()
}


def for_domain(text: str) -> Any | None:
    """The operator whose patch this request falls on, or ``None``.

    Only consulted when nobody was named. Asking FRIDAY for something is not
    wrong — she is the one being spoken to — but a request she has no tools for
    should reach whoever does, rather than being deflected with a joke.
    """
    body = (text or "").strip()
    if len(body) < 6:
        return None
    known = _operators()
    for name, pattern in _DOMAIN_RE.items():
        if name in known and pattern.search(body):
            return known[name]
    return None


def handoff_line(operator: Any) -> str:
    """What FRIDAY says as she passes it over."""
    name = str(getattr(operator, "name", "") or "")
    title = str(getattr(operator, "title", "") or "").lower()
    return f"that's {name}'s patch — {title}. calling them in 👇"


def persona_rule(operator: Any) -> str:
    """The instruction that makes this reply sound like the operator.

    Appended last, after the conversation and after FRIDAY's own rules, because
    whatever comes last anchors hardest — the same reason the charter sits
    after the history rather than before it.
    """
    name = str(getattr(operator, "name", "") or "")
    flavour = _FLAVOUR.get(name.upper(), "")
    return (
        f"\n\n[You are {name}, not FRIDAY. You are one of FRIDAY's operators, "
        f"and your speciality is {flavour} You share FRIDAY's memory and know "
        f"everything she knows about these people — never ask them to "
        f"re-introduce themselves. Answer as {name} in your own voice: do not "
        f"sign off as FRIDAY, do not narrate that you are an operator, and do "
        f"not open with your name or your job unless asked. Same house rules as FRIDAY: "
        f"short, human, no corporate tone, no lecturing. If the question is "
        f"outside your speciality, still answer it — you are a person in a "
        f"chat, not a help desk that forwards tickets.\n\n"
        f"If you are asked to do something you have no connected system for — a "
        f"device that was never linked, an account you cannot see — say exactly "
        f"that in one line, name what it would need to work, and offer to set it "
        f"up. Do not deflect with a joke about what you are not, and do not "
        f"imply you did something you did not do. 'I can't check that yet, "
        f"nothing's linked — want me to walk you through connecting it?' is the "
        f"answer; a quip about not being a hacker is not.]"
    )


#: EDITH asked about security does the scan rather than describing one. Wired
#: only for her: the roster's other patches have no machine-side job behind them
#: yet, and an operator that says it checked when it did not is worse than one
#: that says it cannot.
async def machine_report(app: Any, text: str) -> str | None:
    """A real scan of a real machine, or ``None`` to answer the ordinary way."""
    from friday.link.protocol import JobKind  # noqa: PLC0415
    from friday.link.registry import registry_of  # noqa: PLC0415

    registry = registry_of(app)
    result = await registry.run(JobKind.SECURITY_SCAN, timeout=45.0)
    if not result.ok:
        # The honest failure: say which machine, and what would fix it.
        return f"[The scan could not run. Tell the owner exactly this: {result.error}]"

    scan = result.data or {}
    findings = scan.get("findings") or []
    ran = ", ".join(scan.get("checks_run") or []) or "nothing"
    if not findings:
        return (
            f"[A real scan of {scan.get('host', 'the machine')} just ran and found "
            f"nothing. Checks run: {ran}. Say so plainly — this is a genuine "
            f"result, not a guess.]"
        )
    lines = [
        f"- [{f.get('severity', '?')}] {f.get('title', '')} — {f.get('detail', '')} "
        f"(evidence: {f.get('evidence', '')[:120]})"
        for f in findings[:12]
    ]
    body = "\n".join(lines)
    return (
        f"[A real security scan of {scan.get('host', 'the machine')} just ran. "
        f"Checks: {ran}. These are its actual findings — report them as fact, "
        f"worst first, in your own voice, and say what to do about the serious "
        f"ones. Do not invent findings and do not soften these.\n{body}]"
    )


async def github_report(settings: Any, text: str) -> str | None:
    """Build status or dependency alerts, from the real repository."""
    from friday.link import github  # noqa: PLC0415

    repo = _named_repo(text) or str(getattr(settings, "github_default_repo", "") or "")
    if not repo:
        return None

    if re.search(r"\b(?:vulnerab|secur|safe|alert|dependenc|cve)\w*", text, re.I):
        found = await github.security_alerts(settings, repo)
        if not found.get("ok"):
            return f"[GitHub check failed. Say exactly this: {found.get('error')}]"
        if not found.get("scanning_enabled"):
            return (
                f"[Dependency scanning is OFF for {repo}, so nothing has ever been "
                f"checked. Say that plainly — an empty alert list here means nobody "
                f"is looking, not that the code is clean. {found.get('note', '')}]"
            )
        alerts = found.get("alerts") or []
        if not alerts:
            return f"[{repo} has scanning on and no open dependency alerts. Real result.]"
        lines = "\n".join(
            f"- [{a['severity']}] {a['package']}: {a['summary']}" for a in alerts[:10]
        )
        return (
            f"[Real open dependency alerts for {repo}. Report them worst first and "
            f"say what to update. Do not invent any.\n{lines}]"
        )

    status = await github.build_status(settings, repo)
    if not status.get("ok"):
        return f"[GitHub check failed. Say exactly this: {status.get('error')}]"
    runs = status.get("runs") or []
    if not runs:
        return f"[{repo} has no workflow runs yet. Real result.]"
    lines = "\n".join(
        f"- {r['name']} on {r['branch']}: {r['conclusion']} ({r['when']}) — "
        f"{r['message']}" for r in runs[:5]
    )
    return (
        f"[Real recent CI runs for {repo}. Report the outcome plainly; if something "
        f"failed, say which and on what branch.\n{lines}]"
    )


#: "check owner/repo", "is friday-backend safe". A full owner/name wins; a bare
#: name is left alone, because guessing which of forty repositories was meant is
#: worse than using the configured default.
_REPO = re.compile(r"\b([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)\b")


def _named_repo(text: str) -> str:
    match = _REPO.search(text or "")
    return match.group(1) if match else ""


#: A question about the diary rather than about reminders. ORACLE owns both and
#: they are answered from different places — one from Google, one from the
#: reminder store — so the split has to happen before either is consulted.
_CALENDAR = re.compile(
    r"\b(?:calendar|schedule[ds]?|diary|agenda|meeting|appointment"
    r"|what(?:'s|\s+is)\s+on\b|what\s+(?:do|have)\s+i\s+have\b"
    r"|free\s+(?:time|slot)|busy)\b",
    re.IGNORECASE,
)


def wants_calendar(text: str) -> bool:
    """Whether this asks what is scheduled, rather than to be reminded."""
    return bool(_CALENDAR.search(text or ""))


def mentions_repo(text: str) -> bool:
    """Whether a question names a repository, making it about code not a machine."""
    return bool(_REPO.search(text or "")) or bool(
        re.search(r"\b(?:repo|repository|my\s+app|android\s+app|codebase|"
                  r"dependenc|package|build|ci\b)", text or "", re.IGNORECASE)
    )


async def calendar_report(app: Any, text: str) -> str | None:
    """What is actually on the calendar, for ORACLE."""
    from datetime import UTC, datetime, timedelta  # noqa: PLC0415

    from friday.integrations.calendar import (  # noqa: PLC0415
        CalendarError,
        GoogleCalendarClient,
        whats_my_day,
    )

    settings = getattr(app.state, "settings", None)
    secret = getattr(settings, "google_oauth_token", None)
    token = secret.get_secret_value() if secret is not None else ""
    if not token:
        return (
            "[No Google account is connected, so there is no calendar to read. Say "
            "exactly that and offer to walk through connecting one — do not guess "
            "at what might be scheduled.]"
        )

    day = datetime.now(UTC).date()
    if re.search(r"\btomorrow\b", text, re.I):
        day = day + timedelta(days=1)

    import httpx  # noqa: PLC0415

    try:
        async with httpx.AsyncClient(timeout=15.0) as http:
            client = GoogleCalendarClient(token, http=http)
            events = await whats_my_day(client, day=day.isoformat())
    except (CalendarError, Exception) as exc:  # noqa: BLE001 - report, never crash
        return f"[The calendar could not be read. Say exactly this: {exc}]"

    if not events:
        return f"[Nothing is scheduled for {day.isoformat()}. Real result, say so.]"
    lines = []
    for event in events[:10]:
        start = (event.get("start") or {})
        when = start.get("dateTime") or start.get("date") or "?"
        lines.append(f"- {when}: {event.get('summary', '(no title)')}")
    body = "\n".join(lines)
    return (
        f"[The real calendar for {day.isoformat()}. Report it in your own voice, "
        f"in time order. Do not invent anything that is not listed.\n{body}]"
    )


async def speak(
    token: str, channel: str, operator: Any, text: str, avatar: str = ""
) -> str:
    """Post ``text`` into the channel under the operator's own name.

    Returns the message id, or ``""`` on failure. The caller treats an empty
    return as "say it as FRIDAY instead", so a channel where the bot lacks
    **Manage Webhooks** degrades to a normal reply rather than silence.
    """
    hook = await _hook_for(token, channel)
    if hook is None:
        return ""
    hook_id, hook_token = hook
    # Operators get the same treatment as FRIDAY: a long answer is split across
    # messages rather than losing its ending to the 2000-character limit.
    from friday.discord.gateway import split_for_discord  # noqa: PLC0415

    pieces = split_for_discord(text)
    if len(pieces) > 1:
        sent = ""
        for piece in pieces:
            sent = await speak(token, channel, operator, piece, avatar)
            if not sent:
                return ""
        return sent
    text = pieces[0] if pieces else text
    # Discord fetches the avatar itself, so this has to be a public URL. With
    # no public base it stays empty and the operator posts under its name with
    # the default face — plainer, not broken.
    if not avatar:
        avatar = emblem.avatar_url(emblem.public_base(), getattr(operator, "name", ""))
    payload: dict[str, Any] = {
        "content": text[:1900],
        "username": str(getattr(operator, "name", "") or "OPERATOR")[:80],
        # Nobody in a chat should be able to make an operator ping @everyone by
        # asking it to; a webhook post would honour that by default.
        "allowed_mentions": {"parse": ["users"]},
    }
    if avatar:
        payload["avatar_url"] = avatar
    ok, data = await _call(
        "POST", f"/webhooks/{hook_id}/{hook_token}?wait=true", token, payload,
        authed=False,
    )
    return str((data or {}).get("id") or "") if ok else ""


#: ``{channel_id: (webhook_id, webhook_token)}``. Discord charges a round trip
#: for the lookup and the answer never changes, so it is worth holding.
_CACHE: dict[str, tuple[str, str]] = {}


async def _hook_for(token: str, channel: str) -> tuple[str, str] | None:
    """Find or create this channel's roster webhook."""
    cached = _CACHE.get(channel)
    if cached is not None:
        return cached

    ok, existing = await _call("GET", f"/channels/{channel}/webhooks", token, None)
    if ok and isinstance(existing, list):
        for hook in existing:
            if hook.get("name") == _HOOK_NAME and hook.get("token"):
                found = (str(hook.get("id")), str(hook.get("token")))
                _CACHE[channel] = found
                return found

    ok, made = await _call(
        "POST", f"/channels/{channel}/webhooks", token, {"name": _HOOK_NAME}
    )
    if not ok or not (made or {}).get("token"):
        # Almost always a missing **Manage Webhooks** permission. Logged once
        # with the channel so it is fixable rather than mysterious.
        logger.warning(
            "operators: no webhook in channel %s — needs Manage Webhooks", channel
        )
        return None
    created = (str(made.get("id")), str(made.get("token")))
    _CACHE[channel] = created
    return created


async def _call(
    method: str, path: str, token: str, body: Any, *, authed: bool = True
) -> tuple[bool, Any]:
    """One Discord REST call, off the event loop.

    Executing a webhook is authenticated by the webhook token in the URL and
    must *not* carry the bot's Authorization header, hence ``authed``.
    """
    headers = {"User-Agent": _AGENT}
    if authed:
        headers["Authorization"] = f"Bot {token}"
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(  # noqa: S310
        f"{_API}{path}", data=data, method=method, headers=headers
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
            logger.warning("operators: call failed %s %s", method, path)
            return False, {}

    return await anyio.to_thread.run_sync(_send)
