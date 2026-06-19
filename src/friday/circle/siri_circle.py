"""Firestore-linked Siri circle handler — act on the app's REAL data as the caller.

Given the caller's bearer (a Firebase ID token, or a refresh token exchanged for
one) this resolves the caller's uid, reads/writes the SAME Firestore the web app +
HUD use, and returns a spoken reply for the Siri front door. Every path returns
``None`` on a miss (unknown caller, name not in the circle, parse miss, any error)
so the request falls through to the orchestrator and the live endpoint never breaks.

Covers all of A–D: presence/status, good-time-to-call, set-status/safe-arrival,
reminders, SOS, thinking-of-you nudge, daily question, relay, check-in streak, and
"do I have messages". Messages stay end-to-end encrypted — Siri reports metadata
(counts) and, only for a circle with ``siriReadAloud`` on, reads plaintext voice
notes; it never decrypts the E2EE chat.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta, tzinfo
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from friday.circle.firestore_rest import FirestoreRest, resolve_token
from friday.circle.intents import SetStatus, StatusQuery, parse_intent

_NUDGE = re.compile(
    r"\b(?:send|give)\s+(.+?)\s+a?\s*(?:thinking[- ]of[- ]you|nudge|hug|wave)\b"
)
_NUDGE2 = re.compile(r"\bnudge\s+(.+?)$")
_REMIND = re.compile(r"\bremind\s+(.+?)\s+to\s+(.+?)(?:\s+at\s+([^,]+?))?$")
_SOS = re.compile(r"\b(?:sos|i need help|emergency|help me|panic)\b")
_DAILY_ASK = re.compile(r"\b(?:today'?s question|daily question|question of the day)\b")
_DAILY_ANSWER = re.compile(r"\bmy answer is\s+(.+?)$")
_RELAY = re.compile(
    r"\bask\s+(.+?)\s+((?:what|if|when|where|why|how|whether|would|will|do|does)\b.+?)$"
)
_MESSAGES = re.compile(
    r"\b(?:any (?:new )?messages|do i have (?:any )?messages|unread|new messages)\b"
)
_STREAK = re.compile(r"\b(?:our streak|check[- ]?in streak|how'?s our streak|streak)\b")
_CHECKIN = re.compile(r"\bcheck me in\b")
_TIME = re.compile(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b")
_ME = {"me", "my", "myself", "i"}


def handle(token: str, query: str, now: datetime) -> str | None:
    """Resolve the caller and act on ``query`` against their Firestore; else None."""
    resolved = resolve_token(token)
    if resolved is None:
        return None
    id_token, uid = resolved
    ctx = _Ctx(FirestoreRest(id_token), uid, now)

    text = query.strip()
    low = text.lower().rstrip(".!?")

    if _SOS.search(low):
        return ctx.sos()
    m = _REMIND.search(low)
    if m:
        return ctx.remind(m.group(1), m.group(2), m.group(3))
    m = _NUDGE.search(low) or _NUDGE2.search(low)
    if m:
        return ctx.nudge(m.group(1))
    m = _DAILY_ANSWER.search(low)
    if m:
        return ctx.daily_answer(m.group(1))
    if _DAILY_ASK.search(low):
        return ctx.daily_ask()
    if _MESSAGES.search(low):
        return ctx.messages_summary()
    if _CHECKIN.search(low):
        return ctx.check_in()
    if _STREAK.search(low):
        return ctx.streak()
    m = _RELAY.search(low)
    if m:
        relayed = ctx.relay(m.group(1), m.group(2))
        if relayed is not None:
            return relayed

    intent = parse_intent(text)
    if isinstance(intent, SetStatus):
        return ctx.set_status(intent)
    if isinstance(intent, StatusQuery):
        return ctx.status_query(intent.name)
    return None


def _clean(value: str) -> str:
    return value.strip().strip("\"'").strip()


class _Ctx:
    """One caller's working context over their Firestore circle."""

    def __init__(self, fs: FirestoreRest, uid: str, now: datetime) -> None:
        self._fs = fs
        self._uid = uid
        self._now = now
        self._groups_cache: list[str] | None = None

    # -- lookups ----------------------------------------------------------- #
    def _group_ids(self) -> list[str]:
        if self._groups_cache is None:
            rows = self._fs.list(f"users/{self._uid}/memberships")
            self._groups_cache = [
                str(r["groupId"]) for r in rows if r.get("groupId")
            ]
        return self._groups_cache

    def _resolve(self, name: str) -> tuple[str, dict[str, Any]] | None:
        """Find (group_id, member fields) for a spoken name in the caller's circle."""
        key = _clean(name).lower()
        for gid in self._group_ids():
            members = self._fs.list(f"groups/{gid}/members")
            for member in members:
                if key in _ME and member.get("uid") == self._uid:
                    return gid, member
                display = str(member.get("displayName", "")).strip().lower()
                if display and (display == key or key in display.split()):
                    return gid, member
        return None

    @staticmethod
    def _first_name(member: dict[str, Any]) -> str:
        name = str(member.get("displayName", "they")).strip()
        return name.split()[0] if name else "they"

    @staticmethod
    def _local(member: dict[str, Any], now: datetime) -> tuple[str, bool] | None:
        tz = str(member.get("tz") or "").strip()
        if not tz:
            return None
        try:
            local = now.astimezone(ZoneInfo(tz))
        except (ZoneInfoNotFoundError, ValueError):
            return None
        label = local.strftime("%-I:%M %p").lstrip("0")
        asleep = local.hour >= 23 or local.hour < 7
        return label, asleep

    # -- A: presence / status --------------------------------------------- #
    def status_query(self, name: str) -> str | None:
        found = self._resolve(name)
        if found is None:
            return None
        _gid, member = found
        who = self._first_name(member)
        bits: list[str] = []
        local = self._local(member, self._now)
        if local is not None:
            label, asleep = local
            bits.append(
                f"it's {label} for {who} — likely asleep" if asleep
                else f"it's {label} for {who}"
            )
        presence = str(member.get("presence") or "")
        if presence == "active":
            bits.append("active right now")
        elif presence == "away":
            bits.append("away at the moment")
        for field, label in (("text", ""), ("mood", "feeling "), ("place", "at ")):
            val = member.get(field)
            if val:
                bits.append(f"{label}{val}")
        if member.get("arrived_safe"):
            bits.append("home safe")
        if not bits:
            return f"I don't have anything on {who} yet, Boss."
        return _cap(", ".join(bits)) + "."

    def set_status(self, intent: SetStatus) -> str | None:
        fields: dict[str, Any] = {}
        if intent.text is not None:
            fields["text"] = intent.text
        if intent.mood is not None:
            fields["mood"] = intent.mood
        if intent.place is not None:
            fields["place"] = intent.place
        if intent.arrived_safe is not None:
            fields["arrived_safe"] = intent.arrived_safe
        if not fields:
            return None
        if not self._patch_self(fields):
            return None
        if intent.arrived_safe:
            return "Glad you made it home safe — I've let your circle know."
        if intent.mood:
            return f"Got it — feeling {intent.mood}."
        if intent.place:
            return f"Got it — you're at {intent.place}."
        return f"Done — status set to {intent.text}."

    def _patch_self(self, fields: dict[str, Any]) -> bool:
        ok = False
        for gid in self._group_ids():
            if self._fs.patch(f"groups/{gid}/members/{self._uid}", fields):
                ok = True
        return ok

    # -- B: care & safety -------------------------------------------------- #
    def sos(self) -> str | None:
        gids = self._group_ids()
        if not gids:
            return None
        stamped = self._now.isoformat()
        sent = False
        for gid in gids:
            if self._fs.create(
                f"groups/{gid}/alerts",
                {"fromUid": self._uid, "kind": "sos", "createdAt": stamped},
            ):
                sent = True
            self._fs.patch(
                f"groups/{gid}/members/{self._uid}", {"sos": True, "sosAt": stamped}
            )
        return "SOS sent to your circle — hang in there, Boss." if sent else None

    def remind(self, name: str, task: str, when: str | None) -> str | None:
        found = self._resolve(name)
        if found is None:
            return None
        gid, member = found
        due, due_label = self._due(when, member)
        fields: dict[str, Any] = {
            "fromUid": self._uid,
            "toUid": str(member.get("uid", "")),
            "text": _clean(task),
            "createdAt": self._now.isoformat(),
        }
        if due is not None:
            fields["dueAt"] = due
        if not self._fs.create(f"groups/{gid}/reminders", fields):
            return None
        who = self._first_name(member)
        tail = f" at {due_label}" if due_label else ""
        return f"Done — I'll remind {who} to {_clean(task)}{tail}."

    def _due(self, when: str | None, member: dict[str, Any]) -> tuple[str | None, str]:
        if not when:
            return None, ""
        m = _TIME.search(when.lower())
        if not m:
            return None, _clean(when)
        hour = int(m.group(1)) % 12
        minute = int(m.group(2) or 0)
        if (m.group(3) or "") == "pm":
            hour += 12
        tz_name = str(member.get("tz") or "UTC")
        tz: tzinfo
        try:
            tz = ZoneInfo(tz_name)
        except (ZoneInfoNotFoundError, ValueError):
            tz = UTC
        local_now = self._now.astimezone(tz)
        target = local_now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if target <= local_now:
            target += timedelta(days=1)
        label = target.strftime("%-I:%M %p").lstrip("0") + " their time"
        return target.astimezone(UTC).isoformat(), label

    # -- C: connection & fun ---------------------------------------------- #
    def nudge(self, name: str) -> str | None:
        found = self._resolve(name)
        if found is None:
            return None
        gid, member = found
        if not self._fs.create(
            f"groups/{gid}/nudges",
            {
                "fromUid": self._uid,
                "toUid": str(member.get("uid", "")),
                "kind": "thinking-of-you",
                "createdAt": self._now.isoformat(),
            },
        ):
            return None
        return f"Sent — {self._first_name(member)} will know you're thinking of them. 💭"

    def daily_ask(self) -> str | None:
        for gid in self._group_ids():
            group = self._fs.get(f"groups/{gid}")
            if group and group.get("dailyQuestion"):
                return f"Today's question: {group['dailyQuestion']}"
        return "No question set for today yet, Boss."

    def daily_answer(self, answer: str) -> str | None:
        if not self._patch_self({"dailyAnswer": _clean(answer)}):
            return None
        return "Saved your answer for today."

    def relay(self, name: str, question: str) -> str | None:
        found = self._resolve(name)
        if found is None:
            return None
        gid, member = found
        if not self._fs.create(
            f"groups/{gid}/nudges",
            {
                "fromUid": self._uid,
                "toUid": str(member.get("uid", "")),
                "kind": "relay",
                "text": _clean(question),
                "createdAt": self._now.isoformat(),
            },
        ):
            return None
        return f"I'll pass that to {self._first_name(member)} and read you the reply."

    def messages_summary(self) -> str | None:
        total = 0
        readable: list[str] = []
        for gid in self._group_ids():
            msgs = self._fs.list(f"groups/{gid}/messages")
            nudges = self._fs.list(f"groups/{gid}/nudges")
            total += len(msgs) + len(nudges)
            group = self._fs.get(f"groups/{gid}")
            if group and group.get("siriReadAloud"):
                for note in self._fs.list(f"groups/{gid}/voicenotes"):
                    if note.get("text"):
                        readable.append(str(note["text"]))
        if total == 0 and not readable:
            return "No new messages, Boss."
        head = f"You have {total} message{'s' if total != 1 else ''} in your circle."
        if readable:
            head += " " + " ".join(readable[-3:])
        else:
            head += " Open the app to read them."
        return head

    # -- check-in streak --------------------------------------------------- #
    def check_in(self) -> str | None:
        streak = self._bump_streak()
        if streak is None:
            return None
        return f"Checked in — you're on a {streak}-day streak. 🔥"

    def streak(self) -> str | None:
        member = self._self_member()
        if member is None:
            return None
        count = int(member.get("streak") or 0)
        return f"Your check-in streak is {count} day{'s' if count != 1 else ''}."

    def _self_member(self) -> dict[str, Any] | None:
        for gid in self._group_ids():
            member = self._fs.get(f"groups/{gid}/members/{self._uid}")
            if member is not None:
                return member
        return None

    def _bump_streak(self) -> int | None:
        member = self._self_member()
        if member is None:
            return None
        today = self._now.date().isoformat()
        last = str(member.get("lastCheckin") or "")
        streak = int(member.get("streak") or 0)
        if last == today:
            return streak
        yesterday = (self._now.date() - timedelta(days=1)).isoformat()
        streak = streak + 1 if last == yesterday else 1
        if not self._patch_self({"streak": streak, "lastCheckin": today}):
            return None
        return streak


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:] if text else text
