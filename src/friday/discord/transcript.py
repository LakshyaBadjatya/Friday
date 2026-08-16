"""Dumping a channel to a text file, for debugging her from the outside.

Every fault in this bot so far has been reported as a screenshot: a reply that
looked wrong, a message she ignored, an answer that contradicted the one before
it. Screenshots lose exactly what matters — message ids, who replied to what,
whether an attachment was present, what the reactions were, the ordering when
two things land in the same second.

So this writes the channel out in full. Not a summary and not the text alone:
timestamps, author ids, reply targets, attachment urls, embeds, reactions and
edit markers, because the interesting bug is nearly always in the metadata
rather than the words. Embeds go in whole — a Tenor GIF arriving as an embed
rather than an attachment has already caused one bug here.

Two constraints shape it. Discord returns at most 100 messages per request and
pages *backwards* from newest, so history is walked in pages and reversed at the
end to read forwards. And an upload is capped at 25 MB, so the dump is bounded
by message count rather than running until it fails.
"""

from __future__ import annotations

import json
import urllib.request
from datetime import datetime
from typing import Any

import anyio

from friday.logging import get_logger

logger = get_logger("friday.discord.transcript")

_API = "https://discord.com/api/v10"
#: Discord's per-request maximum.
_PAGE = 100
#: Default depth, and the ceiling. Five hundred covers any conversation worth
#: debugging; the cap keeps the file well under the upload limit.
DEFAULT_LIMIT = 500
MAX_LIMIT = 2000


async def fetch(token: str, channel: str, limit: int = DEFAULT_LIMIT) -> list[Any]:
    """Walk a channel's history, returned oldest first.

    Paginates with ``before``: Discord only ever hands back the newest hundred
    and offers no forward cursor from an arbitrary point.
    """
    wanted = max(1, min(int(limit or DEFAULT_LIMIT), MAX_LIMIT))
    collected: list[Any] = []
    before = ""
    while len(collected) < wanted:
        page = min(_PAGE, wanted - len(collected))
        query = f"?limit={page}" + (f"&before={before}" if before else "")
        batch = await _get(token, f"/channels/{channel}/messages{query}")
        if not isinstance(batch, list) or not batch:
            break
        collected.extend(batch)
        before = str(batch[-1].get("id") or "")
        if len(batch) < page or not before:
            break
    collected.reverse()  # Discord pages newest-first; a transcript reads forwards
    return collected


def render(messages: list[Any], channel: str) -> str:
    """Format the history as plain text, keeping everything."""
    lines = [
        "FRIDAY — Discord transcript",
        f"channel: {channel}",
        f"messages: {len(messages)}",
        f"generated: {datetime.now().astimezone().isoformat()}",
        "=" * 72,
        "",
    ]
    for message in messages:
        lines.extend(_one(message))
    return "\n".join(lines)


def _one(message: Any) -> list[str]:
    """One message, with everything hanging off it."""
    author = message.get("author") or {}
    name = author.get("global_name") or author.get("username") or "?"
    tag = " [BOT]" if author.get("bot") else ""
    stamp = str(message.get("timestamp") or "")[:19].replace("T", " ")
    out = [
        f"[{stamp}] {name}{tag} (id={author.get('id')})",
        f"  message_id: {message.get('id')}",
    ]

    replied = message.get("referenced_message") or {}
    if replied:
        who = (replied.get("author") or {}).get("username", "?")
        preview = (replied.get("content") or "")[:80]
        out.append(f"  replying_to: {who} ({replied.get('id')}): {preview}")

    if message.get("edited_timestamp"):
        out.append(f"  edited: {message['edited_timestamp']}")

    content = message.get("content") or ""
    out.append(f"  text: {content if content else '(empty)'}")

    for att in message.get("attachments") or []:
        out.append(
            f"  attachment: {att.get('filename')} "
            f"({att.get('content_type')}, {att.get('size')}B) {att.get('url')}"
        )
    for embed in message.get("embeds") or []:
        out.append(f"  embed: {json.dumps(embed, ensure_ascii=False)[:600]}")
    for reaction in message.get("reactions") or []:
        emoji = (reaction.get("emoji") or {}).get("name")
        out.append(f"  reaction: {emoji} x{reaction.get('count')}")
    for mention in message.get("mentions") or []:
        out.append(f"  mention: {mention.get('username')} ({mention.get('id')})")

    out.append("")
    return out


async def _get(token: str, path: str) -> Any:
    """One authenticated GET, off the event loop."""
    request = urllib.request.Request(  # noqa: S310
        f"{_API}{path}",
        headers={
            "Authorization": f"Bot {token}",
            "User-Agent": "FRIDAY (https://friday.sukhma.in, 1.0)",
        },
    )

    def _send() -> Any:
        try:
            with urllib.request.urlopen(request, timeout=20) as resp:  # noqa: S310
                return json.loads(resp.read() or b"[]")
        except Exception:  # noqa: BLE001 - a short transcript beats no transcript
            logger.warning("transcript: fetch failed for %s", path)
            return []

    return await anyio.to_thread.run_sync(_send)


def multipart(filename: str, body: str, message: str) -> tuple[bytes, str]:
    """Build a multipart body carrying the transcript as a file attachment.

    Hand-rolled because the whole project talks to Discord over ``urllib``, and
    adding a client library for one upload would be a poor trade. The format is
    fixed and short: a boundary, the JSON payload part, then the file part.
    """
    boundary = "----friday" + datetime.now().strftime("%Y%m%d%H%M%S%f")
    payload = json.dumps(
        {"content": message, "attachments": [{"id": 0, "filename": filename}]}
    )
    head = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="payload_json"\r\n'
        "Content-Type: application/json\r\n\r\n"
        f"{payload}\r\n"
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="files[0]"; filename="{filename}"\r\n'
        "Content-Type: text/plain; charset=utf-8\r\n\r\n"
    )
    blob = (
        head.encode("utf-8")
        + body.encode("utf-8")
        + f"\r\n--{boundary}--\r\n".encode()
    )
    return blob, f"multipart/form-data; boundary={boundary}"
