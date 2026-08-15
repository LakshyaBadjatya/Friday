"""Looking at what gets posted: images, screenshots, memes and GIFs.

Telegram gets an honest "I can't see photos". Discord is where things actually
get posted, so here she looks — a meme nobody reacts to is a joke wasted, and an
assistant that has to be told what is in a screenshot is not much of one.

Three practical notes shaped this.

**Images are fetched, not linked.** Discord CDN links carry expiring signatures,
and a provider fetching one later may get a 403, so the bytes are downloaded here
and inlined as base64. It costs a round trip and is the difference between
working and intermittently not.

**GIFs are a still frame.** A vision model reads one image, so an animation is
flattened to its first frame. That is usually enough to get the joke, and where
it is not she says what she can see rather than inventing the motion.

**Size is capped hard.** A phone screenshot is a couple of hundred kilobytes; a
video-length GIF can be tens of megabytes and would blow both the request limit
and the budget. Over the cap she declines to look rather than failing slowly.

The description comes from a *separate* vision call and is then handed to the
normal conversation as context rather than replacing it. She answers as herself,
having seen the picture; the vision model never speaks to the room directly.
"""

from __future__ import annotations

import base64
import json
from typing import Any

import anyio

from friday.logging import get_logger

logger = get_logger("friday.discord.vision")

#: Content types worth looking at. Anything else (video, audio, archives) is
#: skipped rather than guessed at.
_LOOKABLE = ("image/png", "image/jpeg", "image/jpg", "image/webp", "image/gif")
#: Largest attachment fetched, in bytes.
_MAX_BYTES = 8 * 1024 * 1024
#: How many images from one message get described — a ten-image dump is a
#: gallery, not a question, and describing all of them helps nobody.
_MAX_IMAGES = 2

_PROMPT = (
    "Describe this image in one or two plain sentences, for a friend who cannot "
    "see it. If it is a meme, screenshot or reaction image, say what the joke or "
    "the content actually is — quote any text you can read. Do not editorialise, "
    "and do not describe anything you cannot actually make out."
)


def images_in(message: dict[str, Any]) -> list[dict[str, str]]:
    """Every viewable image on a message: real attachments, then embeds.

    Embeds matter as much as attachments: a Tenor GIF — which is most of what
    actually gets posted — arrives as an embed with a thumbnail URL and no
    attachment at all, so reading only ``attachments`` would miss the majority of
    the images in a group chat.
    """
    found: list[dict[str, str]] = []
    for att in message.get("attachments") or []:
        ctype = (att.get("content_type") or "").split(";")[0].strip().lower()
        url = att.get("url") or ""
        if url and (ctype in _LOOKABLE or _looks_like_image(url)):
            found.append({"url": url, "name": att.get("filename") or "image"})
    for embed in message.get("embeds") or []:
        for key in ("image", "thumbnail", "video"):
            block = embed.get(key) or {}
            url = block.get("proxy_url") or block.get("url") or ""
            if url and _looks_like_image(url):
                found.append({"url": url, "name": embed.get("title") or "gif"})
                break
    return found[:_MAX_IMAGES]


def _looks_like_image(url: str) -> bool:
    return url.split("?", 1)[0].lower().endswith(
        (".png", ".jpg", ".jpeg", ".webp", ".gif")
    )


async def describe(settings: Any, images: list[dict[str, str]]) -> str | None:
    """Describe the images, or ``None`` when they cannot be read.

    ``None`` is a real answer and the caller must say so plainly rather than
    guess — describing an image nothing looked at is the exact failure that
    produced "a document from your finance folder".
    """
    secret = getattr(settings, "gemini_api_key", None)
    key = secret.get_secret_value() if secret is not None else ""
    if not key or not images:
        return None

    parts: list[dict[str, Any]] = [{"type": "text", "text": _PROMPT}]
    for image in images:
        encoded = await anyio.to_thread.run_sync(_fetch, image["url"])
        if encoded is not None:
            parts.append({"type": "image_url", "image_url": {"url": encoded}})
    if len(parts) == 1:  # nothing downloaded
        return None

    base = str(getattr(settings, "gemini_base_url", "") or "").rstrip("/")
    model = str(getattr(settings, "gemini_model", "gemini-2.0-flash") or "")
    return await anyio.to_thread.run_sync(_ask, base, key, model, parts)


def _fetch(url: str) -> str | None:
    """Download an image and return it as a ``data:`` URI, or ``None``."""
    import urllib.request  # noqa: PLC0415

    try:
        request = urllib.request.Request(  # noqa: S310
            url, headers={"User-Agent": "FRIDAY (https://friday.sukhma.in, 1.0)"}
        )
        with urllib.request.urlopen(request, timeout=15) as resp:  # noqa: S310
            declared = int(resp.headers.get("Content-Length") or 0)
            if declared > _MAX_BYTES:
                logger.info("discord vision: attachment too large (%d bytes)", declared)
                return None
            # One byte past the cap, so a missing or lying Content-Length is still
            # caught rather than pulling an unbounded body into memory.
            raw = resp.read(_MAX_BYTES + 1)
            if len(raw) > _MAX_BYTES:
                logger.info("discord vision: attachment exceeded the cap while reading")
                return None
            ctype = (resp.headers.get("Content-Type") or "image/png").split(";")[0]
    except Exception:  # noqa: BLE001 - an unreadable image is not an error
        logger.warning("discord vision: fetch failed")
        return None
    return f"data:{ctype};base64,{base64.b64encode(raw).decode()}"


def _ask(base: str, key: str, model: str, parts: list[dict[str, Any]]) -> str | None:
    """One vision completion. Blocking; callers run it in a worker thread."""
    import urllib.request  # noqa: PLC0415

    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": parts}],
            "max_tokens": 300,
        }
    ).encode()
    request = urllib.request.Request(  # noqa: S310
        f"{base}/chat/completions",
        data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as resp:  # noqa: S310
            payload = json.loads(resp.read().decode("utf-8", errors="replace"))
    except Exception:  # noqa: BLE001 - vision is a bonus, never a hard failure
        logger.warning("discord vision: model call failed")
        return None
    try:
        text = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return None
    return (text or "").strip() or None
