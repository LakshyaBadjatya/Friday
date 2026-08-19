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
import ipaddress
import json
import re
import socket
import time
import urllib.error
import urllib.request
from typing import Any
from urllib.parse import urlparse

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

#: Asked instead when the picture *is* the question.
#:
#: A photographed exam question described in "one or two plain sentences" comes
#: back as "a handwritten physics problem", which is a true sentence that throws
#: away the entire question: the numbers, the units and the vectors are the
#: problem. Nothing downstream can solve what was summarised away, so a turn
#: that means to work the problem transcribes it instead of describing it.
_TRANSCRIBE_PROMPT = (
    "Transcribe everything written in this image, exactly, line by line.\n"
    "- Keep every number, unit, symbol, subscript and vector notation as written.\n"
    "- Where a question is numbered or lettered, keep the numbering.\n"
    "- If more than one colour or hand appears, put ONE short line at the top "
    "saying which is which — it is often how the question being asked is singled "
    "out from the rest of the page. If it is all one colour, say nothing about "
    "colour at all.\n"
    "- Plain text only: no markdown, no headings, no '---' rules, no bullet "
    "points. This goes straight into a message, not onto a page.\n"
    "- Transcribe only. Do not solve anything, and do not comment on it."
)

#: Output ceiling for one look.
#:
#: Was 300, which truncated: a six-line physics question came back cut off mid-word
#: at "charge to ma", with ``finish_reason=length``, and a truncated question is
#: not a question. It has to cover a whole page of transcription now, and the
#: thinking tokens a 2.5-series model spends are charged against this same
#: ceiling — which is why they are turned off below rather than left to eat it.
_MAX_TOKENS = 1500

#: How long the model gets to look.
#:
#: Measured, not guessed: transcribing one page takes this model around fifty
#: seconds even for a ten-kilobyte image over a fast link. The old thirty-second
#: ceiling therefore expired before nearly every real call, and ``describe``
#: returned None — so she said, correctly by her own rules and uselessly, that
#: she could not see the picture that was right there.
_TIMEOUT_SECONDS = 90


#: A turn where the picture carries the question rather than being the subject.
#:
#: "solve the question in blue pen" is not a request to be told what the photo
#: is of; the photo is the problem statement, and every number on it is needed.
_VERBATIM_RULES = re.compile(
    r"\b(?:solve|answer|work\s+(?:it|this|them)\s+out|read|transcribe"
    r"|what\s+does\s+(?:it|this|that)\s+say|questions?|problems?|sums?"
    r"|exercise|homework|assignment|derive|calculate|prove)\b",
    re.IGNORECASE,
)


def wants_transcription(text: str) -> bool:
    """Whether to read the page out in full rather than say what it is."""
    return bool(_VERBATIM_RULES.search(text or ""))


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
        if url and _allowed_host(url) and (ctype in _LOOKABLE or _looks_like_image(url)):
            # Discord serves the same attachment from two hosts and hands us
            # both. The signed cdn link is the primary; media.discordapp.net is
            # kept as a fallback because losing the picture to one host having
            # a bad minute is a poor reason to tell someone you cannot see it.
            proxy = att.get("proxy_url") or ""
            found.append({
                "url": url,
                "name": att.get("filename") or "image",
                "fallback": proxy if proxy and _allowed_host(proxy) else "",
            })
    for embed in message.get("embeds") or []:
        for key in ("image", "thumbnail", "video"):
            block = embed.get(key) or {}
            url = block.get("proxy_url") or block.get("url") or ""
            if url and _looks_like_image(url):
                found.append({"url": url, "name": embed.get("title") or "gif"})
                break
    return found[:_MAX_IMAGES]


#: Hosts whose images are fetched. An allowlist rather than a blocklist because
#: the URL is not as trustworthy as it looks: ``attachments`` comes from Discord's
#: own CDN, but ``embeds`` does not have to — any bot in the channel can post a
#: rich embed carrying an arbitrary URL. Without this, a ``.png`` on a link-local
#: address would be fetched, inlined, and *described back into the chat*, which
#: turns image reading into a credential-shaped exfiltration channel.
_ALLOWED_HOSTS = (
    "cdn.discordapp.com",
    "media.discordapp.net",
    "images-ext-1.discordapp.net",
    "images-ext-2.discordapp.net",
    "media.tenor.com",
    "c.tenor.com",
    "i.imgur.com",
)


def _allowed_host(url: str) -> bool:
    """Whether the URL points at a known image CDN over http(s)."""
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    if parsed.scheme not in ("http", "https"):
        return False  # never file://, ftp://, gopher://
    host = (parsed.hostname or "").lower().rstrip(".")
    return any(host == allowed or host.endswith("." + allowed)
               for allowed in _ALLOWED_HOSTS)


def _looks_like_image(url: str) -> bool:
    """An image extension *and* a host we are willing to fetch from."""
    if not _allowed_host(url):
        return False
    return url.split("?", 1)[0].lower().endswith(
        (".png", ".jpg", ".jpeg", ".webp", ".gif")
    )


async def fetch_all(images: list[dict[str, str]]) -> list[str]:
    """Download the images once, as ``data:`` URIs.

    Split out from :func:`describe` because the bytes are wanted twice now: once
    to read the page, and again to hand the picture itself to the solver. A
    diagram is not transcribable — a circuit or a free-body sketch is the
    question, and no description of it can be worked from — so the solver gets
    the image rather than a paragraph about it. Downloading it a second time to
    do that would double the latency of the slowest thing in the turn.
    """
    fetched: list[str] = []
    for image in images:
        encoded = await anyio.to_thread.run_sync(_fetch, image["url"])
        if encoded is None and image.get("fallback"):
            logger.info("discord vision: retrying the image from its proxy host")
            encoded = await anyio.to_thread.run_sync(_fetch, image["fallback"])
        if encoded is not None:
            fetched.append(encoded)
    if images and not fetched:
        logger.warning(
            "discord vision: none of the %d image(s) could be downloaded", len(images)
        )
    return fetched


async def describe(
    settings: Any,
    images: list[dict[str, str]],
    *,
    verbatim: bool = False,
    fetched: list[str] | None = None,
) -> str | None:
    """Describe the images, or ``None`` when they cannot be read.

    ``None`` is a real answer and the caller must say so plainly rather than
    guess — describing an image nothing looked at is the exact failure that
    produced "a document from your finance folder".

    With ``verbatim`` the page is transcribed rather than summarised, for the
    turn where the picture carries a problem to be worked.
    """
    secret = getattr(settings, "gemini_api_key", None)
    key = secret.get_secret_value() if secret is not None else ""
    if not key or not images:
        return None

    prompt = _TRANSCRIBE_PROMPT if verbatim else _PROMPT
    parts: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    if fetched is None:
        fetched = await fetch_all(images)
    for encoded in fetched:
        parts.append({"type": "image_url", "image_url": {"url": encoded}})
    if len(parts) == 1:  # nothing downloaded
        return None

    base = str(getattr(settings, "gemini_base_url", "") or "").rstrip("/")
    model = str(getattr(settings, "gemini_model", "gemini-3.5-flash") or "")
    return await anyio.to_thread.run_sync(_ask, base, key, model, parts)


def _fetch(url: str) -> str | None:
    """Download an image and return it as a ``data:`` URI, or ``None``."""
    # Re-checked here even though the caller filtered: this is the function that
    # actually opens a socket, and a guard that lives only at the call site stops
    # protecting the moment someone adds a second caller.
    if not _allowed_host(url) or not _resolves_publicly(url):
        logger.warning("discord vision: refused a non-CDN or private-address URL")
        return None
    try:
        request = urllib.request.Request(  # noqa: S310
            url, headers={"User-Agent": "FRIDAY (https://friday.sukhma.in, 1.0)"}
        )
        # A CDN redirect could otherwise walk the fetch to an internal address,
        # so redirects are refused outright rather than followed and re-checked.
        opener = urllib.request.build_opener(_NoRedirect)
        with opener.open(request, timeout=15) as resp:  # noqa: S310
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
    except urllib.error.HTTPError as exc:
        # The status, and from which host. Without this the only symptom of a
        # failed download is her saying she cannot see the picture — which is
        # also what she says when the model refuses, when the key is missing,
        # and when the file is too big. Four different faults, one sentence.
        logger.warning(
            "discord vision: %s refused the image: HTTP %s",
            urlparse(url).hostname, exc.code,
        )
        return None
    except Exception as exc:  # noqa: BLE001 - an unreadable image is not an error
        logger.warning(
            "discord vision: could not download from %s: %s: %s",
            urlparse(url).hostname, type(exc).__name__, exc,
        )
        return None
    return f"data:{ctype};base64,{base64.b64encode(raw).decode()}"


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse 3xx. A followed redirect is an unvalidated second request."""

    def redirect_request(self, *args: Any, **kwargs: Any) -> None:  # noqa: D102
        return None


def _resolves_publicly(url: str) -> bool:
    """Whether the host resolves only to public addresses.

    The allowlist covers the name; this covers the address behind it. DNS can
    point a permitted-looking host at loopback or a private range, and the
    fetcher should refuse that even if the name passed.
    """
    host = (urlparse(url).hostname or "").rstrip(".")
    if not host:
        return False
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError:
        return False
    for info in infos:
        try:
            addr = ipaddress.ip_address(info[4][0])
        except ValueError:
            return False
        if (addr.is_private or addr.is_loopback or addr.is_link_local
                or addr.is_reserved or addr.is_multicast):
            return False
    return bool(infos)


#: Statuses worth asking again about, and how many times to ask in total.
#:
#: "This model is currently experiencing high demand" is the common one and it
#: is genuinely temporary — it came back on roughly two calls in five while this
#: was being measured. One 503 currently costs the whole picture: ``describe``
#: returns None, the caller refuses to invent a description, and she tells
#: someone she cannot see an image that is sitting in front of her. Asking twice
#: turns most of those back into an answer.
_RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
_ATTEMPTS = 3
#: Waited between attempts. Short: someone is watching the channel.
_BACKOFF_SECONDS = (2.0, 5.0)


def _ask(base: str, key: str, model: str, parts: list[dict[str, Any]]) -> str | None:
    """One vision completion, retried through a busy model.

    Blocking; callers run it in a worker thread.
    """
    plain = False
    for attempt in range(_ATTEMPTS):
        answer, retry = _ask_once(base, key, model, parts, plain=plain)
        if answer is not None:
            return answer
        if not retry:
            if plain:
                return None
            # A rejected request is normally not worth repeating — but the one
            # argument here that a model can refuse is the thinking knob, and
            # losing every image to a model swap is a bad way to find that out.
            # Drop it and ask once more before giving up.
            logger.info("discord vision: retrying without the thinking argument")
            plain = True
            continue
        if attempt < _ATTEMPTS - 1:
            time.sleep(_BACKOFF_SECONDS[min(attempt, len(_BACKOFF_SECONDS) - 1)])
    logger.warning("discord vision: gave up after %d attempts", _ATTEMPTS)
    return None


def _ask_once(
    base: str, key: str, model: str, parts: list[dict[str, Any]],
    *, plain: bool = False,
) -> tuple[str | None, bool]:
    """The answer, and whether a failure is worth another attempt."""
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": parts}],
        "max_tokens": _MAX_TOKENS,
    }
    if not plain:
        # Reading a page is not a reasoning task, and on a 2.5-series model the
        # thinking tokens are charged against ``max_tokens`` — two hundred of
        # them were being spent, and taken out of the budget for the answer, to
        # transcribe six lines of handwriting. Not every model accepts the knob
        # though (3.5-flash-lite rejects it outright), which is what ``plain``
        # is for: the caller drops it and asks again rather than treating a
        # model swap as an unreadable image.
        payload["reasoning_effort"] = "none"
    body = json.dumps(payload).encode()
    request = urllib.request.Request(  # noqa: S310
        f"{base}/chat/completions",
        data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(  # noqa: S310
            request, timeout=_TIMEOUT_SECONDS
        ) as resp:
            payload = json.loads(resp.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as exc:
        # The status and body, not just "failed". Google retired
        # gemini-2.0-flash and every image silently stopped being looked at —
        # she answered "nothing much, boss" to a screenshot for days, and the
        # log said only that something had gone wrong, which is unfindable.
        logger.warning(
            "discord vision: model %s rejected the call: HTTP %s %s",
            model, exc.code, exc.read()[:200].decode("utf-8", errors="replace"),
        )
        # A 400 means the request is wrong and will be wrong again; a 503 means
        # the model is busy. Only one of those is worth repeating.
        return None, exc.code in _RETRY_STATUSES
    except TimeoutError:
        logger.warning("discord vision: model call timed out")
        return None, True
    except Exception as exc:  # noqa: BLE001 - vision is a bonus, never a hard failure
        logger.warning("discord vision: model call failed: %s", exc)
        return None, True
    try:
        text = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return None, False
    # An empty body from a healthy call is not worth asking again for; it is
    # what the model had to say.
    return ((text or "").strip() or None), False
