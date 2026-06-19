"""Outbound-message intelligence for the Telegram bot (keyless, best-effort).

Two jobs, both degrade gracefully (never raise to the caller):

* :func:`smart_share` — turn a raw "send this to Telegram" request into a short,
  self-contained message via the LLM, extracting only the key content (a fact,
  formula, place list, plan, …) instead of pasting a whole transcript. When the
  request is too vague it returns a clarifying question instead of a message.
* :func:`build_digest` — assemble the daily 6 AM brief: a weather forecast
  (current + today's high/low + rain/thunderstorm alerts, from wttr.in) and news
  headlines (Google News RSS). Both sources are keyless HTTP GETs.

Callers: ``friday.api.routes_siri`` — ``siri_telegram`` calls :func:`smart_share`,
the new ``siri_digest`` endpoint calls :func:`build_digest` then ``_send_telegram``.
No persisted schema; the digest's only output is the message string.
"""

from __future__ import annotations

import html
import json
import re
import urllib.request
from typing import Any

_UA = {"User-Agent": "FridayAssistant/1.0 (+digest)"}
_TIMEOUT = 12

#: Google News RSS (India/English). Keyless; titles read as "Headline - Publisher".
_NEWS_RSS = "https://news.google.com/rss?hl=en-IN&gl=IN&ceid=IN:en"

#: Pull each item's <title> (the channel title lacks a preceding <item>, so it is
#: skipped). A plain regex avoids the stdlib XML parser's XXE/billion-laughs risk
#: and adds no dependency — the feed only contains HTML-escaped text in <title>.
_ITEM_TITLE_RE = re.compile(r"<item>.*?<title>(.*?)</title>", re.DOTALL | re.IGNORECASE)


def _get(url: str) -> bytes | None:
    try:
        req = urllib.request.Request(url, headers=_UA)
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:  # noqa: S310
            return resp.read()  # type: ignore[no-any-return]
    except Exception:  # noqa: BLE001 - any failure -> source skipped
        return None


# --------------------------------------------------------------------------- #
# Weather forecast (wttr.in j1 JSON)
# --------------------------------------------------------------------------- #
def _weather_block(lat: str, lon: str) -> str | None:
    """Current conditions + today's high/low + rain/thunder alerts, or ``None``."""
    if not lat or not lon:
        return None
    raw = _get(f"https://wttr.in/{lat},{lon}?format=j1")
    if raw is None:
        return None
    try:
        data = json.loads(raw.decode("utf-8", errors="replace"))
        current = data["current_condition"][0]
        today = data["weather"][0]
    except (ValueError, KeyError, IndexError, TypeError):
        return None

    desc = (current.get("weatherDesc") or [{}])[0].get("value", "").strip()
    temp = current.get("temp_C", "?")
    feels = current.get("FeelsLikeC", temp)
    hi = today.get("maxtempC", "?")
    lo = today.get("mintempC", "?")

    try:
        area = data["nearest_area"][0]["areaName"][0]["value"].strip()
    except (KeyError, IndexError, TypeError):
        area = ""

    head = f"🌤 Weather{f' in {area}' if area else ''}: {desc}, {temp}°C now"
    if str(feels) != str(temp):
        head += f" (feels {feels}°C)"
    head += f". High {hi}°C, low {lo}°C."

    # Scan today's 3-hourly slots for rain / thunder / snow likelihood.
    rain = thunder = snow = 0
    for slot in today.get("hourly", []) or []:
        try:
            rain = max(rain, int(slot.get("chanceofrain", 0)))
            thunder = max(thunder, int(slot.get("chanceofthunder", 0)))
            snow = max(snow, int(slot.get("chanceofsnow", 0)))
        except (ValueError, TypeError):
            continue
    alerts: list[str] = []
    if thunder >= 50:
        alerts.append(f"⛈ Thunderstorm likely (~{thunder}%)")
    if rain >= 50:
        alerts.append(f"🌧 Rain likely (~{rain}%) — carry an umbrella")
    elif rain >= 30:
        alerts.append(f"🌦 Some rain possible (~{rain}%)")
    if snow >= 40:
        alerts.append(f"❄️ Snow possible (~{snow}%)")
    if alerts:
        head += "\n" + "\n".join(alerts)
    return head


# --------------------------------------------------------------------------- #
# News headlines (Google News RSS)
# --------------------------------------------------------------------------- #
def _news_block(limit: int = 5) -> str | None:
    raw = _get(_NEWS_RSS)
    if raw is None:
        return None
    text = raw.decode("utf-8", errors="replace")
    titles: list[str] = []
    for match in _ITEM_TITLE_RE.finditer(text):
        title = html.unescape(match.group(1)).strip()
        if not title:
            continue
        # Google appends " - Publisher"; keep the headline, drop the source.
        headline = title.rsplit(" - ", 1)[0].strip() or title
        titles.append(headline)
        if len(titles) >= limit:
            break
    if not titles:
        return None
    return "📰 Top headlines:\n" + "\n".join(f"• {t}" for t in titles)


async def build_digest(lat: str = "", lon: str = "") -> str:
    """Assemble the morning brief text (weather + news), skipping dead sources."""
    parts: list[str] = ["☀️ Good morning, Boss. Here's your brief."]
    weather = _weather_block(lat, lon)
    if weather:
        parts.append(weather)
    news = _news_block()
    if news:
        parts.append(news)
    if len(parts) == 1:
        parts.append("Couldn't reach weather or news right now — I'll try again later.")
    return "\n\n".join(parts)


# --------------------------------------------------------------------------- #
# Smart Telegram share
# --------------------------------------------------------------------------- #
async def smart_share(llm: Any, text: str) -> tuple[str | None, str | None]:
    """Return ``(message, question)`` for a "send this to Telegram" request.

    Exactly one is non-``None``: a polished short message to send, or a short
    clarifying question to speak back when the request is too vague. Short inputs
    and the no-LLM path are sent verbatim (nothing to summarise).
    """
    clean = (text or "").strip()
    if not clean:
        return None, "What should I send, Boss?"
    if llm is None or len(clean) < 40:
        return clean, None

    from friday.providers.llm import Message  # noqa: PLC0415

    prompt = (
        "Prepare a SHORT message to send to a friend on Telegram, drawn from the "
        "user's request below. Keep ONLY the key content (a fact, formula, list of "
        "places, a plan, or an answer) — never paste a whole conversation or add "
        "preamble. 1-5 lines, clear and self-contained.\n"
        "If the request is too vague to know what to send, reply EXACTLY with "
        "'ASK: <one short question>'.\n\n"
        f"Request:\n{clean}\n\nMessage to send:"
    )
    try:
        resp = await llm.complete([Message(role="user", content=prompt)])
    except Exception:  # noqa: BLE001 - fall back to sending the raw text
        return clean, None
    out = (getattr(resp, "text", "") or "").strip()
    if not out:
        return clean, None
    if out.upper().startswith("ASK:"):
        return None, out[4:].strip() or "What should I send, Boss?"
    return out, None
