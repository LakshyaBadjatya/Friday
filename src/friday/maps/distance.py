"""Real distance + driving time between two places — computed, not guessed.

The model used to hallucinate distances ("Kota to Bombay ≈ 1,100 km / 18 h" when
it's ~800 km / ~10 h). This grounds the answer: geocode each place with OpenStreetMap
Nominatim, then ask OSRM for the actual driving route. Keyless and failure-tolerant —
any miss returns ``None`` so the caller falls back to the assistant.
"""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from typing import Any

_NOMINATIM = "https://nominatim.openstreetmap.org/search"
_OSRM = "https://router.project-osrm.org/route/v1/driving"
_UA = "FridayAssistant/1.0 (personal distance lookup)"
_TIMEOUT = 8

# "distance between A and B", "distance from A to B", "distance A to B",
# "how far is A from B", "how far A to B".
_PATTERNS = (
    re.compile(r"\bdistance\s+(?:between|from)\s+(.+?)\s+(?:and|to)\s+(.+?)$"),
    re.compile(r"\bdistance\s+(?:of\s+|for\s+)?(.+?)\s+to\s+(.+?)$"),
    re.compile(r"\bhow far\s+(?:is\s+|away\s+)?(.+?)\s+(?:from|to)\s+(.+?)$"),
)


def _get(url: str) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:  # noqa: S310
            return json.loads(resp.read().decode("utf-8"))
    except Exception:  # noqa: BLE001 - degrade to None
        return None


def _geocode(place: str) -> tuple[float, float] | None:
    query = urllib.parse.urlencode({"q": place, "format": "json", "limit": 1})
    data = _get(f"{_NOMINATIM}?{query}")
    if isinstance(data, list) and data:
        try:
            return float(data[0]["lat"]), float(data[0]["lon"])
        except (KeyError, ValueError, TypeError):
            return None
    return None


def _route(
    a: tuple[float, float], b: tuple[float, float]
) -> tuple[float, float, str] | None:
    # OSRM wants lon,lat order. ``steps=true`` populates each leg's ``summary``
    # (the major roads, e.g. "NH 52, NH 48") — our "via".
    url = f"{_OSRM}/{a[1]},{a[0]};{b[1]},{b[0]}?overview=false&steps=true"
    data = _get(url)
    if isinstance(data, dict):
        routes = data.get("routes") or []
        if routes:
            dist = routes[0].get("distance")
            dur = routes[0].get("duration")
            legs = routes[0].get("legs") or []
            summary = str(legs[0].get("summary", "")) if legs else ""
            if isinstance(dist, int | float):
                return float(dist), float(dur or 0), summary
    return None


def _places(text: str) -> tuple[str, str] | None:
    low = text.strip().lower().rstrip(".!?")
    for pattern in _PATTERNS:
        m = pattern.search(low)
        if m:
            a = m.group(1).strip(" ?.,")
            b = m.group(2).strip(" ?.,")
            if a and b:
                return a, b
    return None


def distance_reply(text: str) -> str | None:
    """A grounded spoken distance answer, or ``None`` if it isn't a distance query
    (or a place couldn't be located / routed)."""
    places = _places(text)
    if places is None:
        return None
    a_name, b_name = places
    a = _geocode(a_name)
    b = _geocode(b_name)
    if a is None or b is None:
        return None
    routed = _route(a, b)
    if routed is None:
        return None
    meters, seconds, summary = routed
    if not meters:
        return None
    km = round(meters / 1000)
    via = f" via {summary}" if summary else ""
    if seconds >= 3600:
        hours = round(seconds / 3600)
        when = f"around {hours} hour{'s' if hours != 1 else ''}"
    else:
        when = f"around {round(seconds / 60)} minutes"
    return (
        f"Fastest from {a_name.title()} to {b_name.title()} is about {km} kilometres"
        f"{via} — total drive time {when}, Boss."
    )
