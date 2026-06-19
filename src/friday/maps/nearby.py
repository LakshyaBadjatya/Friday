"""Find places near the caller's GPS via OpenStreetMap Overpass (keyless).

The Siri shortcut sends the device's exact ``lat``/``lon``; for a "… near me"
query this finds the nearest matching places (tourist attractions, restaurants,
hotels, …) and returns both a short spoken line and a richer shareable text with a
maps link (for the Telegram / share-sheet step). Any failure returns ``None`` so
the request falls back to the assistant.
"""

from __future__ import annotations

import json
import math
import re
import urllib.parse
import urllib.request
from typing import Any

_OVERPASS = "https://overpass-api.de/api/interpreter"
_TIMEOUT = 14
_RADIUS_M = 7000

_NEAR = re.compile(r"\b(near me|nearby|near here|around me|around here|close by|closest)\b")

#: query keyword -> (Overpass tag filter, spoken category label)
_CATEGORIES: tuple[tuple[str, str, str], ...] = (
    ("tourist", '["tourism"~"attraction|museum|viewpoint|gallery|zoo|theme_park"]', "attractions"),
    ("attraction", '["tourism"~"attraction|museum|viewpoint"]', "attractions"),
    ("sightsee", '["tourism"~"attraction|museum|viewpoint"]', "sights"),
    ("museum", '["tourism"="museum"]', "museums"),
    ("restaurant", '["amenity"="restaurant"]', "restaurants"),
    ("food", '["amenity"~"restaurant|fast_food"]', "places to eat"),
    ("eat", '["amenity"~"restaurant|fast_food"]', "places to eat"),
    ("cafe", '["amenity"="cafe"]', "cafes"),
    ("coffee", '["amenity"="cafe"]', "cafes"),
    ("hotel", '["tourism"~"hotel|guest_house"]', "hotels"),
    ("stay", '["tourism"~"hotel|guest_house"]', "hotels"),
    ("atm", '["amenity"="atm"]', "ATMs"),
    ("hospital", '["amenity"="hospital"]', "hospitals"),
    ("pharmacy", '["amenity"="pharmacy"]', "pharmacies"),
    ("petrol", '["amenity"="fuel"]', "petrol pumps"),
    ("fuel", '["amenity"="fuel"]', "fuel stations"),
    ("park", '["leisure"="park"]', "parks"),
    ("temple", '["amenity"="place_of_worship"]', "temples"),
    ("mall", '["shop"="mall"]', "malls"),
    ("market", '["amenity"="marketplace"]', "markets"),
)


def _category(low: str) -> tuple[str, str] | None:
    for keyword, filt, label in _CATEGORIES:
        if keyword in low:
            return filt, label
    if any(w in low for w in ("places", "visit", "see", "explore", "things to do")):
        return _CATEGORIES[0][1], "attractions"
    return None


def _overpass(filt: str, lat: float, lon: float) -> list[dict[str, Any]]:
    query = (
        f"[out:json][timeout:{_TIMEOUT}];"
        f"(node{filt}(around:{_RADIUS_M},{lat},{lon});"
        f"way{filt}(around:{_RADIUS_M},{lat},{lon}););"
        "out center 30;"
    )
    body = urllib.parse.urlencode({"data": query}).encode()
    # Overpass rejects header-less requests with 406; a User-Agent is required.
    req = urllib.request.Request(
        _OVERPASS, data=body, headers={"User-Agent": "FridayAssistant/1.0 (near-me)"}
    )
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:  # noqa: S310
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:  # noqa: BLE001 - degrade to empty
        return []
    return data.get("elements", []) if isinstance(data, dict) else []


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return r * 2 * math.asin(math.sqrt(a))


def nearby_reply(query: str, lat: float, lon: float) -> tuple[str, str] | None:
    """Return ``(spoken, share_text)`` for a "… near me" query, or ``None``."""
    low = query.strip().lower()
    if not _NEAR.search(low):
        return None
    category = _category(low)
    if category is None:
        return None
    filt, label = category
    elements = _overpass(filt, lat, lon)

    named: list[tuple[float, str]] = []
    for el in elements:
        name = (el.get("tags") or {}).get("name")
        if not name:
            continue
        center = el if "lat" in el else el.get("center") or {}
        elat, elon = center.get("lat"), center.get("lon")
        dist = (
            _haversine(lat, lon, float(elat), float(elon))
            if elat is not None and elon is not None
            else 9999.0
        )
        named.append((dist, str(name)))
    if not named:
        return None
    named.sort(key=lambda x: x[0])
    top = [name for _d, name in named[:5]]

    maps = (
        f"https://www.google.com/maps/search/"
        f"{urllib.parse.quote(label)}/@{lat},{lon},14z"
    )
    spoken = f"Top {label} near you: {', '.join(top)}."
    share = (
        f"{label.title()} near you:\n"
        + "\n".join(f"• {n}" for n in top)
        + f"\n\nMap: {maps}"
    )
    return spoken, share
