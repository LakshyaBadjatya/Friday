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

#: Overpass mirrors, tried in order until one returns results. The main instance
#: rate-limits shared cloud IPs (Render) hard, so the mirrors are the difference
#: between "quick results" and an occasional miss.
_OVERPASS_ENDPOINTS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
)
_TIMEOUT = 14
_RADIUS_M = 7000

_NEAR = re.compile(
    r"\b(near ?by|near me|near here|near my|near the|around me|around here|"
    r"close by|closest|nearest|in the area|walking distance|close to me|"
    r"by me|where can i|where to|find me a?|find a|show me|recommend|"
    r"suggest|good places?|best places?|somewhere to|any good)\b"
)

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


#: loose intent words -> a category keyword (so "I'm hungry" finds food, etc.)
_INTENT_HINTS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("hungry", "eat", "food", "lunch", "dinner", "breakfast", "meal"), "food"),
    (("thirsty", "coffee", "drink"), "cafe"),
    (("sleep", "stay", "night", "room", "lodge"), "hotel"),
    (("cash", "withdraw", "money"), "atm"),
    (("sick", "doctor", "emergency", "clinic"), "hospital"),
    (("medicine", "chemist", "drugstore"), "pharmacy"),
    (("petrol", "diesel", "gas", "fuel", "refuel"), "fuel"),
    (("pray", "worship", "darshan"), "temple"),
    (("shopping", "buy"), "mall"),
)


def _category(low: str) -> tuple[str, str] | None:
    for keyword, filt, label in _CATEGORIES:
        if keyword in low:
            return filt, label
    for words, keyword in _INTENT_HINTS:
        if any(w in low for w in words):
            return _category(keyword)
    if any(
        w in low
        for w in ("places", "visit", "see", "explore", "things to do", "sightsee", "tourist")
    ):
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
    # Try each mirror until one yields results — one rate-limiting (429/504/empty)
    # shouldn't sink the whole search.
    for endpoint in _OVERPASS_ENDPOINTS:
        req = urllib.request.Request(
            endpoint, data=body, headers={"User-Agent": "FridayAssistant/1.0 (near-me)"}
        )
        try:
            with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:  # noqa: S310
                data = json.loads(resp.read().decode("utf-8"))
        except Exception:  # noqa: BLE001 - try the next mirror
            continue
        elements = data.get("elements", []) if isinstance(data, dict) else []
        if elements:
            return elements
    return []


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return r * 2 * math.asin(math.sqrt(a))


def nearby_reply(query: str, lat: float, lon: float) -> tuple[str, str] | None:
    """Return ``(spoken, share_text)`` for a "… near me" query, or ``None``.

    Fast, network-free gate: the phrasing must read like a find-places request
    (``_NEAR``) and name a recognisable category (``_category``). When neither
    fires the caller can still fall back to :func:`classify_nearby` (AI).
    """
    low = query.strip().lower()
    if not _NEAR.search(low):
        return None
    category = _category(low)
    if category is None:
        return None
    filt, label = category
    return nearby_from_filter(filt, label, lat, lon)


#: Labels that mean "sightseeing" — these pull famous places from Wikipedia, not
#: just OSM tags (OSM is sparse and vandalised in many Indian cities).
_ATTRACTION_LABELS = {"attractions", "sights"}
_WIKI_API = "https://en.wikipedia.org/w/api.php"
#: Wikipedia geosearch returns ALL notable articles nearby — drop the ones that
#: aren't tourist attractions (admin areas, transport, institutions).
_WIKI_DENY = re.compile(
    r"constituency|railway station|\bjunction\b|assembly|lok sabha|vidhan|"
    r"\bdistrict\b|tehsil|mandal|\bblock\b|airport|bus stand|bus station|"
    r"universit|\bcollege\b|institute|\bschool\b|hospital|cantonment|\bward\b",
    re.IGNORECASE,
)
#: Positive signal that a notable place is a real tourist attraction (so a famous
#: palace/lake/fort is kept, while a residential colony or village is dropped).
_ATTRACTION_HINT = re.compile(
    r"palace|fort|mandir|temple|masjid|church|gurudwara|lake|sagar|jheel|talab|"
    r"garden|udyaan|bagh|\bpark\b|museum|riverfront|mahal|garh|\bdam\b|barrage|"
    r"zoo|wonder|haveli|fountain|memorial|monument|stadium|tomb|cenotaph|chhatri|"
    r"\bgate\b|minar|mahadev|dham|fall|safari|sanctuary|view\s?point|ghat",
    re.IGNORECASE,
)


def _looks_like_junk(name: str) -> bool:
    """True for OSM vandalism/test names — short tokens with no lowercase letters
    (e.g. ``POI``, ``MKLJFAL``, ``PA 1``, ``P 6``). Real names have lowercase."""
    stripped = name.strip()
    return len(stripped) <= 7 and not any(c.islower() for c in stripped)


def _overpass_named(filt: str, lat: float, lon: float) -> list[tuple[float, str]]:
    """Named OSM POIs for ``filt`` as ``(distance_km, name)``, junk filtered."""
    named: list[tuple[float, str]] = []
    for el in _overpass(filt, lat, lon):
        name = (el.get("tags") or {}).get("name")
        if not name or _looks_like_junk(str(name)):
            continue
        center = el if "lat" in el else el.get("center") or {}
        elat, elon = center.get("lat"), center.get("lon")
        dist = (
            _haversine(lat, lon, float(elat), float(elon))
            if elat is not None and elon is not None
            else 9999.0
        )
        named.append((dist, str(name)))
    return named


def _wikipedia_nearby(lat: float, lon: float) -> list[tuple[float, str]]:
    """Famous attraction-type places near ``lat``/``lon`` via Wikipedia geosearch.

    Notability is the filter the user wants ("big attractions the city is famous
    for") — every result has its own Wikipedia article. Admin/transport pages are
    dropped and a positive attraction-keyword test keeps colonies/villages out.
    """
    url = (
        f"{_WIKI_API}?action=query&list=geosearch&gscoord={lat}%7C{lon}"
        "&gsradius=10000&gslimit=30&format=json"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "FridayAssistant/1.0 (near-me)"})
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:  # noqa: S310
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:  # noqa: BLE001 - degrade to empty
        return []
    results: list[tuple[float, str]] = []
    for geo in (data.get("query", {}) or {}).get("geosearch", []):
        title = str(geo.get("title", "")).strip()
        if not title or _WIKI_DENY.search(title) or not _ATTRACTION_HINT.search(title):
            continue
        try:
            dist_km = float(geo.get("dist", 9999.0)) / 1000.0
        except (TypeError, ValueError):
            dist_km = 9999.0
        results.append((dist_km, title))
    return results


def nearby_from_filter(
    filt: str, label: str, lat: float, lon: float
) -> tuple[str, str] | None:
    """Format the nearby reply, merging sources by distance.

    Shared back half of the heuristic (:func:`nearby_reply`) and AI
    (:func:`classify_nearby`) paths. For sightseeing it leads with Wikipedia's
    famous places, then OSM; functional categories (food, fuel, …) use OSM only.
    """
    candidates: list[tuple[float, str]] = []
    if label in _ATTRACTION_LABELS:
        candidates.extend(_wikipedia_nearby(lat, lon))
    candidates.extend(_overpass_named(filt, lat, lon))
    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0])
    top: list[str] = []
    seen: set[str] = set()
    for _dist, name in candidates:
        key = name.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        top.append(name)
        if len(top) >= 6:
            break

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


#: AI category key -> (Overpass filter, spoken label). The model is constrained
#: to these keys so the result is always a valid Overpass query.
_FILTER_BY_KEY: dict[str, tuple[str, str]] = {
    "tourist": (
        '["tourism"~"attraction|museum|viewpoint|gallery|zoo|theme_park"]',
        "attractions",
    ),
    "restaurant": ('["amenity"~"restaurant|fast_food"]', "places to eat"),
    "cafe": ('["amenity"="cafe"]', "cafes"),
    "hotel": ('["tourism"~"hotel|guest_house"]', "hotels"),
    "atm": ('["amenity"="atm"]', "ATMs"),
    "bank": ('["amenity"="bank"]', "banks"),
    "hospital": ('["amenity"~"hospital|clinic|doctors"]', "hospitals"),
    "pharmacy": ('["amenity"="pharmacy"]', "pharmacies"),
    "fuel": ('["amenity"="fuel"]', "petrol pumps"),
    "park": ('["leisure"="park"]', "parks"),
    "temple": ('["amenity"="place_of_worship"]', "places of worship"),
    "mall": ('["shop"="mall"]', "malls"),
    "market": ('["amenity"="marketplace"]', "markets"),
    "shop": ('["shop"]', "shops"),
    "bar": ('["amenity"~"bar|pub"]', "bars"),
}


def _extract_json(text: str) -> dict[str, Any] | None:
    """Pull the first ``{...}`` object out of a model reply (tolerates fences)."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        obj = json.loads(text[start : end + 1])
    except ValueError:
        return None
    return obj if isinstance(obj, dict) else None


async def classify_nearby(llm: Any, query: str) -> tuple[str, str] | None:
    """AI fallback: decide if ``query`` seeks nearby places and infer a category.

    Returns ``(overpass_filter, spoken_label)`` to hand to
    :func:`nearby_from_filter`, or ``None`` when it is not a find-places request
    (or the model/parse fails — the caller then falls back to the assistant).
    """
    if llm is None:
        return None
    from friday.providers.llm import Message  # noqa: PLC0415

    keys = ", ".join(_FILTER_BY_KEY)
    prompt = (
        "Decide if the user wants to FIND PLACES NEAR THEIR CURRENT LOCATION "
        "(food, attractions, hotels, ATMs, fuel, hospitals, shops, etc.).\n"
        f'User said: "{query}"\n'
        "Reply with ONLY a compact JSON object and nothing else:\n"
        '{"nearby": true|false, "category": "<KEY>"}\n'
        f"where <KEY> is exactly one of: {keys}. "
        'If it is not a find-nearby-places request, reply {"nearby": false}.'
    )
    try:
        resp = await llm.complete([Message(role="user", content=prompt)])
    except Exception:  # noqa: BLE001 - degrade to no-match
        return None
    data = _extract_json((getattr(resp, "text", "") or "").strip())
    if not data or not data.get("nearby"):
        return None
    key = str(data.get("category", "")).strip().lower()
    return _FILTER_BY_KEY.get(key)
