"""Real distance + driving time between two places — computed, not guessed.

The model used to hallucinate distances ("Kota to Bombay ≈ 1,100 km / 18 h" when
it's ~800 km / ~10 h). This grounds the answer: geocode each place with OpenStreetMap
Nominatim, then ask OSRM for the actual driving route. Keyless and failure-tolerant —
any miss returns ``None`` so the caller falls back to the assistant.
"""

from __future__ import annotations

import concurrent.futures
import json
import re
import urllib.parse
import urllib.request
from typing import Any

_NOMINATIM = "https://nominatim.openstreetmap.org/search"
_NOM_REVERSE = "https://nominatim.openstreetmap.org/reverse"
# Photon (komoot) is OSM-based, keyless, and far more tolerant of cloud-host IPs
# than Nominatim (which rate-limits/blocks Render). Used as a forward-geocode
# fallback AND for via-city reverse lookups so both succeed in production.
_PHOTON = "https://photon.komoot.io/reverse"
_PHOTON_SEARCH = "https://photon.komoot.io/api"
_OSRM = "https://router.project-osrm.org/route/v1/driving"
_UA = "FridayAssistant/1.0 (personal distance lookup)"
_TIMEOUT = 6
_REVERSE_TIMEOUT = 4

#: In-process geocode cache so repeated lookups (same cities) are instant.
_GEO_CACHE: dict[str, tuple[float, float] | None] = {}

#: Spoken when a query IS a distance ask but can't be grounded. Returned instead
#: of None so the request never falls through to the LLM, which would otherwise
#: hallucinate a wrong number (e.g. "Kota to Vapi = 63 km" — it's ~760 km).
_DISTANCE_FAIL = (
    "I couldn't pull up that exact route just now, Boss — the map service was slow. "
    "Give me a moment and ask again."
)

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


def _geocode_nominatim(place: str) -> tuple[float, float] | None:
    query = urllib.parse.urlencode({"q": place, "format": "json", "limit": 1})
    data = _get(f"{_NOMINATIM}?{query}")
    if isinstance(data, list) and data:
        try:
            return float(data[0]["lat"]), float(data[0]["lon"])
        except (KeyError, ValueError, TypeError):
            return None
    return None


def _geocode_photon(place: str) -> tuple[float, float] | None:
    """Forward-geocode via Photon — the cloud-friendly fallback when Nominatim
    rate-limits Render (which is why distances were falling through to the LLM)."""
    query = urllib.parse.urlencode({"q": place, "limit": 1})
    data = _get(f"{_PHOTON_SEARCH}?{query}")
    if isinstance(data, dict):
        features = data.get("features") or []
        if features:
            coords = (features[0].get("geometry") or {}).get("coordinates") or []
            if len(coords) >= 2:
                try:
                    return float(coords[1]), float(coords[0])  # [lon, lat] -> lat, lon
                except (ValueError, TypeError):
                    return None
    return None


def _geocode(place: str) -> tuple[float, float] | None:
    """Locate ``place`` (Nominatim, then Photon), cached across requests."""
    key = place.strip().lower()
    if key in _GEO_CACHE:
        return _GEO_CACHE[key]
    result = _geocode_nominatim(place) or _geocode_photon(place)
    _GEO_CACHE[key] = result
    return result


def _route(
    a: tuple[float, float], b: tuple[float, float]
) -> tuple[float, float, str, list[Any]] | None:
    # OSRM wants lon,lat order. ``steps=true`` gives each leg's road ``summary``
    # (fallback "via"); the simplified geojson geometry is sampled for via-cities.
    url = (
        f"{_OSRM}/{a[1]},{a[0]};{b[1]},{b[0]}"
        "?overview=simplified&geometries=geojson&steps=true"
    )
    data = _get(url)
    if isinstance(data, dict):
        routes = data.get("routes") or []
        if routes:
            dist = routes[0].get("distance")
            dur = routes[0].get("duration")
            legs = routes[0].get("legs") or []
            summary = str(legs[0].get("summary", "")) if legs else ""
            geometry = routes[0].get("geometry") or {}
            coords = geometry.get("coordinates", []) if isinstance(geometry, dict) else []
            if isinstance(dist, int | float):
                return float(dist), float(dur or 0), summary, list(coords)
    return None


def _reverse_nominatim(lat: float, lon: float) -> str | None:
    """Nominatim reverse at city zoom — returns the MAJOR city/district of a point."""
    query = urllib.parse.urlencode(
        {"lat": lat, "lon": lon, "format": "json", "zoom": "10"}
    )
    data = _get(f"{_NOM_REVERSE}?{query}")
    if isinstance(data, dict):
        address = data.get("address") or {}
        for key in ("city", "town", "state_district", "municipality", "county"):
            value = address.get(key)
            if value:
                return str(value)
    return None


def _reverse_photon(lat: float, lon: float) -> str | None:
    """Photon reverse — cloud-friendly fallback (nearest named place)."""
    query = urllib.parse.urlencode({"lat": lat, "lon": lon})
    data = _get(f"{_PHOTON}?{query}")
    if isinstance(data, dict):
        features = data.get("features") or []
        if features:
            props = features[0].get("properties") or {}
            for key in ("city", "county", "district", "name"):
                value = props.get(key)
                if value:
                    return str(value)
    return None


def _reverse(lat: float, lon: float) -> str | None:
    """Best-effort city for a point: Nominatim (major city), else Photon."""
    return _reverse_nominatim(lat, lon) or _reverse_photon(lat, lon)


#: Strip administrative suffixes so a waypoint reads as a clean city name
#: ("Thandla Tahsil" -> "Thandla", "Kota District" -> "Kota").
_ADMIN_SUFFIX = re.compile(
    r"\s+(tahsil|tehsil|taluka|taluk|mandal|district|division|block|sub-?district)\b.*$",
    re.IGNORECASE,
)


def _clean_city(name: str) -> str:
    return _ADMIN_SUFFIX.sub("", name).strip(" ,") or name


def _safe_result(future: concurrent.futures.Future[str | None]) -> str | None:
    try:
        return future.result(timeout=_REVERSE_TIMEOUT)
    except Exception:  # noqa: BLE001 - a slow/failed lookup is just a miss
        return None


def _via_cities(coords: list[Any]) -> list[str]:
    """Two waypoint CITIES along the route (not road names).

    Each sample point is reverse-geocoded by Photon AND Nominatim *in parallel*
    (first valid wins), so one provider rate-limiting Render no longer drops us to
    raw road summaries. Every lookup runs concurrently and is time-bounded, so a
    slow network can't stall the answer (which used to make Siri time out).
    """
    if len(coords) < 3:
        return []
    points: list[tuple[float, float]] = []
    for fraction in (0.30, 0.55, 0.78):
        point = coords[int(len(coords) * fraction)]
        if isinstance(point, list) and len(point) >= 2:
            points.append((float(point[1]), float(point[0])))  # [lon, lat] -> lat, lon
    if not points:
        return []
    cities: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=2 * len(points)) as pool:
        pairs = [
            (
                pool.submit(_reverse_photon, lat, lon),
                pool.submit(_reverse_nominatim, lat, lon),
            )
            for lat, lon in points
        ]
        for f_photon, f_nominatim in pairs:
            raw = _safe_result(f_photon) or _safe_result(f_nominatim)
            if not raw:
                continue
            city = _clean_city(raw)
            if city and city not in cities:
                cities.append(city)
            if len(cities) >= 2:
                break
    return cities


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
    """A grounded spoken distance answer.

    Returns ``None`` only when the text is NOT a distance query (so other handlers
    take it). When it IS a distance query but can't be grounded, returns a
    graceful message — never ``None`` — so it never falls through to the LLM,
    which would invent a wrong number.
    """
    places = _places(text)
    if places is None:
        return None
    a_name, b_name = places
    # Geocode both endpoints concurrently (each tries Nominatim then Photon).
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        fa = pool.submit(_geocode, a_name)
        fb = pool.submit(_geocode, b_name)
        try:
            a = fa.result(timeout=_TIMEOUT * 2 + 1)
            b = fb.result(timeout=_TIMEOUT * 2 + 1)
        except Exception:  # noqa: BLE001 - treat a stuck lookup as a miss
            return _DISTANCE_FAIL
    if a is None or b is None:
        return _DISTANCE_FAIL
    routed = _route(a, b)
    if routed is None:
        return _DISTANCE_FAIL
    meters, seconds, summary, coords = routed
    if not meters:
        return _DISTANCE_FAIL
    km = round(meters / 1000)
    cities = _via_cities(coords)
    via_label = ", ".join(cities) if cities else summary
    via = f" via {via_label}" if via_label else ""
    return (
        f"Fastest from {a_name.title()} to {b_name.title()} is about {km} kilometres"
        f"{via} — total drive time about {_format_duration(seconds)}, Boss."
    )


def _format_duration(seconds: float) -> str:
    """Precise spoken drive time: 'X hours Y minutes' (the free-flow OSRM estimate)."""
    total_minutes = max(1, round(seconds / 60))
    hours, minutes = divmod(total_minutes, 60)
    h = f"{hours} hour{'s' if hours != 1 else ''}"
    m = f"{minutes} minute{'s' if minutes != 1 else ''}"
    if hours and minutes:
        return f"{h} {m}"
    return h if hours else m
