"""``/maps`` — the flagged interactive globe surface (Tier 3; default off).

A **fully keyless / no-paid-API** map: MapLibre GL renders an OpenStreetMap globe
in the browser, and place search, geocoding and routing are proxied here from the
free OpenStreetMap ecosystem (Nominatim + OSRM) so the browser never holds a key
and we never call a billable Google Maps API. Everything is gated behind
``FRIDAY_ENABLE_MAPS`` (read lazily off :func:`~friday.config.get_settings`); when
the flag is off every route is ``404`` so the feature simply does not exist:

* ``GET /maps`` — serves the no-build ``index.html`` (MapLibre globe).
* ``GET /maps/config`` — ``{"enabled": True, "provider": "maplibre-osm"}`` (no key).
* ``GET /maps/static/{filename}`` — serves the frontend assets (no-traversal guard).
* ``GET /maps/geocode?q=`` — Nominatim free-text geocode / place search.
* ``GET /maps/route?from_lat=&from_lng=&to_lat=&to_lng=`` — OSRM driving route.
* ``GET /maps/weather?location=`` — keyless current conditions (wttr.in).

Upstream calls are server-side with an identifying ``User-Agent`` (Nominatim
requires one) to honour the OSM community fair-use policies.
"""

from __future__ import annotations

from pathlib import Path

import httpx
from fastapi import APIRouter
from fastapi.responses import FileResponse, JSONResponse

from friday.config import get_settings
from friday.logging import get_logger
from friday.tools.weather import WeatherArgs, WeatherTool

logger = get_logger("friday.api.routes_maps")

router = APIRouter()

#: Keyless current-weather lookup (wttr.in) reused by ``GET /maps/weather`` so the
#: globe can overlay live conditions without a new external integration. Stateless
#: (holds only a timeout), so a single shared instance is safe.
_weather_tool = WeatherTool()

#: Free, keyless OpenStreetMap-ecosystem services the globe proxies through so it
#: never calls a billable Google Maps API. Calls are made server-side with an
#: identifying User-Agent (Nominatim requires one) to respect their fair-use
#: policies; the browser only ever talks to FRIDAY's own endpoints.
_OSM_USER_AGENT = "FRIDAY-maps/1.0 (self-hosted personal assistant)"
_NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
_NOMINATIM_REVERSE_URL = "https://nominatim.openstreetmap.org/reverse"
_OSRM_URL = "https://router.project-osrm.org/route/v1/driving"
_OSM_TIMEOUT = 10.0

#: The no-build frontend asset directory for the maps surface.
STATIC_DIR = Path(__file__).resolve().parent.parent / "maps" / "static"
#: The maps single-page entrypoint served by ``GET /maps``.
INDEX_PATH = STATIC_DIR / "index.html"

#: Asset content types we serve from ``/maps/static`` (no executable fallthrough).
_MEDIA_TYPES = {
    ".html": "text/html",
    ".js": "text/javascript",
    ".css": "text/css",
    ".map": "application/json",
    ".json": "application/json",
}


def _maps_enabled() -> bool:
    """Whether the maps surface is enabled, read lazily from settings."""
    return bool(getattr(get_settings(), "enable_maps", False))


def _disabled() -> JSONResponse:
    """The canonical ``maps disabled`` 404 response."""
    return JSONResponse(status_code=404, content={"detail": "maps disabled"})


@router.get("/maps", response_model=None)
async def maps_index() -> FileResponse | JSONResponse:
    """Serve the maps ``index.html`` single-page app; 404 when disabled/missing."""
    if not _maps_enabled():
        return _disabled()
    if not INDEX_PATH.is_file():  # pragma: no cover - frontend asset always present
        return JSONResponse(
            status_code=404,
            content={"detail": "maps UI not found (frontend assets missing)"},
        )
    return FileResponse(INDEX_PATH, media_type="text/html")


@router.get("/maps/config", response_model=None)
async def maps_config() -> JSONResponse:
    """Return the runtime maps config; 404 when disabled.

    The map renders with MapLibre GL + OpenStreetMap and needs **no API key**, so
    this only signals that the surface is enabled (kept as a runtime fetch the
    page already performs to decide whether to boot or show the disabled notice).
    """
    if not _maps_enabled():
        return _disabled()
    return JSONResponse(status_code=200, content={"enabled": True, "provider": "maplibre-osm"})


@router.get("/maps/static/{filename:path}", response_model=None)
async def maps_static(filename: str) -> FileResponse | JSONResponse:
    """Serve a single asset from ``static/``; 404 when disabled/missing/escaping."""
    if not _maps_enabled():
        return _disabled()
    # Resolve and confine strictly inside STATIC_DIR (path-traversal guard).
    candidate = (STATIC_DIR / filename).resolve()
    try:
        candidate.relative_to(STATIC_DIR.resolve())
    except ValueError:
        return JSONResponse(status_code=404, content={"detail": "not found"})
    if not candidate.is_file():
        return JSONResponse(status_code=404, content={"detail": "not found"})
    media_type = _MEDIA_TYPES.get(candidate.suffix.lower(), "application/octet-stream")
    return FileResponse(candidate, media_type=media_type)


@router.get("/maps/weather", response_model=None)
async def maps_weather(location: str = "") -> JSONResponse:
    """Current weather for ``location`` (keyless wttr.in); 404 when maps disabled.

    ``location`` is taken as a plain query param with an empty default (no
    parameter-level validation) so the feature-flag check runs *first* — a
    disabled feature returns 404 even for a missing/blank location rather than a
    422 that would leak that the route exists. The keyless :class:`WeatherTool`
    is reused, so this adds no new external integration or key.
    """
    if not _maps_enabled():
        return _disabled()
    loc = location.strip()
    if not loc or len(loc) > 200:
        return JSONResponse(
            status_code=422, content={"detail": "a non-empty 'location' query param is required"}
        )
    try:
        args = WeatherArgs(location=loc)
    except ValueError:
        return JSONResponse(status_code=422, content={"detail": "invalid location"})

    result = await _weather_tool(args)
    if not result.ok:
        detail = result.error.message if result.error else "weather lookup failed"
        return JSONResponse(status_code=502, content={"detail": detail})
    return JSONResponse(status_code=200, content=result.data)


@router.get("/maps/geocode", response_model=None)
async def maps_geocode(q: str = "", limit: int = 5) -> JSONResponse:
    """Geocode / place-search ``q`` via Nominatim (keyless); 404 when maps disabled.

    Powers both "fly to <place>" and "find <query>" — Nominatim's free-text search
    is the same endpoint. Flag is checked before validation so a disabled feature
    returns 404, never a 422 that leaks existence. Returns ``{"results": [{name,
    lat, lng}, ...]}`` (empty list when nothing matched).
    """
    if not _maps_enabled():
        return _disabled()
    query = q.strip()
    if not query or len(query) > 300:
        return JSONResponse(
            status_code=422, content={"detail": "a non-empty 'q' query param is required"}
        )
    count = max(1, min(int(limit) if str(limit).lstrip("-").isdigit() else 5, 10))
    try:
        async with httpx.AsyncClient(
            timeout=_OSM_TIMEOUT, headers={"User-Agent": _OSM_USER_AGENT}
        ) as client:
            resp = await client.get(
                _NOMINATIM_URL, params={"q": query, "format": "jsonv2", "limit": count}
            )
    except httpx.HTTPError as exc:
        return JSONResponse(status_code=502, content={"detail": f"geocoding upstream error: {exc}"})
    if resp.status_code != httpx.codes.OK:
        return JSONResponse(
            status_code=502, content={"detail": f"geocoding returned HTTP {resp.status_code}"}
        )
    try:
        rows = resp.json()
    except ValueError:
        return JSONResponse(
            status_code=502, content={"detail": "geocoding returned a non-JSON body"}
        )

    results: list[dict[str, object]] = []
    for row in rows if isinstance(rows, list) else []:
        try:
            results.append(
                {
                    "name": str(row.get("display_name", "")),
                    "lat": float(row["lat"]),
                    "lng": float(row["lon"]),
                }
            )
        except (KeyError, TypeError, ValueError):
            continue
    return JSONResponse(status_code=200, content={"results": results})


@router.get("/maps/route", response_model=None)
async def maps_route(
    from_lat: str = "", from_lng: str = "", to_lat: str = "", to_lng: str = ""
) -> JSONResponse:
    """Driving route between two points via OSRM (keyless); 404 when maps disabled.

    Returns ``{distance_km, duration_min, coordinates}`` where ``coordinates`` is
    a GeoJSON ``[[lng, lat], ...]`` path the front-end draws directly. The public
    OSRM demo serves the driving profile only.
    """
    if not _maps_enabled():
        return _disabled()
    try:
        flat, flng, tlat, tlng = (
            float(from_lat), float(from_lng), float(to_lat), float(to_lng)
        )
    except (TypeError, ValueError):
        return JSONResponse(
            status_code=422,
            content={"detail": "from_lat/from_lng/to_lat/to_lng must all be numbers"},
        )
    lat_ok = -90 <= flat <= 90 and -90 <= tlat <= 90
    lng_ok = -180 <= flng <= 180 and -180 <= tlng <= 180
    if not (lat_ok and lng_ok):
        return JSONResponse(status_code=422, content={"detail": "coordinates out of range"})

    url = f"{_OSRM_URL}/{flng},{flat};{tlng},{tlat}"
    try:
        async with httpx.AsyncClient(
            timeout=_OSM_TIMEOUT, headers={"User-Agent": _OSM_USER_AGENT}
        ) as client:
            resp = await client.get(url, params={"overview": "full", "geometries": "geojson"})
    except httpx.HTTPError as exc:
        return JSONResponse(status_code=502, content={"detail": f"routing upstream error: {exc}"})
    if resp.status_code != httpx.codes.OK:
        return JSONResponse(
            status_code=502, content={"detail": f"routing returned HTTP {resp.status_code}"}
        )
    try:
        route = resp.json()["routes"][0]
        payload = {
            "distance_km": round(float(route["distance"]) / 1000, 2),
            "duration_min": round(float(route["duration"]) / 60),
            "coordinates": route["geometry"]["coordinates"],
        }
    except (ValueError, KeyError, IndexError, TypeError):
        return JSONResponse(
            status_code=502, content={"detail": "routing returned an unexpected body"}
        )
    return JSONResponse(status_code=200, content=payload)


@router.get("/maps/reverse", response_model=None)
async def maps_reverse(lat: str = "", lng: str = "") -> JSONResponse:
    """Reverse-geocode a point to an address via Nominatim; 404 when maps disabled.

    Powers click-to-identify. Returns ``{"name": "<address or empty>"}``.
    """
    if not _maps_enabled():
        return _disabled()
    try:
        flat, flng = float(lat), float(lng)
    except (TypeError, ValueError):
        return JSONResponse(
            status_code=422, content={"detail": "lat/lng must be numbers"}
        )
    if not (-90 <= flat <= 90 and -180 <= flng <= 180):
        return JSONResponse(status_code=422, content={"detail": "coordinates out of range"})
    try:
        async with httpx.AsyncClient(
            timeout=_OSM_TIMEOUT, headers={"User-Agent": _OSM_USER_AGENT}
        ) as client:
            resp = await client.get(
                _NOMINATIM_REVERSE_URL, params={"lat": flat, "lon": flng, "format": "jsonv2"}
            )
    except httpx.HTTPError as exc:
        return JSONResponse(
            status_code=502, content={"detail": f"reverse geocoding upstream error: {exc}"}
        )
    if resp.status_code != httpx.codes.OK:
        return JSONResponse(
            status_code=502,
            content={"detail": f"reverse geocoding returned HTTP {resp.status_code}"},
        )
    try:
        body = resp.json()
    except ValueError:
        return JSONResponse(
            status_code=502, content={"detail": "reverse geocoding returned a non-JSON body"}
        )
    name = str(body.get("display_name", "")) if isinstance(body, dict) else ""
    return JSONResponse(status_code=200, content={"name": name})
