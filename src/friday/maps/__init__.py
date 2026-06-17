"""The Maps feature (Tier 3) — off by default behind ``FRIDAY_ENABLE_MAPS``.

A flagged, no-build page that renders an interactive **OpenStreetMap globe** with
MapLibre GL — **fully keyless, no paid API**. The globe idle-rotates and supports
"fly to <place>", "distance to <place>", place search, driving routes, live
weather and multi-stop tours, driven by an in-page Web Speech mic and by
``?fly=`` / ``?to=`` / ``?cmd=`` query params on load.

Geocoding, search, routing and weather are proxied server-side from the free
OpenStreetMap ecosystem (Nominatim + OSRM) and wttr.in via ``/maps/geocode``,
``/maps/reverse``, ``/maps/route`` and ``/maps/weather`` — so the browser never
holds a key and no billable Google Maps API is ever called. The router
self-guards on the flag, so the offline default exposes no maps surface (every
route -> 404).

The integration agent wires this slice by including :data:`friday.maps.router`.
"""

from __future__ import annotations

from friday.api.routes_maps import router

__all__ = ["router"]
