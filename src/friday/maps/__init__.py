"""The Maps feature (Tier 3) — off by default behind ``FRIDAY_ENABLE_MAPS``.

A flagged, no-build page that loads the Google Maps Platform JS API and renders a
**Photorealistic 3D globe** (``gmp-map-3d`` / ``Map3DElement``). The globe idle-
rotates (``flyCameraAround``), supports "fly to <place>" (geocode then
``flyCameraTo`` with a Google-Earth-style animation) and "distance to <place>"
(geocode + draw a line + show the distance), driven by an in-page Web Speech mic
and by ``?fly=`` / ``?to=`` query params on load.

Security: the Google Maps API key is NEVER baked into the served HTML. The page
fetches it at runtime from ``GET /maps/config``; on the backend the key is a
:class:`~pydantic.SecretStr` (never logged). The router self-guards on the flag,
so the offline default exposes no maps surface (every route -> 404).

The integration agent wires this slice by including :data:`friday.maps.router`.
"""

from __future__ import annotations

from friday.api.routes_maps import router

__all__ = ["router"]
