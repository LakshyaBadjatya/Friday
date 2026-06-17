"""Integration tests for the flagged ``/maps`` surface (Tier 3; default off).

The Maps feature is a Photorealistic 3D globe served as a no-build page that
loads the Google Maps JS API. The API key is NEVER baked into the HTML; the
page fetches it at runtime from ``GET /maps/config``.

These tests mount :data:`friday.maps.router` on a FRESH ``FastAPI()`` app (NOT
``create_app``) with the relevant settings monkeypatched, so the slice passes
before any ``app.py`` wiring exists. The maps client/key is read lazily inside
the route from :func:`~friday.config.get_settings`.

Covered:
* ``GET /maps`` and ``GET /maps/config`` both ``404`` when ``enable_maps`` off.
* ``GET /maps`` serves the index HTML when enabled (loads the Maps JS API; no
  key baked in).
* ``GET /maps/config`` returns ``{"apiKey": <key or "">, "enabled": True}`` when
  enabled — the key field is present and carries the configured secret.
* ``GET /maps/config`` returns ``apiKey == ""`` when enabled but no key is set
  (never crashes on a missing key).
* The static ``maps.js`` passes ``node --check`` (valid JavaScript).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import httpx
import pytest
import respx
from fastapi import FastAPI
from fastapi.testclient import TestClient

import friday.api.routes_maps as routes_maps
from friday.config import Settings
from friday.maps import router as maps_router

_STATIC_DIR = Path(routes_maps.__file__).resolve().parent.parent / "maps" / "static"
_MAPS_JS = _STATIC_DIR / "maps.js"
_INDEX_HTML = _STATIC_DIR / "index.html"


def _app() -> FastAPI:
    """A fresh app with ONLY the maps router mounted (no ``create_app``)."""
    app = FastAPI()
    app.include_router(maps_router)
    return app


def _enabled_settings(key: str | None = "test-key-123") -> Settings:
    return Settings(
        _env_file=None,
        enable_maps=True,
        google_maps_api_key=key,
    )


def _disabled_settings() -> Settings:
    return Settings(_env_file=None, enable_maps=False)


def test_maps_index_disabled_returns_404(monkeypatch: pytest.MonkeyPatch) -> None:
    """``GET /maps`` is 404 when the maps flag is off (feature does not exist)."""
    monkeypatch.setattr(routes_maps, "get_settings", _disabled_settings)
    with TestClient(_app()) as client:
        resp = client.get("/maps")
    assert resp.status_code == 404


def test_maps_config_disabled_returns_404(monkeypatch: pytest.MonkeyPatch) -> None:
    """``GET /maps/config`` is 404 when the maps flag is off."""
    monkeypatch.setattr(routes_maps, "get_settings", _disabled_settings)
    with TestClient(_app()) as client:
        resp = client.get("/maps/config")
    assert resp.status_code == 404


def test_maps_index_enabled_serves_html(monkeypatch: pytest.MonkeyPatch) -> None:
    """Enabled ``GET /maps`` serves the index HTML; loads Maps JS, no key baked."""
    monkeypatch.setattr(routes_maps, "get_settings", lambda: _enabled_settings())
    with TestClient(_app()) as client:
        resp = client.get("/maps")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    body = resp.text
    # The page loads the keyless MapLibre + OpenStreetMap stack — no Google API.
    assert "maplibre-gl" in body
    assert "/maps/static/maps.js" in body
    assert "googleapis.com" not in body


def test_maps_config_enabled_returns_provider_no_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Enabled ``GET /maps/config`` signals the keyless MapLibre/OSM provider."""
    monkeypatch.setattr(routes_maps, "get_settings", lambda: _enabled_settings())
    with TestClient(_app()) as client:
        resp = client.get("/maps/config")
    assert resp.status_code == 200
    body = resp.json()
    assert body == {"enabled": True, "provider": "maplibre-osm"}
    assert "apiKey" not in body  # no key is ever delivered to the browser


def test_maps_config_needs_no_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """The map is keyless: config is identical whether or not a Google key is set."""
    monkeypatch.setattr(
        routes_maps, "get_settings", lambda: _enabled_settings(key=None)
    )
    with TestClient(_app()) as client:
        resp = client.get("/maps/config")
    assert resp.status_code == 200
    assert resp.json() == {"enabled": True, "provider": "maplibre-osm"}


def test_maps_serves_static_js(monkeypatch: pytest.MonkeyPatch) -> None:
    """Enabled ``GET /maps/static/maps.js`` serves the frontend controller."""
    monkeypatch.setattr(routes_maps, "get_settings", lambda: _enabled_settings())
    with TestClient(_app()) as client:
        resp = client.get("/maps/static/maps.js")
    assert resp.status_code == 200
    assert "javascript" in resp.headers["content-type"]
    # No key is baked into the JS either; it is fetched from /maps/config.
    assert "test-key-123" not in resp.text


def test_maps_static_disabled_returns_404(monkeypatch: pytest.MonkeyPatch) -> None:
    """The static asset route is also absent when the flag is off."""
    monkeypatch.setattr(routes_maps, "get_settings", _disabled_settings)
    with TestClient(_app()) as client:
        resp = client.get("/maps/static/maps.js")
    assert resp.status_code == 404


def test_maps_js_is_valid_javascript() -> None:
    """``maps.js`` passes ``node --check`` (parses as valid JavaScript)."""
    node = shutil.which("node")
    if node is None:  # pragma: no cover - node is present in this environment
        pytest.skip("node is not available")
    result = subprocess.run(
        [node, "--check", str(_MAPS_JS)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_maps_js_modules_are_valid_javascript() -> None:
    """Every ES module under ``static/js/`` passes ``node --check`` (valid JS)."""
    node = shutil.which("node")
    if node is None:  # pragma: no cover - node is present in this environment
        pytest.skip("node is not available")
    modules = sorted((_STATIC_DIR / "js").glob("*.js"))
    assert modules, "expected split maps modules under static/js/"
    for module in modules:
        result = subprocess.run([node, "--check", str(module)], capture_output=True, text=True)
        assert result.returncode == 0, f"{module.name}: {result.stderr}"


def test_maps_index_references_runtime_config() -> None:
    """The index HTML wires the runtime ``/maps/config`` fetch (no baked key)."""
    html = _INDEX_HTML.read_text(encoding="utf-8")
    assert "/maps/config" in html
    # The keyless MapLibre + OpenStreetMap stack is loaded — never a Google API.
    assert "maplibre-gl" in html
    assert "googleapis.com" not in html


# --------------------------------------------------------------------------- #
# GET /maps/weather — live conditions overlay (reuses the keyless WeatherTool)
# --------------------------------------------------------------------------- #
_KOTA_URL = "https://wttr.in/Kota"
_KOTA_J1 = {
    "current_condition": [
        {
            "temp_C": "38",
            "FeelsLikeC": "41",
            "humidity": "20",
            "windspeedKmph": "12",
            "weatherDesc": [{"value": "Sunny"}],
        }
    ],
    "nearest_area": [{"areaName": [{"value": "Kota"}]}],
}


def test_maps_weather_disabled_404_even_without_location(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Flag-off returns 404 even for a missing location (no existence leak via 422)."""
    monkeypatch.setattr(routes_maps, "get_settings", _disabled_settings)
    with TestClient(_app()) as client:
        resp = client.get("/maps/weather")  # no location at all
    assert resp.status_code == 404
    assert resp.json()["detail"] == "maps disabled"


def test_maps_weather_enabled_missing_location_is_422(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(routes_maps, "get_settings", lambda: _enabled_settings())
    with TestClient(_app()) as client:
        resp = client.get("/maps/weather")
    assert resp.status_code == 422


@respx.mock
def test_maps_weather_enabled_returns_card(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(routes_maps, "get_settings", lambda: _enabled_settings())
    respx.get(_KOTA_URL).mock(return_value=httpx.Response(200, json=_KOTA_J1))
    with TestClient(_app()) as client:
        resp = client.get("/maps/weather", params={"location": "Kota"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["location"] == "Kota"
    assert body["temp_c"] == 38
    assert body["wind_kph"] == 12
    assert "Sunny" in body["description"]


@respx.mock
def test_maps_weather_upstream_failure_is_502(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(routes_maps, "get_settings", lambda: _enabled_settings())
    respx.get(_KOTA_URL).mock(return_value=httpx.Response(500, text="boom"))
    with TestClient(_app()) as client:
        resp = client.get("/maps/weather", params={"location": "Kota"})
    assert resp.status_code == 502


# --------------------------------------------------------------------------- #
# GET /maps/geocode + /maps/reverse + /maps/route — keyless OSM proxies
# --------------------------------------------------------------------------- #
_NOMINATIM_SEARCH = "https://nominatim.openstreetmap.org/search"
_NOMINATIM_REVERSE = "https://nominatim.openstreetmap.org/reverse"


def test_maps_geocode_disabled_404_even_without_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Flag-off returns 404 even for a missing query (no existence leak via 422)."""
    monkeypatch.setattr(routes_maps, "get_settings", _disabled_settings)
    with TestClient(_app()) as client:
        resp = client.get("/maps/geocode")
    assert resp.status_code == 404


def test_maps_geocode_missing_query_is_422(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(routes_maps, "get_settings", lambda: _enabled_settings())
    with TestClient(_app()) as client:
        resp = client.get("/maps/geocode")
    assert resp.status_code == 422


@respx.mock
def test_maps_geocode_returns_normalized_results(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(routes_maps, "get_settings", lambda: _enabled_settings())
    respx.get(_NOMINATIM_SEARCH).mock(
        return_value=httpx.Response(
            200, json=[{"display_name": "London, UK", "lat": "51.5074", "lon": "-0.1278"}]
        )
    )
    with TestClient(_app()) as client:
        resp = client.get("/maps/geocode", params={"q": "London"})
    assert resp.status_code == 200
    results = resp.json()["results"]
    assert results[0] == {"name": "London, UK", "lat": 51.5074, "lng": -0.1278}


@respx.mock
def test_maps_geocode_upstream_failure_is_502(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(routes_maps, "get_settings", lambda: _enabled_settings())
    respx.get(_NOMINATIM_SEARCH).mock(return_value=httpx.Response(500, text="boom"))
    with TestClient(_app()) as client:
        resp = client.get("/maps/geocode", params={"q": "London"})
    assert resp.status_code == 502


@respx.mock
def test_maps_reverse_returns_address(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(routes_maps, "get_settings", lambda: _enabled_settings())
    respx.get(_NOMINATIM_REVERSE).mock(
        return_value=httpx.Response(200, json={"display_name": "10 Downing St, London"})
    )
    with TestClient(_app()) as client:
        resp = client.get("/maps/reverse", params={"lat": "51.5", "lng": "-0.12"})
    assert resp.status_code == 200
    assert resp.json()["name"] == "10 Downing St, London"


def test_maps_reverse_bad_coords_is_422(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(routes_maps, "get_settings", lambda: _enabled_settings())
    with TestClient(_app()) as client:
        resp = client.get("/maps/reverse", params={"lat": "abc", "lng": "0"})
    assert resp.status_code == 422


@respx.mock
def test_maps_route_returns_distance_duration_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(routes_maps, "get_settings", lambda: _enabled_settings())
    respx.route(host="router.project-osrm.org").mock(
        return_value=httpx.Response(
            200,
            json={
                "routes": [
                    {
                        "distance": 343000.0,
                        "duration": 12600.0,
                        "geometry": {"coordinates": [[-0.12, 51.5], [2.35, 48.85]]},
                    }
                ]
            },
        )
    )
    with TestClient(_app()) as client:
        resp = client.get(
            "/maps/route",
            params={"from_lat": "51.5", "from_lng": "-0.12", "to_lat": "48.85", "to_lng": "2.35"},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["distance_km"] == 343.0
    assert body["duration_min"] == 210
    assert body["coordinates"][0] == [-0.12, 51.5]


def test_maps_route_bad_coords_is_422(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(routes_maps, "get_settings", lambda: _enabled_settings())
    with TestClient(_app()) as client:
        resp = client.get(
            "/maps/route",
            params={"from_lat": "x", "from_lng": "0", "to_lat": "0", "to_lng": "0"},
        )
    assert resp.status_code == 422
