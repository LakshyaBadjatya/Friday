/*
 * FRIDAY Maps — globe core (browser ES module), MapLibre GL + OpenStreetMap.
 *
 * Keyless / no-paid-API: MapLibre GL renders an OpenStreetMap raster globe, and
 * geocoding / reverse-geocoding are proxied from FRIDAY's own backend (which in
 * turn uses the free Nominatim service). Exposes the same function names the
 * feature + command modules already import, so swapping the provider here is the
 * only change they need.
 *
 * Responsibilities: build the globe, idle-rotate it, camera helpers
 * (flyToCoords/home/zoomBy/tiltBy/fitTo), overlays (markers + lines), geocode /
 * reverseGeocode / haversineKm, flyTo / distanceTo, and click-to-identify.
 */

"use strict";

import { state, HOME, HOME_ZOOM, setHud, setConn, toast, showPanel } from "./ui.js";

const mlg = () => window.maplibregl;

// ── Globe construction ────────────────────────────────────────────────────────
export async function buildGlobe() {
  if (!mlg()) throw new Error("MapLibre GL is not loaded");
  // A Google-Earth-style satellite globe — KEYLESS: Esri World Imagery for the
  // imagery, a transparent Esri places/boundaries layer for labels, and free AWS
  // "terrarium" DEM tiles for 3D terrain relief. No Google, no API key.
  const style = {
    version: 8,
    sources: {
      sat: {
        type: "raster",
        tiles: [
          "https://services.arcgisonline.com/arcgis/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        ],
        tileSize: 256,
        maxzoom: 19,
        attribution: "Imagery © Esri, Maxar, Earthstar Geographics",
      },
      labels: {
        type: "raster",
        tiles: [
          "https://services.arcgisonline.com/arcgis/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
        ],
        tileSize: 256,
        maxzoom: 19,
      },
      dem: {
        type: "raster-dem",
        tiles: ["https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"],
        encoding: "terrarium",
        tileSize: 256,
        maxzoom: 14,
      },
    },
    layers: [
      { id: "sat", type: "raster", source: "sat" },
      { id: "labels", type: "raster", source: "labels" },
    ],
  };
  const map = new (mlg().Map)({
    container: "map",
    style,
    center: [HOME.lng, HOME.lat],
    zoom: HOME_ZOOM,
    attributionControl: true,
  });
  state.map = map;
  await new Promise((resolve) => map.on("load", resolve));
  // Globe projection (MapLibre v5) + 3D terrain relief — both feature-detected so
  // an older MapLibre or an unreachable DEM degrades gracefully (flat / no relief).
  try {
    map.setProjection({ type: "globe" });
  } catch (_e) {
    /* mercator fallback */
  }
  try {
    if (typeof map.setTerrain === "function") {
      map.setTerrain({ source: "dem", exaggeration: 1.3 });
    }
  } catch (_e) {
    /* terrain optional */
  }
  wireClickToIdentify();
  setConn("ok", "connected");
  setHud("status", "ready");
  startIdleRotation();
}

// ── Idle rotation ─────────────────────────────────────────────────────────────
export function startIdleRotation() {
  state.rotating = true;
  setHud("mode", "idle rotate");
  const btn = document.getElementById("rotate");
  if (btn) btn.setAttribute("aria-pressed", "true");
  if (state.spinTimer) clearInterval(state.spinTimer);
  state.spinTimer = setInterval(() => {
    if (!state.map || !state.rotating) return;
    const c = state.map.getCenter();
    state.map.easeTo({ center: [c.lng + 3, c.lat], duration: 1000, easing: (t) => t });
  }, 1000);
}

export function stopIdleRotation() {
  state.rotating = false;
  setHud("mode", "manual");
  const btn = document.getElementById("rotate");
  if (btn) btn.setAttribute("aria-pressed", "false");
  if (state.spinTimer) {
    clearInterval(state.spinTimer);
    state.spinTimer = null;
  }
}

// ── Camera helpers ────────────────────────────────────────────────────────────
export function flyToCoords(coords, opts = {}) {
  stopIdleRotation();
  state.focus = { lat: coords.lat, lng: coords.lng };
  if (!state.map) return;
  state.map.flyTo({
    center: [coords.lng, coords.lat],
    zoom: opts.zoom ?? 12,
    pitch: opts.pitch ?? 45,
    duration: opts.duration ?? 3000,
  });
}

export function home() {
  clearOverlays();
  state.focus = { ...HOME };
  if (state.map) {
    state.map.flyTo({ center: [HOME.lng, HOME.lat], zoom: HOME_ZOOM, pitch: 0, bearing: 0, duration: 2000 });
  }
  setHud("mode", "home");
  startIdleRotation();
}

/** factor < 1 zooms in, > 1 zooms out (preserves the command parser's contract). */
export function zoomBy(factor) {
  if (!state.map) return;
  stopIdleRotation();
  state.map.zoomTo(state.map.getZoom() + (factor < 1 ? 1 : -1), { duration: 500 });
}

export function tiltBy(delta) {
  if (!state.map) return;
  stopIdleRotation();
  const pitch = Math.min(85, Math.max(0, state.map.getPitch() + delta));
  state.map.easeTo({ pitch, duration: 500 });
}

export function fitTo(points) {
  if (!state.map || !points.length) return;
  stopIdleRotation();
  let minLng = 180;
  let minLat = 90;
  let maxLng = -180;
  let maxLat = -90;
  for (const p of points) {
    minLng = Math.min(minLng, p.lng);
    maxLng = Math.max(maxLng, p.lng);
    minLat = Math.min(minLat, p.lat);
    maxLat = Math.max(maxLat, p.lat);
  }
  state.map.fitBounds([[minLng, minLat], [maxLng, maxLat]], { padding: 80, duration: 2500, maxZoom: 12 });
}

// ── Overlays (markers + lines) ────────────────────────────────────────────────
export function addMarker(coords, label, onClick) {
  if (!state.map || !mlg()) return null;
  const marker = new (mlg().Marker)({ color: "#2b6cff" }).setLngLat([coords.lng, coords.lat]);
  if (label) marker.setPopup(new (mlg().Popup)({ offset: 24 }).setText(String(label)));
  marker.addTo(state.map);
  if (typeof onClick === "function") {
    const el = marker.getElement();
    if (el) el.addEventListener("click", () => onClick());
  }
  state.markers.push(marker);
  return marker;
}

let _lineSeq = 0;
export function addLine(coords, opts = {}) {
  if (!state.map) return null;
  const id = "friday-line-" + _lineSeq++;
  const data = {
    type: "Feature",
    geometry: { type: "LineString", coordinates: coords.map((c) => [c.lng, c.lat]) },
  };
  try {
    state.map.addSource(id, { type: "geojson", data });
    state.map.addLayer({
      id,
      type: "line",
      source: id,
      paint: { "line-color": opts.strokeColor || "#2b6cff", "line-width": opts.strokeWidth || 4 },
    });
    state.lineIds.push(id);
  } catch (_e) {
    return null;
  }
  return id;
}

export function clearOverlays() {
  for (const marker of state.markers) {
    try {
      marker.remove();
    } catch (_e) {
      /* already gone */
    }
  }
  state.markers = [];
  for (const id of state.lineIds) {
    try {
      if (state.map.getLayer(id)) state.map.removeLayer(id);
    } catch (_e) {
      /* ignore */
    }
    try {
      if (state.map.getSource(id)) state.map.removeSource(id);
    } catch (_e) {
      /* ignore */
    }
  }
  state.lineIds = [];
}

// ── Geocoding (via FRIDAY's keyless backend proxy → Nominatim) ────────────────
/** Geocode free text to `{lat, lng, name}` (or null on failure). */
export async function geocode(place) {
  try {
    const resp = await fetch("/maps/geocode?limit=1&q=" + encodeURIComponent(place), {
      headers: { Accept: "application/json" },
    });
    if (!resp.ok) return null;
    const data = await resp.json();
    const r = data.results && data.results[0];
    if (!r) return null;
    return { lat: r.lat, lng: r.lng, name: r.name };
  } catch (_e) {
    return null;
  }
}

/** Reverse-geocode `{lat,lng}` to an address string (or null). */
export async function reverseGeocode(coords) {
  try {
    const resp = await fetch(
      "/maps/reverse?lat=" + encodeURIComponent(coords.lat) + "&lng=" + encodeURIComponent(coords.lng),
      { headers: { Accept: "application/json" } }
    );
    if (!resp.ok) return null;
    const data = await resp.json();
    return data.name || null;
  } catch (_e) {
    return null;
  }
}

/** Great-circle distance (km) between two `{lat, lng}` points (haversine). */
export function haversineKm(a, b) {
  const R = 6371;
  const toRad = (d) => (d * Math.PI) / 180;
  const dLat = toRad(b.lat - a.lat);
  const dLng = toRad(b.lng - a.lng);
  const s =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(toRad(a.lat)) * Math.cos(toRad(b.lat)) * Math.sin(dLng / 2) ** 2;
  return 2 * R * Math.asin(Math.min(1, Math.sqrt(s)));
}

// ── High-level actions ────────────────────────────────────────────────────────
export async function flyTo(place) {
  setHud("status", "locating " + place + "…");
  const target = await geocode(place);
  if (!target) {
    toast("Couldn't find “" + place + "”.");
    setHud("status", "ready");
    return null;
  }
  flyToCoords(target);
  addMarker(target, target.name || place);
  setHud("mode", "fly to " + place);
  setHud("status", "ready");
  return target;
}

export async function distanceTo(place) {
  setHud("status", "measuring to " + place + "…");
  const target = await geocode(place);
  if (!target) {
    toast("Couldn't find “" + place + "”.");
    setHud("status", "ready");
    return;
  }
  const from = { lat: state.focus.lat, lng: state.focus.lng };
  const km = haversineKm(from, target);
  clearOverlays();
  addLine([from, target]);
  addMarker(target, place);
  const pretty = km >= 10 ? Math.round(km).toLocaleString() : km.toFixed(1);
  toast("Distance to " + place + ": " + pretty + " km");
  setHud("mode", "distance " + pretty + " km");
  setHud("status", "ready");
}

// ── Click-to-identify ─────────────────────────────────────────────────────────
function wireClickToIdentify() {
  if (!state.map) return;
  state.map.on("click", async (e) => {
    const lat = e.lngLat.lat;
    const lng = e.lngLat.lng;
    const address = await reverseGeocode({ lat, lng });
    showPanel({
      title: "Here",
      rows: [
        ["Address", address || "(no address found)"],
        ["Lat", lat.toFixed(5)],
        ["Lng", lng.toFixed(5)],
      ],
    });
  });
}
