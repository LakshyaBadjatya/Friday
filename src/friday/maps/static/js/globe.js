/*
 * FRIDAY Maps — globe core (browser ES module).
 *
 * The Map3DElement lifecycle and everything that touches the camera or the
 * globe's overlays:
 *   - buildGlobe(): import the 3D libraries, create the globe, wire click-to-
 *     identify, start the idle orbit.
 *   - idle rotation (flyCameraAround loop) + start/stop.
 *   - camera helpers: flyToCoords, home, zoomBy, tiltBy, fitTo.
 *   - overlays: addMarker / addLine / clearOverlays (tracked so we can wipe them).
 *   - geocode / reverseGeocode / haversineKm.
 *   - flyTo(place) / distanceTo(place): the high-level geocode→camera actions.
 *
 * Imports only ui.js (state + helpers), so the dependency graph stays acyclic.
 */

"use strict";

import { state, HOME, ORBIT_RANGE, setHud, setConn, toast, showPanel } from "./ui.js";

// ── Library bootstrap + globe construction ────────────────────────────────────
export async function buildGlobe() {
  const maps3d = await window.google.maps.importLibrary("maps3d");
  const geo = await window.google.maps.importLibrary("geocoding");

  state.Map3DElement = maps3d.Map3DElement;
  state.Polyline3DElement = maps3d.Polyline3DElement || null;
  state.Marker3DElement = maps3d.Marker3DElement || null;
  state.geocoder = new geo.Geocoder();

  // Optional libraries — feature-detect so a not-yet-propagated API does not
  // crash the globe; the dependent feature simply reports "unavailable".
  try {
    state.places = await window.google.maps.importLibrary("places");
  } catch (_e) {
    state.places = null;
  }
  try {
    state.routes = await window.google.maps.importLibrary("routes");
  } catch (_e) {
    state.routes = null;
  }

  const map = new state.Map3DElement({
    center: { lat: HOME.lat, lng: HOME.lng, altitude: HOME.altitude },
    range: ORBIT_RANGE,
    tilt: 0,
    heading: 0,
    mode: "HYBRID", // photorealistic 3D tiles
  });
  map.id = "map3d-el";
  const host = document.getElementById("map3d");
  if (host) host.replaceWith(map);
  else document.getElementById("stage").appendChild(map);
  state.map = map;

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
  orbitOnce();
}

export function stopIdleRotation() {
  state.rotating = false;
  setHud("mode", "manual");
  const btn = document.getElementById("rotate");
  if (btn) btn.setAttribute("aria-pressed", "false");
  if (state.map && typeof state.map.stopCameraAnimation === "function") {
    state.map.stopCameraAnimation();
  }
}

function orbitOnce() {
  if (!state.map || !state.rotating) return;
  if (typeof state.map.flyCameraAround !== "function") return;
  state.map.flyCameraAround({
    camera: {
      center: { lat: state.focus.lat, lng: state.focus.lng, altitude: state.focus.altitude || 0 },
      range: ORBIT_RANGE,
      tilt: 45,
    },
    durationMillis: 60000,
    rounds: 1,
  });
  const rearm = () => {
    if (state.rotating) orbitOnce();
  };
  if (typeof state.map.addEventListener === "function") {
    state.map.addEventListener("gmp-animationend", rearm, { once: true });
  } else {
    setTimeout(rearm, 60000);
  }
}

// ── Camera helpers ────────────────────────────────────────────────────────────
/** Animate the camera to `coords` ({lat,lng}); options override range/tilt. */
export function flyToCoords(coords, opts = {}) {
  stopIdleRotation();
  state.focus = { lat: coords.lat, lng: coords.lng, altitude: 0 };
  if (state.map && typeof state.map.flyCameraTo === "function") {
    state.map.flyCameraTo({
      endCamera: {
        center: { lat: coords.lat, lng: coords.lng, altitude: 0 },
        range: opts.range ?? 2500,
        tilt: opts.tilt ?? 65,
        heading: opts.heading ?? 0,
      },
      durationMillis: opts.durationMillis ?? 5000,
    });
  }
}

/** Reset to the whole-globe view and resume the idle orbit. */
export function home() {
  clearOverlays();
  state.focus = { ...HOME };
  if (state.map && typeof state.map.flyCameraTo === "function") {
    state.map.flyCameraTo({
      endCamera: { center: { ...HOME }, range: ORBIT_RANGE, tilt: 0, heading: 0 },
      durationMillis: 2000,
    });
  }
  setHud("mode", "home");
  startIdleRotation();
}

/** Multiply the current camera range (factor < 1 zooms in, > 1 zooms out). */
export function zoomBy(factor) {
  if (!state.map) return;
  stopIdleRotation();
  const range = (Number(state.map.range) || ORBIT_RANGE) * factor;
  flyToCoords(state.focus, { range, tilt: Number(state.map.tilt) || 65, durationMillis: 900 });
}

/** Nudge the camera tilt by `delta` degrees, clamped to [0, 90]. */
export function tiltBy(delta) {
  if (!state.map) return;
  stopIdleRotation();
  const tilt = Math.min(90, Math.max(0, (Number(state.map.tilt) || 0) + delta));
  flyToCoords(state.focus, { range: Number(state.map.range) || 2500, tilt, durationMillis: 700 });
}

/** Frame a set of {lat,lng} points by flying to their centroid at a fitted range. */
export function fitTo(points) {
  if (!points.length) return;
  const lat = points.reduce((s, p) => s + p.lat, 0) / points.length;
  const lng = points.reduce((s, p) => s + p.lng, 0) / points.length;
  let span = 0;
  for (const p of points) span = Math.max(span, haversineKm({ lat, lng }, p));
  const range = Math.max(2500, span * 2200); // ~metres; generous margin
  flyToCoords({ lat, lng }, { range, tilt: 55, durationMillis: 3000 });
}

// ── Overlays (markers + lines) ────────────────────────────────────────────────
export function addMarker(coords, label) {
  if (!state.Marker3DElement || !state.map) return null;
  const marker = new state.Marker3DElement({
    position: { lat: coords.lat, lng: coords.lng, altitude: 0 },
    label: label || "",
  });
  state.map.append(marker);
  state.overlays.push(marker);
  return marker;
}

export function addLine(coords, opts = {}) {
  if (!state.Polyline3DElement || !state.map) return null;
  const line = new state.Polyline3DElement({
    coordinates: coords.map((c) => ({ lat: c.lat, lng: c.lng, altitude: 0 })),
    strokeColor: opts.strokeColor || "#2b6cff",
    strokeWidth: opts.strokeWidth || 8,
    altitudeMode: "CLAMP_TO_GROUND",
    geodesic: opts.geodesic !== false,
  });
  state.map.append(line);
  state.overlays.push(line);
  return line;
}

export function clearOverlays() {
  for (const el of state.overlays) {
    if (el && typeof el.remove === "function") el.remove();
  }
  state.overlays = [];
}

// ── Geocoding ─────────────────────────────────────────────────────────────────
/** Geocode free text to `{lat, lng, address}` (or null on failure). */
export async function geocode(place) {
  if (!state.geocoder) return null;
  try {
    const { results } = await state.geocoder.geocode({ address: place });
    if (!results || !results.length) return null;
    const r = results[0];
    const loc = r.geometry.location;
    return { lat: loc.lat(), lng: loc.lng(), address: r.formatted_address || place };
  } catch (_e) {
    return null;
  }
}

/** Reverse-geocode `{lat,lng}` to a human address (or null). */
export async function reverseGeocode(coords) {
  if (!state.geocoder) return null;
  try {
    const { results } = await state.geocoder.geocode({ location: coords });
    if (!results || !results.length) return null;
    return results[0].formatted_address || null;
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
/** Geocode `place`, fly in Google-Earth style, and drop a labeled marker. */
export async function flyTo(place) {
  setHud("status", "locating " + place + "…");
  const target = await geocode(place);
  if (!target) {
    toast("Couldn't find “" + place + "”.");
    setHud("status", "ready");
    return null;
  }
  flyToCoords(target);
  addMarker(target, place);
  setHud("mode", "fly to " + place);
  setHud("status", "ready");
  return target;
}

/** Geocode `place`, draw a line from the current focus, and show the distance. */
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
  if (!state.map || typeof state.map.addEventListener !== "function") return;
  state.map.addEventListener("gmp-click", async (ev) => {
    const pos = ev && (ev.position || ev.latLng);
    if (!pos) return;
    const lat = typeof pos.lat === "function" ? pos.lat() : pos.lat;
    const lng = typeof pos.lng === "function" ? pos.lng() : pos.lng;
    if (typeof lat !== "number" || typeof lng !== "number") return;
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
