/*
 * FRIDAY Maps — frontend controller (no build step).
 *
 * Photorealistic 3D globe on the Google Maps Platform JS API. Responsibilities:
 *   1. Bootstrap: fetch the API key at RUNTIME from `/maps/config` (never baked
 *      into the page), then inject the official Maps JS loader and
 *      `importLibrary("maps3d")` / `importLibrary("geocoding")`.
 *   2. Globe: create a `Map3DElement` (gmp-map-3d) and idle-ROTATE it with
 *      `flyCameraAround` on a loop.
 *   3. "fly to <place>": geocode the place, then `flyCameraTo` with a
 *      Google-Earth-style animated fly-in.
 *   4. "distance to <place>": geocode, draw a `Polyline3DElement` from the
 *      current focus to the target, and show the great-circle distance.
 *   5. Voice: the Web Speech API drives the same commands in-page
 *      ("open maps", "fly to X", "show me distance to X").
 *   6. Deep-link: `?fly=` / `?to=` auto-animate on load.
 *
 * Everything is defensive: a disabled backend, a missing key, a blocked
 * SpeechRecognition, or a failed geocode shows a friendly HUD/toast message and
 * never hard-crashes the page.
 */

"use strict";

// ── DOM handles ──────────────────────────────────────────────────────────────
const $ = (id) => document.getElementById(id);
const mapHost = $("map3d");
const fallbackEl = $("fallback");
const fallbackMsg = $("fallback-msg");
const connStatus = $("conn-status");
const cmdForm = $("cmd-form");
const cmdInput = $("cmd");
const micBtn = $("mic");
const rotateBtn = $("rotate");
const toastEl = $("toast");
const hud = {
  mode: $("hud-mode"),
  voice: $("hud-voice"),
  heard: $("hud-heard"),
  status: $("hud-status"),
};

// ── UI helpers ───────────────────────────────────────────────────────────────
function setHud(key, text) {
  if (hud[key]) hud[key].textContent = text;
}

let toastTimer = null;
function toast(message) {
  if (!toastEl) return;
  toastEl.textContent = message;
  toastEl.hidden = false;
  if (toastTimer) clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    toastEl.hidden = true;
  }, 4200);
}

function setConn(state, label) {
  if (!connStatus) return;
  connStatus.classList.remove("ok", "err");
  if (state) connStatus.classList.add(state);
  const labelEl = connStatus.querySelector(".conn-label");
  if (labelEl) labelEl.textContent = label;
}

function showFallback(message) {
  if (fallbackEl) fallbackEl.style.display = "flex";
  if (fallbackMsg && message) fallbackMsg.textContent = message;
  setConn("err", "unavailable");
  setHud("status", "unavailable");
}

// ── State ────────────────────────────────────────────────────────────────────
const HOME = { lat: 20, lng: 0, altitude: 0 };
const ORBIT_RANGE = 12_000_000; // metres — a comfortable whole-globe view
const state = {
  map: null,
  Map3DElement: null,
  Polyline3DElement: null,
  Marker3DElement: null,
  geocoder: null,
  focus: { ...HOME },
  rotating: true,
  distanceLine: null,
  distanceMarker: null,
};

// ── Runtime bootstrap: key from /maps/config, then the Maps JS loader ─────────

/**
 * Inject the official Google Maps JS API bootstrap loader with the runtime key.
 * Mirrors Google's inline loader, parameterised by the fetched key. Returns a
 * promise that resolves once `google.maps.importLibrary` is available.
 */
function injectMapsLoader(apiKey) {
  return new Promise((resolve, reject) => {
    if (window.google && window.google.maps && window.google.maps.importLibrary) {
      resolve();
      return;
    }
    const params = new URLSearchParams({
      key: apiKey,
      v: "alpha", // 3D maps (Map3DElement) live on the alpha channel
      libraries: "maps3d,geocoding,marker",
      loading: "async",
    });
    const script = document.createElement("script");
    script.src = "https://maps.googleapis.com/maps/api/js?" + params.toString();
    script.async = true;
    script.defer = true;
    script.onload = () => resolve();
    script.onerror = () => reject(new Error("failed to load Google Maps JS API"));
    document.head.appendChild(script);
  });
}

async function fetchConfig() {
  const resp = await fetch("/maps/config", { headers: { Accept: "application/json" } });
  if (!resp.ok) {
    throw new Error("maps config unavailable (HTTP " + resp.status + ")");
  }
  return resp.json();
}

// ── Globe construction + idle rotation ───────────────────────────────────────

async function buildGlobe() {
  const { Map3DElement } = await window.google.maps.importLibrary("maps3d");
  const maps3d = await window.google.maps.importLibrary("maps3d");
  const geo = await window.google.maps.importLibrary("geocoding");

  state.Map3DElement = Map3DElement;
  state.Polyline3DElement = maps3d.Polyline3DElement || null;
  state.Marker3DElement = maps3d.Marker3DElement || null;
  state.geocoder = new geo.Geocoder();

  const map = new Map3DElement({
    center: { lat: HOME.lat, lng: HOME.lng, altitude: HOME.altitude },
    range: ORBIT_RANGE,
    tilt: 0,
    heading: 0,
    // Photorealistic 3D tiles globe.
    mode: "HYBRID",
  });
  map.id = "map3d-el";
  if (mapHost) {
    mapHost.replaceWith(map);
  } else {
    document.getElementById("stage").appendChild(map);
  }
  state.map = map;

  setConn("ok", "connected");
  setHud("status", "ready");
  startIdleRotation();
}

/** Start the continuous idle orbit (flyCameraAround) loop. */
function startIdleRotation() {
  state.rotating = true;
  setHud("mode", "idle rotate");
  if (rotateBtn) rotateBtn.setAttribute("aria-pressed", "true");
  orbitOnce();
}

function stopIdleRotation() {
  state.rotating = false;
  setHud("mode", "manual");
  if (rotateBtn) rotateBtn.setAttribute("aria-pressed", "false");
  if (state.map && typeof state.map.stopCameraAnimation === "function") {
    state.map.stopCameraAnimation();
  }
}

/** Run one slow 360° orbit around the current focus, then loop while rotating. */
function orbitOnce() {
  if (!state.map || !state.rotating) return;
  if (typeof state.map.flyCameraAround !== "function") return;
  state.map.flyCameraAround({
    camera: {
      center: {
        lat: state.focus.lat,
        lng: state.focus.lng,
        altitude: state.focus.altitude || 0,
      },
      range: ORBIT_RANGE,
      tilt: 45,
    },
    durationMillis: 60000,
    rounds: 1,
  });
  // Chain the next orbit when this one finishes (best-effort; the event name is
  // tolerated to be absent on older channels — then we re-arm on a timer).
  const rearm = () => {
    if (state.rotating) orbitOnce();
  };
  if (typeof state.map.addEventListener === "function") {
    state.map.addEventListener("gmp-animationend", rearm, { once: true });
  } else {
    setTimeout(rearm, 60000);
  }
}

// ── Geocoding + camera moves ─────────────────────────────────────────────────

/** Geocode a free-text place to `{lat, lng}` (or null on failure). */
async function geocode(place) {
  if (!state.geocoder) return null;
  try {
    const { results } = await state.geocoder.geocode({ address: place });
    if (!results || !results.length) return null;
    const loc = results[0].geometry.location;
    return { lat: loc.lat(), lng: loc.lng() };
  } catch (err) {
    return null;
  }
}

/** Google-Earth-style animated fly-in to a place. */
async function flyTo(place) {
  setHud("status", "locating " + place + "…");
  const target = await geocode(place);
  if (!target) {
    toast("Couldn't find “" + place + "”.");
    setHud("status", "ready");
    return;
  }
  stopIdleRotation();
  state.focus = { lat: target.lat, lng: target.lng, altitude: 0 };
  if (state.map && typeof state.map.flyCameraTo === "function") {
    state.map.flyCameraTo({
      endCamera: {
        center: { lat: target.lat, lng: target.lng, altitude: 0 },
        range: 2500,
        tilt: 65,
        heading: 0,
      },
      durationMillis: 5000,
    });
  }
  setHud("mode", "fly to " + place);
  setHud("status", "ready");
}

// ── Distance ─────────────────────────────────────────────────────────────────

/** Great-circle distance (km) between two `{lat, lng}` points (haversine). */
function haversineKm(a, b) {
  const R = 6371;
  const toRad = (d) => (d * Math.PI) / 180;
  const dLat = toRad(b.lat - a.lat);
  const dLng = toRad(b.lng - a.lng);
  const s =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(toRad(a.lat)) * Math.cos(toRad(b.lat)) * Math.sin(dLng / 2) ** 2;
  return 2 * R * Math.asin(Math.min(1, Math.sqrt(s)));
}

function clearDistance() {
  if (state.distanceLine && state.distanceLine.remove) state.distanceLine.remove();
  if (state.distanceMarker && state.distanceMarker.remove) {
    state.distanceMarker.remove();
  }
  state.distanceLine = null;
  state.distanceMarker = null;
}

/** Geocode `place`, draw a line from the current focus, and show the distance. */
async function distanceTo(place) {
  setHud("status", "measuring to " + place + "…");
  const target = await geocode(place);
  if (!target) {
    toast("Couldn't find “" + place + "”.");
    setHud("status", "ready");
    return;
  }
  const from = { lat: state.focus.lat, lng: state.focus.lng };
  const km = haversineKm(from, target);

  clearDistance();
  if (state.Polyline3DElement && state.map) {
    const line = new state.Polyline3DElement({
      coordinates: [
        { lat: from.lat, lng: from.lng, altitude: 0 },
        { lat: target.lat, lng: target.lng, altitude: 0 },
      ],
      strokeColor: "#2b6cff",
      strokeWidth: 8,
      altitudeMode: "CLAMP_TO_GROUND",
      geodesic: true,
    });
    state.map.append(line);
    state.distanceLine = line;
  }
  if (state.Marker3DElement && state.map) {
    const marker = new state.Marker3DElement({
      position: { lat: target.lat, lng: target.lng, altitude: 0 },
      label: place,
    });
    state.map.append(marker);
    state.distanceMarker = marker;
  }

  const pretty = km >= 10 ? Math.round(km).toLocaleString() : km.toFixed(1);
  toast("Distance to " + place + ": " + pretty + " km");
  setHud("mode", "distance " + pretty + " km");
  setHud("status", "ready");
}

// ── Command parsing (shared by typed input + voice) ──────────────────────────

/** Route a free-text command to fly-to / distance-to / rotate / open. */
async function runCommand(raw) {
  const text = (raw || "").trim();
  if (!text) return;
  const lower = text.toLowerCase();

  let m = lower.match(/distance (?:to|from)?\s*(.+)$/);
  if (lower.startsWith("distance") || lower.includes("distance to")) {
    m = lower.match(/distance\s*(?:to|from)?\s*(.+)$/);
    if (m && m[1]) {
      await distanceTo(m[1].trim());
      return;
    }
  }

  m = lower.match(/^(?:fly|go|take me|navigate)\s*(?:to|over)?\s*(.+)$/);
  if (m && m[1]) {
    await flyTo(m[1].trim());
    return;
  }

  if (lower === "open maps" || lower === "open map") {
    toast("Maps is open.");
    return;
  }
  if (lower === "rotate" || lower === "spin") {
    startIdleRotation();
    return;
  }
  if (lower === "stop" || lower === "stop rotating") {
    stopIdleRotation();
    return;
  }

  // Bare place name -> fly there.
  await flyTo(text);
}

// ── Web Speech mic ───────────────────────────────────────────────────────────

let recognition = null;
let listening = false;

function setupSpeech() {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) {
    setHud("voice", "unsupported");
    if (micBtn) micBtn.disabled = true;
    return;
  }
  recognition = new SR();
  recognition.continuous = true;
  recognition.interimResults = false;
  recognition.lang = "en-US";

  recognition.onresult = (event) => {
    const last = event.results[event.results.length - 1];
    const phrase = last[0].transcript.trim();
    setHud("heard", phrase);
    runCommand(phrase);
  };
  recognition.onerror = () => {
    setHud("voice", "error");
  };
  recognition.onend = () => {
    listening = false;
    setHud("voice", "off");
    if (micBtn) micBtn.setAttribute("aria-pressed", "false");
  };
}

function toggleMic() {
  if (!recognition) return;
  if (listening) {
    recognition.stop();
    return;
  }
  try {
    recognition.start();
    listening = true;
    setHud("voice", "listening");
    if (micBtn) micBtn.setAttribute("aria-pressed", "true");
  } catch (err) {
    toast("Microphone unavailable.");
  }
}

// ── Deep-link query params (?fly= / ?to=) ────────────────────────────────────

async function handleDeepLink() {
  const params = new URLSearchParams(window.location.search);
  const fly = params.get("fly");
  const to = params.get("to");
  if (fly) await flyTo(fly);
  if (to) await distanceTo(to);
}

// ── Wiring ───────────────────────────────────────────────────────────────────

function wireControls() {
  if (cmdForm) {
    cmdForm.addEventListener("submit", (event) => {
      event.preventDefault();
      const value = cmdInput ? cmdInput.value : "";
      runCommand(value);
      if (cmdInput) cmdInput.value = "";
    });
  }
  if (micBtn) micBtn.addEventListener("click", toggleMic);
  if (rotateBtn) {
    rotateBtn.addEventListener("click", () => {
      if (state.rotating) stopIdleRotation();
      else startIdleRotation();
    });
  }
}

async function main() {
  wireControls();
  setupSpeech();

  let config;
  try {
    config = await fetchConfig();
  } catch (err) {
    showFallback("Maps is disabled.");
    return;
  }
  if (!config || !config.enabled || !config.apiKey) {
    showFallback("No Google Maps API key configured.");
    return;
  }

  try {
    await injectMapsLoader(config.apiKey);
    await buildGlobe();
  } catch (err) {
    showFallback("Google Maps failed to load.");
    return;
  }

  await handleDeepLink();
}

// Kick off once the DOM is parsed (the module is deferred, so it already is).
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", main, { once: true });
} else {
  main();
}

// Exported for unit reasoning / potential test harnesses; harmless in-browser.
export { haversineKm, runCommand };
