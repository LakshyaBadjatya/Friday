/*
 * FRIDAY Maps — entrypoint (no build step, browser ES module).
 *
 * Thin bootstrap + UI wiring. The map is MapLibre GL + OpenStreetMap (keyless,
 * no paid API). The behaviour lives in focused modules under `./js/`:
 *   ui.js       — shared state + HUD/toast/panel/saved-places helpers
 *   globe.js    — MapLibre globe, camera, overlays, geocode, fly/distance, click-id
 *   places.js   — Nominatim place search + markers
 *   routes.js   — OSRM driving route + distance/ETA
 *   weather.js  — /maps/weather card + "fly to my location"
 *   tour.js     — multi-stop fly-through + saved-places chips
 *   commands.js — pure text→intent parser + dispatcher
 *
 * Responsibilities here: confirm the surface is enabled via `/maps/config`
 * (no key — MapLibre needs none), build the globe, wire the controls + Web-Speech
 * mic, refresh saved places, and run `?fly=`/`?to=`/`?cmd=` deep-links. Defensive
 * throughout — a disabled backend or load failure shows a friendly fallback
 * rather than a black page.
 */

"use strict";

import { $, setHud, showFallback, toast } from "./js/ui.js";
import {
  buildGlobe,
  startIdleRotation,
  stopIdleRotation,
  home,
  zoomBy,
  tiltBy,
  flyTo,
  distanceTo,
  haversineKm,
} from "./js/globe.js";
import { runCommand, parseCommand } from "./js/commands.js";
import { refreshSavedPlaces, savePlace } from "./js/tour.js";

// ── Runtime bootstrap ─────────────────────────────────────────────────────────
// MapLibre GL itself is loaded from a CDN <script> in index.html (keyless), so
// there is no loader to inject here — buildGlobe() just uses window.maplibregl.

async function fetchConfig() {
  const resp = await fetch("/maps/config", { headers: { Accept: "application/json" } });
  if (!resp.ok) throw new Error("maps config unavailable (HTTP " + resp.status + ")");
  return resp.json();
}

// ── Web Speech mic ────────────────────────────────────────────────────────────
let recognition = null;
let listening = false;

function setupSpeech() {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  const micBtn = $("mic");
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
  recognition.onerror = () => setHud("voice", "error");
  recognition.onend = () => {
    listening = false;
    setHud("voice", "off");
    if (micBtn) micBtn.setAttribute("aria-pressed", "false");
  };
}

function toggleMic() {
  if (!recognition) return;
  const micBtn = $("mic");
  if (listening) {
    recognition.stop();
    return;
  }
  try {
    recognition.start();
    listening = true;
    setHud("voice", "listening");
    if (micBtn) micBtn.setAttribute("aria-pressed", "true");
  } catch (_e) {
    toast("Microphone unavailable.");
  }
}

// ── Control wiring ────────────────────────────────────────────────────────────
function wireControls() {
  const cmdForm = $("cmd-form");
  const cmdInput = $("cmd");
  if (cmdForm) {
    cmdForm.addEventListener("submit", (event) => {
      event.preventDefault();
      runCommand(cmdInput ? cmdInput.value : "");
      if (cmdInput) cmdInput.value = "";
    });
  }
  const mic = $("mic");
  if (mic) mic.addEventListener("click", toggleMic);

  const rotate = $("rotate");
  if (rotate) {
    rotate.addEventListener("click", () => {
      if (rotate.getAttribute("aria-pressed") === "true") stopIdleRotation();
      else startIdleRotation();
    });
  }

  const bind = (id, fn) => {
    const el = $(id);
    if (el) el.addEventListener("click", fn);
  };
  bind("home", () => home());
  bind("zoom-in", () => zoomBy(0.5));
  bind("zoom-out", () => zoomBy(2));
  bind("tilt-up", () => tiltBy(15));
  bind("tilt-down", () => tiltBy(-15));
  bind("save", () => savePlace(""));
}

// ── Deep links ────────────────────────────────────────────────────────────────
async function handleDeepLink() {
  const params = new URLSearchParams(window.location.search);
  const cmd = params.get("cmd");
  const fly = params.get("fly");
  const to = params.get("to");
  if (cmd) await runCommand(cmd);
  if (fly) await flyTo(fly);
  if (to) await distanceTo(to);
}

// ── Main ──────────────────────────────────────────────────────────────────────
async function main() {
  wireControls();
  setupSpeech();
  refreshSavedPlaces();

  let config;
  try {
    config = await fetchConfig();
  } catch (_e) {
    showFallback("Maps is disabled.");
    return;
  }
  if (!config || !config.enabled) {
    showFallback("Maps is disabled.");
    return;
  }
  if (!window.maplibregl) {
    showFallback("Map library failed to load.");
    return;
  }

  try {
    await buildGlobe();
  } catch (_e) {
    showFallback("The map failed to load.");
    return;
  }

  await handleDeepLink();
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", main, { once: true });
} else {
  main();
}

// Exported for unit reasoning / potential test harnesses; harmless in-browser.
export { haversineKm, runCommand, parseCommand };
