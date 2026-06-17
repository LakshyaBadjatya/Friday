/*
 * FRIDAY Maps — entrypoint (no build step, browser ES module).
 *
 * Thin bootstrap + UI wiring. The behaviour lives in focused modules under
 * `./js/`:
 *   ui.js       — shared state + HUD/toast/panel/saved-places helpers
 *   globe.js    — Map3DElement, camera, overlays, geocode, fly/distance, click-id
 *   places.js   — Places (New) search + POI markers
 *   routes.js   — Directions → 3D route polyline + distance/ETA
 *   weather.js  — /maps/weather card + "fly to my location"
 *   tour.js     — multi-stop fly-through + saved-places chips
 *   commands.js — pure text→intent parser + dispatcher
 *
 * Responsibilities here: fetch the runtime key from `/maps/config`, inject the
 * official Maps JS loader (with all libraries), build the globe, wire the
 * controls + Web-Speech mic, refresh saved places, and run `?fly=`/`?to=`/`?cmd=`
 * deep-links. Defensive throughout — a disabled backend or load failure shows a
 * friendly fallback rather than a black page.
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
function injectMapsLoader(apiKey) {
  return new Promise((resolve, reject) => {
    if (window.google && window.google.maps && window.google.maps.importLibrary) {
      resolve();
      return;
    }
    const params = new URLSearchParams({
      key: apiKey,
      v: "alpha", // 3D maps (Map3DElement) live on the alpha channel
      libraries: "maps3d,geocoding,marker,places,routes",
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
  if (!config || !config.enabled || !config.apiKey) {
    showFallback("No Google Maps API key configured.");
    return;
  }

  try {
    await injectMapsLoader(config.apiKey);
    await buildGlobe();
  } catch (_e) {
    showFallback("Google Maps failed to load.");
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
