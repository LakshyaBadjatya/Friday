/*
 * FRIDAY Maps — multi-stop tour + saved places (browser ES module).
 *
 * "tour Paris, Rome, Cairo": geocode each stop, mark them, draw the connecting
 * legs, then fly through them in sequence. Saved places persist in localStorage
 * and render as quick-fly chips.
 */

"use strict";

import { state, setHud, toast, renderSavedPlaces } from "./ui.js";
import { geocode, addMarker, addLine, clearOverlays, flyToCoords, fitTo, flyTo } from "./globe.js";

const SAVED_KEY = "friday.maps.saved";
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

// ── Tour ──────────────────────────────────────────────────────────────────────
/** Fly through `stops` (array of place strings) in order, marking each. */
export async function runTour(stops) {
  const places = stops.map((s) => s.trim()).filter(Boolean);
  if (places.length < 2) {
    toast("A tour needs at least two stops.");
    return;
  }
  setHud("status", "planning tour…");

  // Geocode all up front so a single failure doesn't strand a half-tour.
  const located = [];
  for (const place of places) {
    const coords = await geocode(place);
    if (coords) located.push({ place, coords });
  }
  if (located.length < 2) {
    toast("Couldn't locate enough of those stops.");
    setHud("status", "ready");
    return;
  }

  clearOverlays();
  for (const stop of located) addMarker(stop.coords, stop.place);
  addLine(located.map((s) => s.coords), { strokeColor: "#ffaf3b", strokeWidth: 8 });
  fitTo(located.map((s) => s.coords));
  await sleep(2500);

  for (let i = 0; i < located.length; i += 1) {
    const stop = located[i];
    setHud("mode", "tour " + (i + 1) + "/" + located.length + ": " + stop.place);
    flyToCoords(stop.coords, { range: 3000, tilt: 60, durationMillis: 4000 });
    await sleep(5200);
  }
  toast("Tour complete: " + located.map((s) => s.place).join(" → "));
  setHud("status", "ready");
}

// ── Saved places (localStorage) ────────────────────────────────────────────────
function loadSaved() {
  try {
    const raw = window.localStorage.getItem(SAVED_KEY);
    const list = raw ? JSON.parse(raw) : [];
    return Array.isArray(list) ? list.filter((s) => typeof s === "string") : [];
  } catch (_e) {
    return [];
  }
}

function persist(list) {
  try {
    window.localStorage.setItem(SAVED_KEY, JSON.stringify(list.slice(0, 50)));
  } catch (_e) {
    /* storage full / disabled — chips just won't persist */
  }
}

/** Render the saved-places chips; clicking one flies there. */
export function refreshSavedPlaces() {
  renderSavedPlaces(loadSaved(), (place) => flyTo(place));
}

/** Save `place` (de-duplicated, newest first) and re-render the chips. */
export function savePlace(place) {
  const name = (place || "").trim() || (state.focus && "current view");
  if (!name) return;
  const list = loadSaved().filter((s) => s.toLowerCase() !== name.toLowerCase());
  list.unshift(name);
  persist(list);
  refreshSavedPlaces();
  toast("Saved “" + name + "”.");
}
