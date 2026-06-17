/*
 * FRIDAY Maps — Directions / routes (browser ES module).
 *
 * "route from London to Paris [by car|walk|transit|bike]": run the classic
 * DirectionsService (the `routes` library), render the overview path as a 3D
 * polyline on the globe, frame it, and report total distance + duration.
 * Degrades to a friendly toast when the `routes` library did not load.
 */

"use strict";

import { state, setHud, toast, showPanel } from "./ui.js";
import { addLine, clearOverlays, fitTo } from "./globe.js";

/** Map a spoken/typed mode word to a google.maps.TravelMode value. */
function travelMode(word) {
  const tm = (window.google && window.google.maps && window.google.maps.TravelMode) || {};
  switch ((word || "").toLowerCase()) {
    case "walk":
    case "walking":
    case "foot":
      return tm.WALKING || "WALKING";
    case "transit":
    case "bus":
    case "train":
      return tm.TRANSIT || "TRANSIT";
    case "bike":
    case "bicycle":
    case "cycling":
      return tm.BICYCLING || "BICYCLING";
    default:
      return tm.DRIVING || "DRIVING";
  }
}

/** Pull the overview path (array of {lat,lng}) from a DirectionsRoute. */
function pathOf(route) {
  const raw = route.overview_path || [];
  return raw
    .map((ll) => ({
      lat: typeof ll.lat === "function" ? ll.lat() : ll.lat,
      lng: typeof ll.lng === "function" ? ll.lng() : ll.lng,
    }))
    .filter((c) => typeof c.lat === "number" && typeof c.lng === "number");
}

/** Compute total distance/duration text by summing the legs. */
function legTotals(route) {
  let meters = 0;
  let seconds = 0;
  for (const leg of route.legs || []) {
    if (leg.distance) meters += leg.distance.value || 0;
    if (leg.duration) seconds += leg.duration.value || 0;
  }
  const km = (meters / 1000).toFixed(meters >= 100000 ? 0 : 1);
  const mins = Math.round(seconds / 60);
  const dur = mins >= 60 ? Math.floor(mins / 60) + "h " + (mins % 60) + "m" : mins + " min";
  return { km, dur };
}

/**
 * Route from `from` to `to` (free-text places) by `mode`, draw it, frame it,
 * and show distance + ETA.
 */
export async function showRoute(from, to, mode) {
  if (!state.routes || typeof state.routes.DirectionsService !== "function") {
    toast("Directions are unavailable (enable the Routes/Directions API).");
    return;
  }
  setHud("status", "routing " + from + " → " + to + "…");
  let result;
  try {
    const svc = new state.routes.DirectionsService();
    result = await svc.route({
      origin: from,
      destination: to,
      travelMode: travelMode(mode),
    });
  } catch (_e) {
    toast("Couldn't find a route from “" + from + "” to “" + to + "”.");
    setHud("status", "ready");
    return;
  }

  const route = result && result.routes && result.routes[0];
  const path = route ? pathOf(route) : [];
  if (!path.length) {
    toast("No route found.");
    setHud("status", "ready");
    return;
  }

  clearOverlays();
  addLine(path, { strokeColor: "#16c784", strokeWidth: 10 });
  fitTo([path[0], path[path.length - 1]]);

  const { km, dur } = legTotals(route);
  showPanel({
    title: from + " → " + to,
    rows: [
      ["Mode", (mode || "drive")],
      ["Distance", km + " km"],
      ["Duration", dur],
    ],
  });
  toast("Route: " + km + " km · " + dur);
  setHud("mode", "route " + km + " km");
  setHud("status", "ready");
}
