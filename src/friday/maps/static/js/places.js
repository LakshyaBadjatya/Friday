/*
 * FRIDAY Maps — Places search (browser ES module).
 *
 * "find coffee near Tokyo" / "search museums here": a Places (New) text search
 * biased to the current focus. Each result becomes a labeled 3D marker; clicking
 * a marker opens its details panel. Degrades gracefully (a friendly toast) when
 * the `places` library did not load.
 */

"use strict";

import { state, setHud, toast, showPanel } from "./ui.js";
import { addMarker, clearOverlays, flyToCoords, fitTo } from "./globe.js";

/** Read a LatLng-ish value whose lat/lng may be numbers or accessor functions. */
function coordOf(loc) {
  if (!loc) return null;
  const lat = typeof loc.lat === "function" ? loc.lat() : loc.lat;
  const lng = typeof loc.lng === "function" ? loc.lng() : loc.lng;
  if (typeof lat !== "number" || typeof lng !== "number") return null;
  return { lat, lng };
}

function detailsRows(place, coords) {
  return [
    ["Address", place.formattedAddress || ""],
    ["Rating", place.rating != null ? place.rating + " ★" : ""],
    ["Lat", coords ? coords.lat.toFixed(5) : ""],
    ["Lng", coords ? coords.lng.toFixed(5) : ""],
  ];
}

/**
 * Text-search `query` around the current focus, drop markers, and frame them.
 * `query` is the full search text (e.g. "coffee", "museums in Rome").
 */
export async function searchPlaces(query) {
  if (!state.places || !state.places.Place || typeof state.places.Place.searchByText !== "function") {
    toast("Places search is unavailable (enable the Places API).");
    return;
  }
  setHud("status", "searching “" + query + "”…");
  let results;
  try {
    const out = await state.places.Place.searchByText({
      textQuery: query,
      fields: ["displayName", "formattedAddress", "location", "rating"],
      maxResultCount: 8,
      locationBias: {
        center: { lat: state.focus.lat, lng: state.focus.lng },
        radius: 50000,
      },
    });
    results = out && out.places ? out.places : [];
  } catch (_e) {
    toast("Places search failed.");
    setHud("status", "ready");
    return;
  }

  if (!results.length) {
    toast("No places found for “" + query + "”.");
    setHud("status", "ready");
    return;
  }

  clearOverlays();
  const points = [];
  for (const place of results) {
    const coords = coordOf(place.location);
    if (!coords) continue;
    points.push(coords);
    const name =
      (place.displayName && (place.displayName.text || place.displayName)) || "place";
    const marker = addMarker(coords, String(name));
    if (marker && typeof marker.addEventListener === "function") {
      marker.addEventListener("gmp-click", () =>
        showPanel({ title: String(name), rows: detailsRows(place, coords) })
      );
    }
  }

  if (points.length === 1) flyToCoords(points[0], { range: 4000, tilt: 60 });
  else if (points.length > 1) fitTo(points);

  toast(points.length + " result" + (points.length === 1 ? "" : "s") + " for “" + query + "”.");
  setHud("mode", "places: " + query);
  setHud("status", "ready");
}
