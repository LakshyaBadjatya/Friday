/*
 * FRIDAY Maps — place search (browser ES module), OpenStreetMap / Nominatim.
 *
 * "find coffee near Tokyo" / "search museums" → a free Nominatim text search via
 * FRIDAY's /maps/geocode proxy. Each result becomes a labeled marker; clicking a
 * marker opens its details panel. No Google Places, no key, no billing.
 * (Nominatim has no ratings, so the panel shows name + coordinates.)
 */

"use strict";

import { setHud, toast, showPanel } from "./ui.js";
import { addMarker, clearOverlays, flyToCoords, fitTo } from "./globe.js";

export async function searchPlaces(query) {
  setHud("status", "searching “" + query + "”…");
  let results;
  try {
    const resp = await fetch("/maps/geocode?limit=8&q=" + encodeURIComponent(query), {
      headers: { Accept: "application/json" },
    });
    if (!resp.ok) {
      toast("Search failed for “" + query + "”.");
      setHud("status", "ready");
      return;
    }
    const data = await resp.json();
    results = Array.isArray(data.results) ? data.results : [];
  } catch (_e) {
    toast("Search failed.");
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
    if (typeof place.lat !== "number" || typeof place.lng !== "number") continue;
    const coords = { lat: place.lat, lng: place.lng };
    points.push(coords);
    const name = place.name || "place";
    addMarker(coords, name, () =>
      showPanel({
        title: name.split(",")[0],
        rows: [
          ["Address", name],
          ["Lat", coords.lat.toFixed(5)],
          ["Lng", coords.lng.toFixed(5)],
        ],
      })
    );
  }

  if (points.length === 1) flyToCoords(points[0], { zoom: 13, pitch: 50 });
  else if (points.length > 1) fitTo(points);

  toast(points.length + " result" + (points.length === 1 ? "" : "s") + " for “" + query + "”.");
  setHud("mode", "places: " + query);
  setHud("status", "ready");
}
