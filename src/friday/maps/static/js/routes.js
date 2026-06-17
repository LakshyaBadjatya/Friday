/*
 * FRIDAY Maps — directions / routes (browser ES module), OSRM.
 *
 * "route from London to Paris": geocode both ends (Nominatim via /maps/geocode),
 * fetch a driving route from the free OSRM service via /maps/route, draw it as a
 * line, frame it, and report distance + duration. No Google Directions, no key.
 * The public OSRM demo serves the driving profile only, so the requested mode is
 * noted but not honoured.
 */

"use strict";

import { setHud, toast, showPanel } from "./ui.js";
import { geocode, addLine, addMarker, clearOverlays, fitTo } from "./globe.js";

export async function showRoute(from, to, mode) {
  setHud("status", "routing " + from + " → " + to + "…");

  const [origin, dest] = await Promise.all([geocode(from), geocode(to)]);
  if (!origin || !dest) {
    toast("Couldn't locate " + (!origin ? "“" + from + "”" : "“" + to + "”") + ".");
    setHud("status", "ready");
    return;
  }

  let route;
  try {
    const params =
      "from_lat=" + origin.lat + "&from_lng=" + origin.lng + "&to_lat=" + dest.lat + "&to_lng=" + dest.lng;
    const resp = await fetch("/maps/route?" + params, { headers: { Accept: "application/json" } });
    if (!resp.ok) {
      toast("No route found from “" + from + "” to “" + to + "”.");
      setHud("status", "ready");
      return;
    }
    route = await resp.json();
  } catch (_e) {
    toast("Routing failed.");
    setHud("status", "ready");
    return;
  }

  const coords = Array.isArray(route.coordinates) ? route.coordinates : [];
  if (!coords.length) {
    toast("No route geometry returned.");
    setHud("status", "ready");
    return;
  }

  clearOverlays();
  // OSRM coordinates are GeoJSON [lng, lat]; addLine wants {lat, lng}.
  addLine(coords.map(([lng, lat]) => ({ lat, lng })), { strokeColor: "#16c784", strokeWidth: 5 });
  addMarker(origin, from);
  addMarker(dest, to);
  fitTo([origin, dest]);

  showPanel({
    title: from + " → " + to,
    rows: [
      ["Mode", "drive" + (mode && mode.toLowerCase() !== "drive" ? " (OSRM is drive-only)" : "")],
      ["Distance", route.distance_km != null ? route.distance_km + " km" : ""],
      ["Duration", route.duration_min != null ? route.duration_min + " min" : ""],
    ],
  });
  toast("Route: " + route.distance_km + " km · " + route.duration_min + " min");
  setHud("mode", "route " + route.distance_km + " km");
  setHud("status", "ready");
}
