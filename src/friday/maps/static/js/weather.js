/*
 * FRIDAY Maps — weather + my-location (browser ES module).
 *
 * "weather in Tokyo": fetch the keyless conditions from the FRIDAY backend
 * (GET /maps/weather) — never a Google API — fly there, and show a weather card.
 * "fly to my location" / "where am I": use the browser geolocation API
 * (permission-guarded), fly in, and label where you are.
 */

"use strict";

import { setHud, toast, showPanel } from "./ui.js";
import { flyTo, flyToCoords, addMarker, reverseGeocode } from "./globe.js";

/** Fetch + show current weather for `place`, and fly there. */
export async function showWeather(place) {
  setHud("status", "weather for " + place + "…");
  let card;
  try {
    const resp = await fetch("/maps/weather?location=" + encodeURIComponent(place), {
      headers: { Accept: "application/json" },
    });
    if (!resp.ok) {
      toast("Weather unavailable for “" + place + "”.");
      setHud("status", "ready");
      return;
    }
    card = await resp.json();
  } catch (_e) {
    toast("Weather lookup failed.");
    setHud("status", "ready");
    return;
  }

  showPanel({
    title: "Weather — " + (card.location || place),
    rows: [
      ["Now", card.description],
      ["Temp", card.temp_c != null ? card.temp_c + " °C" : ""],
      ["Feels like", card.feels_like_c != null ? card.feels_like_c + " °C" : ""],
      ["Humidity", card.humidity_pct != null ? card.humidity_pct + " %" : ""],
      ["Wind", card.wind_kph != null ? card.wind_kph + " km/h" : ""],
    ],
  });
  // Fly there too (best-effort; the card already showed regardless of geocode).
  await flyTo(card.location || place);
  setHud("mode", "weather: " + (card.location || place));
  setHud("status", "ready");
}

/** Fly to the browser's current geolocation (permission-guarded). */
export function flyToMyLocation() {
  if (!navigator.geolocation) {
    toast("Geolocation is not available in this browser.");
    return;
  }
  setHud("status", "locating you…");
  navigator.geolocation.getCurrentPosition(
    async (pos) => {
      const coords = { lat: pos.coords.latitude, lng: pos.coords.longitude };
      flyToCoords(coords, { range: 6000, tilt: 60 });
      const address = await reverseGeocode(coords);
      addMarker(coords, address || "You are here");
      setHud("mode", "my location");
      setHud("status", "ready");
    },
    (_err) => {
      toast("Couldn't get your location (permission denied or unavailable).");
      setHud("status", "ready");
    },
    { enableHighAccuracy: true, timeout: 10000, maximumAge: 60000 }
  );
}
