/*
 * FRIDAY Maps — command parsing + dispatch (browser ES module).
 *
 * `parseCommand(text)` is a PURE function (no DOM, no globe) that maps free text
 * (typed or spoken) to an intent object — kept pure so its routing is reviewable
 * in isolation. `runCommand(text)` dispatches the parsed intent to the feature
 * modules. Order matters: more specific intents (route/distance/weather/…) are
 * matched before the bare "fly there" fallback.
 */

"use strict";

import { flyTo, distanceTo, startIdleRotation, stopIdleRotation, home, zoomBy, tiltBy } from "./globe.js";
import { searchPlaces } from "./places.js";
import { showRoute } from "./routes.js";
import { showWeather, flyToMyLocation } from "./weather.js";
import { runTour, savePlace } from "./tour.js";

/** Split a free-text stop list ("A, B and C" / "A -> B -> C") into trimmed stops. */
function splitStops(text) {
  return text
    .split(/\s*(?:,|->|→|\band\b)\s*/i)
    .map((s) => s.trim())
    .filter(Boolean);
}

/**
 * Map free text to `{intent, ...args}`. Pure: same input → same output, no
 * side effects. Intents: route, distance, search, weather, mylocation, tour,
 * save, rotate, stop, home, zoom, tilt, open, fly.
 */
export function parseCommand(raw) {
  const text = (raw || "").trim();
  if (!text) return { intent: "none" };

  let m = text.match(/^(?:route|directions)\s+from\s+(.+?)\s+to\s+(.+?)(?:\s+(?:by|via)\s+(\w+))?$/i);
  if (m) return { intent: "route", from: m[1].trim(), to: m[2].trim(), mode: (m[3] || "").trim() };

  if (/^(?:fly|go|take me)\s+to\s+my location$|^where am i\??$|^locate me$|^my location$/i.test(text)) {
    return { intent: "mylocation" };
  }

  m = text.match(/^(?:show\s+)?distance\s*(?:to|from)?\s*(.+)$/i);
  if (m) return { intent: "distance", place: m[1].trim() };

  m = text.match(/^(?:find|search|places?)\s+(.+)$/i);
  if (m) return { intent: "search", query: m[1].trim() };

  m = text.match(/^weather\s*(?:in|at|for)?\s*(.+)$/i);
  if (m) return { intent: "weather", place: m[1].trim() };

  m = text.match(/^tour\s*(?:of|through)?\s*(.+)$/i);
  if (m) return { intent: "tour", stops: splitStops(m[1]) };

  m = text.match(/^save\s*(?:this|here)?\s*(.*)$/i);
  if (m) return { intent: "save", place: m[1].trim() };

  if (/^(?:rotate|spin)$/i.test(text)) return { intent: "rotate" };
  if (/^stop(?:\s+rotating)?$/i.test(text)) return { intent: "stop" };
  if (/^(?:home|reset)$/i.test(text)) return { intent: "home" };
  if (/^zoom\s+in$/i.test(text)) return { intent: "zoom", factor: 0.5 };
  if (/^zoom\s+out$/i.test(text)) return { intent: "zoom", factor: 2 };
  if (/^tilt\s+up$/i.test(text)) return { intent: "tilt", delta: 15 };
  if (/^tilt\s+down$/i.test(text)) return { intent: "tilt", delta: -15 };
  if (/^open maps?$/i.test(text)) return { intent: "open" };

  m = text.match(/^(?:fly|go|take me|navigate)\s*(?:to|over)?\s*(.+)$/i);
  if (m) return { intent: "fly", place: m[1].trim() };

  return { intent: "fly", place: text }; // bare place name → fly there
}

/** Parse and execute a command. Returns the parsed intent (handy for callers/tests). */
export async function runCommand(raw) {
  const cmd = parseCommand(raw);
  switch (cmd.intent) {
    case "route":
      await showRoute(cmd.from, cmd.to, cmd.mode);
      break;
    case "distance":
      await distanceTo(cmd.place);
      break;
    case "search":
      await searchPlaces(cmd.query);
      break;
    case "weather":
      await showWeather(cmd.place);
      break;
    case "mylocation":
      flyToMyLocation();
      break;
    case "tour":
      await runTour(cmd.stops);
      break;
    case "save":
      savePlace(cmd.place);
      break;
    case "rotate":
      startIdleRotation();
      break;
    case "stop":
      stopIdleRotation();
      break;
    case "home":
      home();
      break;
    case "zoom":
      zoomBy(cmd.factor);
      break;
    case "tilt":
      tiltBy(cmd.delta);
      break;
    case "open":
      break; // already open; no-op
    case "fly":
      await flyTo(cmd.place);
      break;
    default:
      break;
  }
  return cmd;
}
