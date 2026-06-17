/*
 * FRIDAY Maps — UI + shared state (no build step, browser ES module).
 *
 * Owns the small pieces every other module needs:
 *   - `state`: the single shared mutable globe state (map handle, 3D element
 *     classes, current focus, idle-rotation flag, live overlays).
 *   - DOM/HUD/toast/fallback helpers and the slide-in info panel + saved-places
 *     chip strip used by the feature modules.
 *
 * Pure DOM only — imports nothing, so it can never create an import cycle.
 */

"use strict";

// ── Shared constants + state ──────────────────────────────────────────────────
export const HOME = { lat: 20, lng: 0, altitude: 0 };
export const ORBIT_RANGE = 12_000_000; // metres — a comfortable whole-globe view

export const state = {
  map: null,
  Map3DElement: null,
  Polyline3DElement: null,
  Marker3DElement: null,
  geocoder: null,
  places: null, // Places library namespace (when available)
  routes: null, // Routes/Directions library namespace (when available)
  focus: { ...HOME },
  rotating: true,
  overlays: [], // every Marker3D / Polyline3D we appended, so we can clear them
};

// ── DOM handles ───────────────────────────────────────────────────────────────
export const $ = (id) => document.getElementById(id);

const hud = {
  mode: () => $("hud-mode"),
  voice: () => $("hud-voice"),
  heard: () => $("hud-heard"),
  status: () => $("hud-status"),
};

// ── HUD / toast / connection / fallback ───────────────────────────────────────
export function setHud(key, text) {
  const el = hud[key] && hud[key]();
  if (el) el.textContent = text;
}

let toastTimer = null;
export function toast(message) {
  const el = $("toast");
  if (!el) return;
  el.textContent = message;
  el.hidden = false;
  if (toastTimer) clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    el.hidden = true;
  }, 4200);
}

export function setConn(stateName, label) {
  const el = $("conn-status");
  if (!el) return;
  el.classList.remove("ok", "err");
  if (stateName) el.classList.add(stateName);
  const labelEl = el.querySelector(".conn-label");
  if (labelEl) labelEl.textContent = label;
}

export function showFallback(message) {
  const el = $("fallback");
  const msg = $("fallback-msg");
  if (el) el.style.display = "flex";
  if (msg && message) msg.textContent = message;
  setConn("err", "unavailable");
  setHud("status", "unavailable");
}

// ── Slide-in info panel (place details / weather card) ────────────────────────
/**
 * Render an info panel from a `{title, rows}` model. `rows` is a list of
 * `[label, value]` pairs; falsy values are skipped. Escapes all text — the panel
 * never injects HTML from geocoded/searched/weather data.
 */
export function showPanel(model) {
  const panel = $("panel");
  if (!panel) return;
  const esc = (s) =>
    String(s).replace(/[&<>"']/g, (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]
    );
  const rows = (model.rows || [])
    .filter(([, v]) => v !== undefined && v !== null && v !== "")
    .map(([k, v]) => `<div class="panel-row"><span>${esc(k)}</span><b>${esc(v)}</b></div>`)
    .join("");
  panel.innerHTML =
    `<button class="panel-close" aria-label="close" id="panel-close">×</button>` +
    `<h2>${esc(model.title || "")}</h2>${rows}`;
  panel.hidden = false;
  const close = $("panel-close");
  if (close) close.addEventListener("click", hidePanel, { once: true });
}

export function hidePanel() {
  const panel = $("panel");
  if (panel) panel.hidden = true;
}

// ── Saved-places chip strip ───────────────────────────────────────────────────
/**
 * Render the saved-places chips. `places` is a list of strings; `onPick` is
 * called with the chosen place when a chip is clicked.
 */
export function renderSavedPlaces(places, onPick) {
  const host = $("saved");
  if (!host) return;
  host.innerHTML = "";
  for (const place of places) {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "chip";
    chip.textContent = place;
    chip.addEventListener("click", () => onPick(place));
    host.appendChild(chip);
  }
  host.hidden = places.length === 0;
}
