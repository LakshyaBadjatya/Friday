/*
 * FRIDAY service worker — the offline shell.
 *
 * Served from "/service-worker.js" (the root) so its default control scope is
 * the whole origin: it can intercept navigations for the dashboard/HUD, not just
 * a sub-path. It precaches a tiny app shell on install and, for navigations,
 * uses a network-first strategy with the cached "/offline.html" page as the
 * fallback when the network is down. Non-navigation GETs fall back to whatever
 * is in the cache. Everything is vanilla — no bundler, no eval of server output.
 */

"use strict";

// Bump this version to invalidate the precache when the shell changes.
const CACHE_VERSION = "friday-pwa-v1";

// The minimal shell precached on install. "/hud" is the start_url; the manifest
// and offline page complete the installable shell.
const SHELL_ASSETS = [
  "/hud",
  "/offline.html",
  "/manifest.webmanifest",
];

// The page shown when a navigation fails offline.
const OFFLINE_URL = "/offline.html";

self.addEventListener("install", (event) => {
  event.waitUntil(
    (async () => {
      const cache = await caches.open(CACHE_VERSION);
      // addAll is atomic; if any asset 404s the install fails loudly. The shell
      // assets are all same-origin static routes, so this is safe.
      await cache.addAll(SHELL_ASSETS);
      // Activate this worker immediately rather than waiting for old tabs to close.
      await self.skipWaiting();
    })(),
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    (async () => {
      // Drop any stale precaches from a previous version.
      const keys = await caches.keys();
      await Promise.all(
        keys.map((key) => (key === CACHE_VERSION ? null : caches.delete(key))),
      );
      // Take control of all open clients without a reload.
      await self.clients.claim();
    })(),
  );
});

self.addEventListener("fetch", (event) => {
  const request = event.request;

  // Only handle same-origin GETs; let everything else hit the network untouched.
  if (request.method !== "GET") {
    return;
  }
  const url = new URL(request.url);
  if (url.origin !== self.location.origin) {
    return;
  }

  // Navigations: network-first, fall back to the cached page, then the offline shell.
  if (request.mode === "navigate") {
    event.respondWith(
      (async () => {
        try {
          const networkResponse = await fetch(request);
          return networkResponse;
        } catch (err) {
          const cache = await caches.open(CACHE_VERSION);
          const cached = await cache.match(request);
          if (cached) {
            return cached;
          }
          const offline = await cache.match(OFFLINE_URL);
          if (offline) {
            return offline;
          }
          throw err;
        }
      })(),
    );
    return;
  }

  // Static assets: cache-first with a network fallback (and cache-on-success).
  event.respondWith(
    (async () => {
      const cache = await caches.open(CACHE_VERSION);
      const cached = await cache.match(request);
      if (cached) {
        return cached;
      }
      try {
        const networkResponse = await fetch(request);
        if (networkResponse && networkResponse.ok) {
          cache.put(request, networkResponse.clone());
        }
        return networkResponse;
      } catch (err) {
        const offline = await cache.match(OFFLINE_URL);
        if (offline) {
          return offline;
        }
        throw err;
      }
    })(),
  );
});
