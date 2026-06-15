"""The PWA feature — an always-available, no-build installable app shell.

A Progressive Web App shell that makes FRIDAY installable and offline-capable.
It is static assets only (a web app manifest, a root-scope service worker, and an
offline fallback page) with NO feature flag: installing a PWA is harmless and the
shell carries no secrets, so it is always reachable.

The three surfaces are served at ROOT scope (no ``/pwa`` prefix) — the service
worker in particular MUST live at ``/service-worker.js`` so its default control
scope covers the whole origin (the dashboard/HUD), not a sub-path:

* ``GET /manifest.webmanifest`` — the installable ``FRIDAY`` manifest
  (``display: standalone``, ``start_url: /hud``).
* ``GET /service-worker.js`` — precaches the shell + serves ``/offline.html`` when
  a navigation fails offline.
* ``GET /offline.html`` — the offline fallback shell that points back at the HUD.

Everything is vanilla HTML/CSS/JS — no bundler, no eval of any server output. The
integration agent wires this slice by including :data:`friday.pwa.router`.
"""

from __future__ import annotations

from friday.api.routes_pwa import router

__all__ = ["router"]
