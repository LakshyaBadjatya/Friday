"""The HUD feature (Tier 3) — off by default behind ``FRIDAY_ENABLE_HUD``.

A flagged, no-build "cockpit" page served at ``/hud``: an arc-reactor-style boot
sequence animation with particle/glow styling, plus a Cmd/Ctrl-K command palette
that drives the existing same-origin ``/chat`` (talk to FRIDAY) and ``/admin``
(inspect health / metrics) endpoints. Everything is vanilla HTML/CSS/JS — no
bundler, no build step, no eval of any server output.

The router self-guards on the flag, so the offline default exposes no HUD
surface (every route -> 404). The integration agent wires this slice by
including :data:`friday.hud.router`.
"""

from __future__ import annotations

from friday.api.routes_hud import router

__all__ = ["router"]
