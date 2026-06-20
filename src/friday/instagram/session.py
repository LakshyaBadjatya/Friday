"""Load a saved instagrapi session from the ``instagram_session_json`` secret.

The session JSON is produced once on the user's own machine by
``scripts/ig_login.py`` (``json.dumps(cl.get_settings())``) and pasted into the
``FRIDAY_INSTAGRAM_SESSION_JSON`` secret. Reusing it lets the server skip a fresh
password login from a datacenter IP (the usual challenge trigger). This module is
pure parsing — it never imports instagrapi, logs the secret, or hits the network.
"""

from __future__ import annotations

import json
from typing import Any


def parse_session(raw: str | None) -> dict[str, Any] | None:
    """Parse the saved session secret into instagrapi settings, or ``None``.

    Returns ``None`` for empty/blank/invalid input (a fresh password login is then
    attempted by the client) rather than raising — the route must never break.
    """
    if not raw or not raw.strip():
        return None
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return None
    return data if isinstance(data, dict) else None
