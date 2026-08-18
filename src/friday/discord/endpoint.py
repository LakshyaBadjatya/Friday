"""Where Discord's REST API is, and why that is not simply a constant.

Discord's API sits behind Cloudflare, which rate-limits by source IP. On a host
that shares one outbound address with strangers — Render's free tier does — a
429 carrying Cloudflare's error code 1015 can arrive for traffic that is not
ours, and it is not a brief one: the block that prompted this asked for 76,255
seconds. For twenty-one hours she answered every message correctly and had every
send refused, while staying online, because the gateway socket connects to a
different edge that was never blocked.

So the base is configurable, and pointing it at a proxy on a different network
(``scripts/discord-proxy-worker.js``) is enough to route around a blocked egress
without touching a single call site.

Read from the environment rather than from :class:`~friday.config.Settings`
deliberately. These are module-level constants in five modules, several of whose
functions never see a ``Settings`` — threading one through ``_send_one`` and
friends to express "which host is Discord on" would be a lot of plumbing for a
value that cannot change while the process runs.
"""

from __future__ import annotations

import os

#: Discord's own API, used when nothing is configured. The overwhelmingly normal
#: case: a proxy is a workaround for a blocked host, not the default posture.
DEFAULT_API = "https://discord.com/api/v10"

#: The gateway is *not* proxied. Only REST calls were ever blocked, a WebSocket
#: is harder to forward correctly, and keeping it direct means a broken proxy
#: costs replies rather than taking her offline entirely.
DEFAULT_GATEWAY = "wss://gateway.discord.gg/?v=10&encoding=json"


def api_base() -> str:
    """The base every REST URL is built from, without a trailing slash.

    ``FRIDAY_DISCORD_API_BASE`` should include the API version, so that the paths
    the callers build stay identical whether or not a proxy is in the way — e.g.
    ``https://friday-discord.example.workers.dev/<secret>/v10``.
    """
    configured = (os.getenv("FRIDAY_DISCORD_API_BASE") or "").strip()
    return configured.rstrip("/") if configured else DEFAULT_API


def is_proxied() -> bool:
    """Whether sends are going somewhere other than Discord itself.

    Worth a line at startup: a reply that vanishes is confusing enough without
    having to guess which host it was posted to.
    """
    return api_base() != DEFAULT_API
