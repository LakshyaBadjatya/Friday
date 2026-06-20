"""Instagram DMs for FRIDAY — count, read-aloud, and reply over ``/siri/ask``.

An isolated package mirroring ``circle/``: a Protocol-based ``client`` (the only
instagrapi importer, lazy), a ``service`` holding all logic, regex ``intents``, and
a thin ``siri_instagram`` handler. Wired into ``routes_siri`` behind the
``enable_instagram_dms`` flag; instagrapi is an optional/lazy backend kept out of
``pyproject``.
"""

from __future__ import annotations

from friday.instagram.client import (
    InstagramAuthError,
    InstagramClient,
    InstagramError,
    InstagramNotInstalled,
    InstagrapiClient,
)
from friday.instagram.models import IgMessage, IgThread, display_name
from friday.instagram.service import InstagramService
from friday.instagram.session import parse_session
from friday.instagram.siri_instagram import handle

__all__ = [
    "IgMessage",
    "IgThread",
    "InstagramAuthError",
    "InstagramClient",
    "InstagramError",
    "InstagramNotInstalled",
    "InstagramService",
    "InstagrapiClient",
    "display_name",
    "handle",
    "parse_session",
]
