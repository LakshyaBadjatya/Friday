"""Thin Siri entry point for Instagram DMs: regex-classify first, then dispatch.

``handle`` is the single seam :mod:`friday.api.routes_siri` calls. It classifies
the spoken query with cheap regex (no network) and only then asks the service to
act. A bare "read them aloud" (no Instagram token) is honoured only inside a short
window after the user just asked about Instagram — tracked by ``marker`` (the
timestamp of the last Instagram turn). Returns ``None`` to fall through.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from friday.instagram.intents import handle_intent, is_bare_read, parse_intent
from friday.instagram.service import InstagramService

#: How long after an Instagram turn a bare "read them aloud" still refers to it.
_BARE_READ_WINDOW = timedelta(minutes=2)


def handle(
    service: InstagramService,
    query: str,
    now: datetime,
    *,
    marker: datetime | None,
) -> str | None:
    """Act on ``query`` via ``service``; ``None`` means 'fall through'.

    ``marker`` is the time of the user's last Instagram turn (``None`` if there
    hasn't been one), used to decide whether a bare "read them aloud" applies.
    """
    text = query.strip()
    intent = parse_intent(text)
    if intent is None:
        if (
            is_bare_read(text)
            and marker is not None
            and (now - marker) <= _BARE_READ_WINDOW
        ):
            return service.read_unread_aloud()
        return None
    return handle_intent(service, intent)
