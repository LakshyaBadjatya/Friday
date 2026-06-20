"""Domain models for Instagram DMs — provider-agnostic, no instagrapi here.

Frozen dataclasses the service reasons over, decoupled from instagrapi's own
objects (which :mod:`friday.instagram.client` converts into these). ``display_name``
picks the friendliest label for speech: a person's full name when known, else their
``@username``. Nothing here imports instagrapi or touches the network.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class IgThread:
    """One Instagram DM thread (conversation) with a single person."""

    thread_id: str
    username: str = ""
    full_name: str = ""
    unread_count: int = 0
    last_text: str = ""
    last_at: datetime | None = None


@dataclass(frozen=True)
class IgMessage:
    """A single message within a thread."""

    message_id: str
    from_username: str = ""
    text: str = ""
    created_at: datetime | None = None


def display_name(thread: IgThread) -> str:
    """Friendliest spoken label for ``thread``: full name, else ``@username``."""
    full = (thread.full_name or "").strip()
    if full:
        return full
    user = (thread.username or "").strip()
    if user:
        return f"@{user}"
    return "someone"
