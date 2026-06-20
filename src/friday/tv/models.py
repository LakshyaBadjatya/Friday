"""The structured command a TV client executes, plus its action vocabulary."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel


class TVActionType(StrEnum):
    """What the TV client should do with an action."""

    OPEN_APP = "open_app"  # launch an installed app by (fuzzy) name
    PLAY = "play"  # play a query in an app (default: YouTube)
    SEARCH = "search"  # search a query in an app (no auto-play)
    MEDIA = "media"  # transport key: play_pause/stop/next/previous/rewind/fast_forward
    NAVIGATE = "navigate"  # navigation key: home/back


class TVAction(BaseModel):
    """A single resolved TV command. ``speak`` is the line FRIDAY says aloud."""

    type: TVActionType
    app: str | None = None
    query: str | None = None
    key: str | None = None
    speak: str
