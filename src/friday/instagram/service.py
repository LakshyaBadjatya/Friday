"""Instagram DM business logic + speech shaping — all client errors → soft strings.

The service holds every rule (counting, read-aloud limits/overflow, name
resolution) against the :class:`~friday.instagram.client.InstagramClient` Protocol,
so it's fully testable with a fake client. Each public method returns a plain
spoken string and NEVER raises: client failures are mapped to soft messages
(``InstagramAuthError`` -> re-run-setup, ``InstagramNotInstalled`` -> pip-install,
anything else -> "couldn't reach Instagram"), so the Siri endpoint can't 500.
"""

from __future__ import annotations

from friday.instagram.client import (
    InstagramAuthError,
    InstagramClient,
    InstagramNotInstalled,
)
from friday.instagram.models import IgThread, display_name
from friday.siri.speech import for_speech

_AUTH_MSG = (
    "Instagram needs me to verify this login — re-run the Instagram setup on your "
    "machine, Boss."
)
_NOT_INSTALLED_MSG = (
    "Instagram support isn't installed yet — run pip install instagrapi."
)
_UNREACHABLE_MSG = "I couldn't reach Instagram right now, Boss."
_NONE_MSG = "No new Instagram DMs, Boss."


class _SoftError(Exception):
    """Internal: carries a pre-shaped soft spoken message out of a client call."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


def _soft(exc: Exception) -> _SoftError:
    """Map any client exception to the right soft spoken message."""
    if isinstance(exc, InstagramAuthError):
        return _SoftError(_AUTH_MSG)
    if isinstance(exc, InstagramNotInstalled):
        return _SoftError(_NOT_INSTALLED_MSG)
    return _SoftError(_UNREACHABLE_MSG)


class InstagramService:
    """Spoken-reply logic over an :class:`InstagramClient`. Methods never raise."""

    def __init__(self, client: InstagramClient, *, read_aloud_limit: int = 5) -> None:
        self._client = client
        self._limit = max(1, read_aloud_limit)

    # -- count ------------------------------------------------------------- #
    def unread_summary(self) -> str:
        """Spoken count of unread DMs, with up to three sender breakdowns."""
        try:
            threads = self._client.unread_threads()
        except Exception as exc:  # noqa: BLE001 - mapped to a soft message
            return _soft(exc).message
        total = sum(max(0, t.unread_count) for t in threads)
        if total <= 0:
            return _NONE_MSG
        parts = [
            f"{t.unread_count} from {display_name(t)}"
            for t in threads
            if t.unread_count > 0
        ][:3]
        breakdown = ", ".join(parts)
        plural = "s" if total != 1 else ""
        head = f"You have {total} unread Instagram DM{plural}"
        if breakdown:
            head += f" — {breakdown}"
        return f"{head}. Say 'read my Instagram messages' to hear them."

    # -- read aloud -------------------------------------------------------- #
    def read_unread_aloud(self) -> str:
        """Read up to ``read_aloud_limit`` unread DMs aloud, bounded for speech."""
        try:
            threads = self._client.unread_threads()
            lines, remaining = self._collect_lines(threads)
        except Exception as exc:  # noqa: BLE001 - mapped to a soft message
            return _soft(exc).message
        if not lines:
            return _NONE_MSG
        spoken = " ".join(lines)
        if remaining > 0:
            spoken += f" …and {remaining} more. Open Instagram for the rest."
        return for_speech(spoken)

    def _collect_lines(self, threads: list[IgThread]) -> tuple[list[str], int]:
        """Build "From X: text." lines up to the limit; return (lines, overflow)."""
        lines: list[str] = []
        produced = 0
        overflow = 0
        for thread in threads:
            want = thread.unread_count if thread.unread_count > 0 else 1
            texts = self._newest_texts(thread, want)
            who = display_name(thread)
            for text in texts:
                if produced >= self._limit:
                    overflow += 1
                    continue
                lines.append(f"From {who}: {text}.")
                produced += 1
        return lines, overflow

    def _newest_texts(self, thread: IgThread, want: int) -> list[str]:
        """Newest ``want`` message texts for ``thread`` (last_text if fetch is empty)."""
        try:
            messages = self._client.thread_messages(thread.thread_id, want)
        except Exception:  # noqa: BLE001 - fall back to the thread preview
            messages = []
        texts = [m.text.strip() for m in messages if m.text and m.text.strip()]
        if not texts and thread.last_text.strip():
            texts = [thread.last_text.strip()]
        return texts[:want]

    # -- reply ------------------------------------------------------------- #
    def reply(self, name: str, text: str) -> str:
        """Resolve ``name`` in recent chats and send ``text``; spoken confirmation."""
        try:
            thread = self._resolve(name)
            if thread is None:
                return f"I couldn't find {name} in your recent Instagram chats."
            sent = self._client.send_dm(thread.thread_id, text)
        except Exception as exc:  # noqa: BLE001 - mapped to a soft message
            return _soft(exc).message
        if not sent:
            return "I couldn't send that on Instagram right now, Boss."
        return f"Sent to {display_name(thread)} on Instagram."

    def _resolve(self, name: str) -> IgThread | None:
        """Find a thread whose username/full name matches ``name`` (case-insensitive)."""
        key = name.strip().lower()
        if not key:
            return None
        try:
            threads = self._client.recent_threads(20)
        except Exception:  # noqa: BLE001 - fall back to the unread set
            threads = self._client.unread_threads()
        for thread in threads:
            username = thread.username.strip().lower()
            full = thread.full_name.strip().lower()
            if key in username or key in full:
                return thread
            if full and key == full.split()[0]:
                return thread
        return None
