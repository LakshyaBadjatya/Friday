"""The Instagram access boundary — the ONLY file that imports instagrapi.

``InstagramClient`` is the storage-agnostic Protocol the service depends on;
``InstagrapiClient`` is the real, best-effort adapter over the unofficial private
API. instagrapi is an OPTIONAL backend (like faster-whisper / pyautogui): it is
NOT in ``pyproject``, is imported lazily inside :meth:`InstagrapiClient._client`,
and a missing install maps to :class:`InstagramNotInstalled` — which the service
turns into a soft spoken "run pip install instagrapi" rather than a crash.

The adapter is exercised manually against the live API (it can't be unit-tested
offline); the service is what's tested, against a fake implementing the Protocol.
Every adapter method wraps its instagrapi calls and re-raises as ``Instagram*``
errors so the service has a single, typed surface to catch.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Protocol

from friday.instagram.models import IgMessage, IgThread


class InstagramError(Exception):
    """Base for all Instagram client failures."""


class InstagramAuthError(InstagramError):
    """Login needs re-verification (challenge / expired session / login required)."""


class InstagramNotInstalled(InstagramError):
    """The optional ``instagrapi`` dependency isn't installed."""


class InstagramClient(Protocol):
    """The Instagram surface the service depends on (provider-agnostic)."""

    def unread_threads(self) -> list[IgThread]: ...

    def recent_threads(self, limit: int) -> list[IgThread]: ...

    def thread_messages(self, thread_id: str, limit: int) -> list[IgMessage]: ...

    def send_dm(self, thread_id: str, text: str) -> bool: ...


def _to_thread(obj: Any) -> IgThread:
    """Convert an instagrapi DirectThread into our :class:`IgThread`, defensively."""
    users = getattr(obj, "users", None) or []
    user = users[0] if users else None
    username = str(getattr(user, "username", "") or "")
    full_name = str(getattr(user, "full_name", "") or "")
    messages = getattr(obj, "messages", None) or []
    last = messages[0] if messages else None
    last_text = str(getattr(last, "text", "") or "") if last is not None else ""
    last_at = getattr(last, "timestamp", None) if last is not None else None
    return IgThread(
        thread_id=str(getattr(obj, "id", "") or getattr(obj, "thread_id", "") or ""),
        username=username,
        full_name=full_name,
        unread_count=int(getattr(obj, "unread_count", 0) or 0),
        last_text=last_text,
        last_at=last_at,
    )


def _to_message(obj: Any) -> IgMessage:
    """Convert an instagrapi DirectMessage into our :class:`IgMessage`, defensively."""
    user_id = getattr(obj, "user_id", "")
    return IgMessage(
        message_id=str(getattr(obj, "id", "") or ""),
        from_username=str(user_id or ""),
        text=str(getattr(obj, "text", "") or ""),
        created_at=getattr(obj, "timestamp", None),
    )


class InstagrapiClient:
    """Best-effort adapter over the unofficial private API (lazy instagrapi).

    The underlying ``instagrapi.Client`` is built and logged in once, on first use,
    inside :meth:`_client` and cached. A saved ``session_settings`` lets it reuse a
    trusted residential-IP session (set on the user's machine) instead of a fresh
    datacenter login. NOT unit-tested — verified manually against the live API.
    """

    def __init__(
        self,
        username: str,
        password: str,
        session_settings: dict[str, Any] | None = None,
        *,
        delay_range: tuple[int, int] = (1, 3),
    ) -> None:
        self._username = username
        self._password = password
        self._session = session_settings
        self._delay_range = list(delay_range)
        self._cl: Any | None = None

    def _client(self) -> Any:
        """Build (once) and return the logged-in instagrapi client.

        Lazy-imports instagrapi (``ImportError`` -> :class:`InstagramNotInstalled`),
        applies a saved session if present, logs in only when needed, and maps
        instagrapi's challenge/login-required failures to :class:`InstagramAuthError`.
        """
        if self._cl is not None:
            return self._cl
        try:
            from instagrapi import Client  # noqa: PLC0415
            from instagrapi.exceptions import (  # noqa: PLC0415
                ChallengeRequired,
                LoginRequired,
            )
        except ImportError as exc:  # instagrapi not installed
            raise InstagramNotInstalled(
                "instagrapi is not installed (pip install instagrapi)"
            ) from exc

        cl = Client()
        try:
            cl.delay_range = self._delay_range
            if self._session:
                # Reuse the saved (residential-IP) session as-is; do NOT call
                # login() — that would re-trigger the password/2FA flow even when
                # the session is valid. If the session has expired, the actual API
                # call raises LoginRequired -> mapped to InstagramAuthError.
                cl.set_settings(self._session)
            else:
                cl.login(self._username, self._password)
        except (ChallengeRequired, LoginRequired) as exc:
            raise InstagramAuthError(str(exc)) from exc
        except InstagramError:
            raise
        except Exception as exc:  # noqa: BLE001 - any other login failure
            raise InstagramError(str(exc)) from exc
        self._cl = cl
        return cl

    def unread_threads(self) -> list[IgThread]:
        try:
            cl = self._client()
            raw = cl.direct_threads(amount=20, selected_filter="unread")
            threads: list[IgThread] = []
            for t in raw:
                thread = _to_thread(t)
                # The "unread" filter returns only unread conversations, but
                # instagrapi doesn't populate unread_count in the list view, so a
                # 0/None there still means "at least one unread message".
                if thread.unread_count < 1:
                    thread = replace(thread, unread_count=1)
                threads.append(thread)
            return threads
        except InstagramError:
            raise
        except Exception as exc:  # noqa: BLE001 - map any API error
            raise InstagramError(str(exc)) from exc

    def recent_threads(self, limit: int) -> list[IgThread]:
        try:
            cl = self._client()
            raw = cl.direct_threads(amount=limit)
            return [_to_thread(t) for t in raw]
        except InstagramError:
            raise
        except Exception as exc:  # noqa: BLE001 - map any API error
            raise InstagramError(str(exc)) from exc

    def thread_messages(self, thread_id: str, limit: int) -> list[IgMessage]:
        try:
            cl = self._client()
            raw = cl.direct_messages(thread_id, amount=limit)
            return [_to_message(m) for m in raw]
        except InstagramError:
            raise
        except Exception as exc:  # noqa: BLE001 - map any API error
            raise InstagramError(str(exc)) from exc

    def send_dm(self, thread_id: str, text: str) -> bool:
        try:
            cl = self._client()
            result = cl.direct_send(text, thread_ids=[int(thread_id)])
            return result is not None
        except InstagramError:
            raise
        except Exception as exc:  # noqa: BLE001 - map any API error
            raise InstagramError(str(exc)) from exc
