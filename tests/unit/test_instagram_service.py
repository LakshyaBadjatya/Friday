"""Unit tests for InstagramService against a fake client implementing the Protocol.

Covers: zero-unread, multi-thread summary phrasing, read-aloud formatting +
limit + overflow tail, reply hit / not-found / send-failure, and the three
soft-error mappings (auth, not-installed, any-other) — proving the service never
raises. All offline; the fake stands in for instagrapi.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from friday.instagram.client import (
    InstagramAuthError,
    InstagramNotInstalled,
)
from friday.instagram.models import IgMessage, IgThread
from friday.instagram.service import InstagramService

NOW = datetime(2026, 6, 20, 12, 0, tzinfo=UTC)


class FakeInstagramClient:
    """In-memory :class:`InstagramClient` for tests; optionally raises ``raises``."""

    def __init__(
        self,
        *,
        unread: list[IgThread] | None = None,
        recent: list[IgThread] | None = None,
        messages: dict[str, list[IgMessage]] | None = None,
        raises: Exception | None = None,
        send_result: bool = True,
    ) -> None:
        self._unread = unread or []
        self._recent = recent or []
        self._messages = messages or {}
        self._raises = raises
        self._send_result = send_result
        self.sent: list[tuple[str, str]] = []

    def _maybe_raise(self) -> None:
        if self._raises is not None:
            raise self._raises

    def unread_threads(self) -> list[IgThread]:
        self._maybe_raise()
        return self._unread

    def recent_threads(self, limit: int) -> list[IgThread]:
        self._maybe_raise()
        return self._recent

    def thread_messages(self, thread_id: str, limit: int) -> list[IgMessage]:
        self._maybe_raise()
        return self._messages.get(thread_id, [])[:limit]

    def send_dm(self, thread_id: str, text: str) -> bool:
        self._maybe_raise()
        self.sent.append((thread_id, text))
        return self._send_result


def _thread(tid: str, *, username: str = "", full_name: str = "", unread: int = 0,
            last_text: str = "") -> IgThread:
    return IgThread(
        thread_id=tid,
        username=username,
        full_name=full_name,
        unread_count=unread,
        last_text=last_text,
    )


# -- count ----------------------------------------------------------------- #
def test_unread_summary_zero() -> None:
    svc = InstagramService(FakeInstagramClient(unread=[]))
    assert svc.unread_summary() == "No new Instagram DMs, Boss."


def test_unread_summary_multi_thread_phrasing() -> None:
    threads = [
        _thread("1", full_name="Rahul Mehta", unread=2),
        _thread("2", username="priya_k", unread=1),
        _thread("3", full_name="Sam", unread=3),
        _thread("4", full_name="Extra", unread=1),  # 4th: beyond the 3 in breakdown
    ]
    svc = InstagramService(FakeInstagramClient(unread=threads))
    reply = svc.unread_summary()
    assert reply.startswith("You have 7 unread Instagram DMs — ")
    assert "2 from Rahul Mehta" in reply
    assert "1 from @priya_k" in reply
    assert "3 from Sam" in reply
    assert "Extra" not in reply  # breakdown capped at 3
    assert reply.endswith("Say 'read my Instagram messages' to hear them.")


def test_unread_summary_singular() -> None:
    svc = InstagramService(
        FakeInstagramClient(unread=[_thread("1", full_name="Rahul", unread=1)])
    )
    reply = svc.unread_summary()
    assert "1 unread Instagram DM " in reply  # singular: "DM", no trailing s before " —"
    assert "DMs" not in reply


# -- read aloud ------------------------------------------------------------ #
def test_read_aloud_formatting() -> None:
    threads = [_thread("1", full_name="Rahul", unread=1)]
    messages = {"1": [IgMessage(message_id="m1", text="hey are you free")]}
    svc = InstagramService(FakeInstagramClient(unread=threads, messages=messages))
    assert svc.read_unread_aloud() == "From Rahul: hey are you free."


def test_read_aloud_none() -> None:
    svc = InstagramService(FakeInstagramClient(unread=[]))
    assert svc.read_unread_aloud() == "No new Instagram DMs, Boss."


def test_read_aloud_falls_back_to_last_text() -> None:
    # No fetched messages -> use the thread's last_text preview.
    threads = [_thread("1", full_name="Rahul", unread=1, last_text="see you soon")]
    svc = InstagramService(FakeInstagramClient(unread=threads, messages={}))
    assert svc.read_unread_aloud() == "From Rahul: see you soon."


def test_read_aloud_limit_and_overflow() -> None:
    # Two threads, 3 unread each = 6 messages; limit 2 -> 2 read + "…and 4 more".
    threads = [
        _thread("1", full_name="Rahul", unread=3),
        _thread("2", full_name="Priya", unread=3),
    ]
    messages = {
        "1": [IgMessage(message_id=f"a{i}", text=f"a{i}") for i in range(3)],
        "2": [IgMessage(message_id=f"b{i}", text=f"b{i}") for i in range(3)],
    }
    svc = InstagramService(
        FakeInstagramClient(unread=threads, messages=messages), read_aloud_limit=2
    )
    reply = svc.read_unread_aloud()
    assert reply.startswith("From Rahul: a0. From Rahul: a1.")
    assert "…and 4 more. Open Instagram for the rest." in reply


# -- reply ----------------------------------------------------------------- #
def test_reply_hit_full_name() -> None:
    recent = [_thread("9", full_name="Rahul Mehta", username="rahul_m")]
    fake = FakeInstagramClient(recent=recent)
    svc = InstagramService(fake)
    assert svc.reply("rahul", "on my way") == "Sent to Rahul Mehta on Instagram."
    assert fake.sent == [("9", "on my way")]


def test_reply_hit_username() -> None:
    recent = [_thread("9", username="priya_k")]
    fake = FakeInstagramClient(recent=recent)
    svc = InstagramService(fake)
    assert svc.reply("priya", "hi") == "Sent to @priya_k on Instagram."


def test_reply_not_found() -> None:
    svc = InstagramService(FakeInstagramClient(recent=[]))
    assert (
        svc.reply("nobody", "hi")
        == "I couldn't find nobody in your recent Instagram chats."
    )


def test_reply_send_failure() -> None:
    recent = [_thread("9", full_name="Rahul")]
    svc = InstagramService(FakeInstagramClient(recent=recent, send_result=False))
    assert svc.reply("rahul", "hi") == "I couldn't send that on Instagram right now, Boss."


# -- soft error mapping ---------------------------------------------------- #
@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (
            InstagramAuthError(),
            "Instagram needs me to verify this login — re-run the Instagram setup "
            "on your machine, Boss.",
        ),
        (
            InstagramNotInstalled(),
            "Instagram support isn't installed yet — run pip install instagrapi.",
        ),
        (RuntimeError("boom"), "I couldn't reach Instagram right now, Boss."),
    ],
)
def test_soft_error_messages(exc: Exception, expected: str) -> None:
    svc = InstagramService(FakeInstagramClient(raises=exc))
    assert svc.unread_summary() == expected
    assert svc.read_unread_aloud() == expected
    assert svc.reply("rahul", "hi") == expected
