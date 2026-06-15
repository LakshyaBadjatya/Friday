"""Unit tests for :class:`friday.integrations.email.GmailClient` (offline).

Every HTTP call is mocked with ``respx`` against the Gmail v1 REST endpoints —
no live network, no real OAuth. The client is a thin ``httpx`` adapter
(mirroring :mod:`friday.integrations.calendar` / :mod:`friday.tools.web_search`):
the OAuth bearer token is held as a plain ``str | None`` (sourced from a
:class:`~pydantic.SecretStr` in config so it never logs) and sent ONLY as the
``Authorization: Bearer`` header.

Covered:
* ``list_messages`` issues an authenticated ``GET`` with the ``q`` query and
  parses the ``messages`` list.
* ``summarize_inbox`` runs ONE non-fatal LLM pass over the listed messages and
  returns its text; an LLM failure degrades to a deterministic fallback (never
  raises).
* ``draft_reply`` creates a DRAFT only (the dedicated drafts endpoint) and never
  hits the send endpoint; it returns the created-draft JSON.
* A missing token raises a clear :class:`EmailError` BEFORE any network I/O — for
  the list, summarize, and draft paths.
* A non-2xx status surfaces as a typed :class:`EmailError` (no leaked 500).
* The token never appears in the client's ``repr``/``str``.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from friday.errors import ProviderError
from friday.integrations.email import EmailError, GmailClient
from friday.providers.llm import FakeLLM, LLMResponse

_LIST_URL = "https://gmail.googleapis.com/gmail/v1/users/me/messages"
_DRAFTS_URL = "https://gmail.googleapis.com/gmail/v1/users/me/drafts"
_SEND_URL = "https://gmail.googleapis.com/gmail/v1/users/me/messages/send"

_LIST_BODY = {
    "messages": [
        {"id": "m1", "threadId": "t1"},
        {"id": "m2", "threadId": "t2"},
    ],
    "resultSizeEstimate": 2,
}


def _llm(text: str) -> FakeLLM:
    """A FakeLLM that returns a single scripted text response."""
    return FakeLLM([LLMResponse(text=text)])


@respx.mock
async def test_list_messages_parses_messages() -> None:
    """``list_messages`` issues an authed GET with ``q`` and returns the list."""
    route = respx.get(_LIST_URL).mock(
        return_value=httpx.Response(200, json=_LIST_BODY)
    )

    async with httpx.AsyncClient() as http:
        client = GmailClient("tok-secret", http=http)
        messages = await client.list_messages("is:unread")

    assert [m["id"] for m in messages] == ["m1", "m2"]
    sent = route.calls.last.request
    # The bearer token rides ONLY in the Authorization header.
    assert sent.headers["Authorization"] == "Bearer tok-secret"
    # The Gmail search query is passed as the ``q`` query param.
    assert sent.url.params["q"] == "is:unread"


@respx.mock
async def test_list_messages_empty_returns_empty_list() -> None:
    """A response with no ``messages`` key yields an empty list (never raises)."""
    respx.get(_LIST_URL).mock(return_value=httpx.Response(200, json={}))

    async with httpx.AsyncClient() as http:
        client = GmailClient("tok-secret", http=http)
        messages = await client.list_messages("in:inbox")

    assert messages == []


@respx.mock
async def test_list_messages_non_2xx_raises_email_error() -> None:
    """A non-2xx status surfaces as a typed :class:`EmailError` (no leak)."""
    respx.get(_LIST_URL).mock(return_value=httpx.Response(403, text="forbidden"))

    async with httpx.AsyncClient() as http:
        client = GmailClient("tok-secret", http=http)
        with pytest.raises(EmailError) as exc_info:
            await client.list_messages("in:inbox")
    assert "403" in str(exc_info.value)


@respx.mock
async def test_summarize_inbox_uses_llm() -> None:
    """``summarize_inbox`` lists the inbox and returns the LLM's summary text."""
    respx.get(_LIST_URL).mock(return_value=httpx.Response(200, json=_LIST_BODY))

    async with httpx.AsyncClient() as http:
        client = GmailClient("tok-secret", http=http)
        summary = await client.summarize_inbox(_llm("You have 2 unread messages."))

    assert summary == "You have 2 unread messages."


@respx.mock
async def test_summarize_inbox_llm_failure_is_non_fatal() -> None:
    """An LLM failure degrades to a deterministic fallback (never raises)."""
    respx.get(_LIST_URL).mock(return_value=httpx.Response(200, json=_LIST_BODY))

    class _BoomLLM(FakeLLM):
        async def complete(self, messages, tools=None):  # type: ignore[no-untyped-def]
            raise ProviderError("llm down")

    async with httpx.AsyncClient() as http:
        client = GmailClient("tok-secret", http=http)
        summary = await client.summarize_inbox(_BoomLLM([]))

    # Non-fatal: a deterministic, count-bearing fallback rather than a raise.
    assert isinstance(summary, str)
    assert "2" in summary


@respx.mock
async def test_summarize_inbox_empty_inbox_skips_llm() -> None:
    """An empty inbox returns a deterministic message without calling the LLM."""
    respx.get(_LIST_URL).mock(return_value=httpx.Response(200, json={}))

    # A FakeLLM with no scripted responses would raise if called -> proves no call.
    async with httpx.AsyncClient() as http:
        client = GmailClient("tok-secret", http=http)
        summary = await client.summarize_inbox(FakeLLM([]))

    assert isinstance(summary, str)
    assert summary  # non-empty


@respx.mock
async def test_draft_reply_creates_draft_only_never_sends() -> None:
    """``draft_reply`` POSTs to the DRAFTS endpoint and never touches ``send``."""
    created = {"id": "d1", "message": {"id": "msgd1", "threadId": "t1"}}
    drafts_route = respx.post(_DRAFTS_URL).mock(
        return_value=httpx.Response(200, json=created)
    )
    # If the client ever sends, this route would record a call — it must not.
    send_route = respx.post(_SEND_URL).mock(
        return_value=httpx.Response(200, json={"id": "should-not-happen"})
    )
    # The reply needs the original message's thread/headers; the client reads it.
    respx.get(f"{_LIST_URL}/m1").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "m1",
                "threadId": "t1",
                "payload": {
                    "headers": [
                        {"name": "Subject", "value": "Hello"},
                        {"name": "From", "value": "alice@example.com"},
                        {"name": "Message-ID", "value": "<orig@x>"},
                    ]
                },
            },
        )
    )

    async with httpx.AsyncClient() as http:
        client = GmailClient("tok-secret", http=http)
        draft = await client.draft_reply("m1", "Thanks, will do.")

    assert draft["id"] == "d1"
    assert drafts_route.called
    # CRITICAL: a draft is NEVER auto-sent.
    assert not send_route.called
    sent = drafts_route.calls.last.request
    assert sent.headers["Authorization"] == "Bearer tok-secret"
    # The reply stays on the original thread.
    assert b"t1" in sent.content


@respx.mock
async def test_draft_reply_non_2xx_raises_email_error() -> None:
    """A non-2xx draft response surfaces as a typed :class:`EmailError`."""
    respx.get(f"{_LIST_URL}/m1").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "m1",
                "threadId": "t1",
                "payload": {"headers": [{"name": "From", "value": "a@b.c"}]},
            },
        )
    )
    respx.post(_DRAFTS_URL).mock(return_value=httpx.Response(500, text="boom"))

    async with httpx.AsyncClient() as http:
        client = GmailClient("tok-secret", http=http)
        with pytest.raises(EmailError) as exc_info:
            await client.draft_reply("m1", "body")
    assert "500" in str(exc_info.value)


async def test_missing_token_list_raises_before_network() -> None:
    """A missing token raises a clear error BEFORE any network I/O (list path)."""
    async with httpx.AsyncClient() as http:
        client = GmailClient(None, http=http)
        with pytest.raises(EmailError) as exc_info:
            await client.list_messages("in:inbox")
    assert "token" in str(exc_info.value).lower()


async def test_missing_token_summarize_raises_before_network() -> None:
    """A missing token raises a clear error BEFORE any network I/O (summarize)."""
    async with httpx.AsyncClient() as http:
        client = GmailClient(None, http=http)
        with pytest.raises(EmailError) as exc_info:
            await client.summarize_inbox(FakeLLM([]))
    assert "token" in str(exc_info.value).lower()


async def test_missing_token_draft_raises_before_network() -> None:
    """A missing token raises a clear error BEFORE any network I/O (draft path)."""
    async with httpx.AsyncClient() as http:
        client = GmailClient(None, http=http)
        with pytest.raises(EmailError) as exc_info:
            await client.draft_reply("m1", "body")
    assert "token" in str(exc_info.value).lower()


def test_token_never_in_repr() -> None:
    """The bearer token never leaks into the client's ``repr``/``str``."""
    client = GmailClient("super-secret-oauth-token", http=None)  # type: ignore[arg-type]
    assert "super-secret-oauth-token" not in repr(client)
    assert "super-secret-oauth-token" not in str(client)


def test_has_token_reflects_presence() -> None:
    """``has_token`` reflects whether a token is configured (without exposing it)."""
    assert GmailClient("t", http=None).has_token is True  # type: ignore[arg-type]
    assert GmailClient(None, http=None).has_token is False  # type: ignore[arg-type]
