"""A thin ``httpx`` adapter over the Gmail v1 REST API (read + DRAFT only).

:class:`GmailClient` mirrors the keyless-tool style of
:mod:`friday.integrations.calendar` and :mod:`friday.tools.web_search`: a small
async surface built on an injected ``httpx.AsyncClient``, with typed failures
(:class:`EmailError`) rather than bare exceptions leaking out. It uses NO Google
SDK — every call is a plain HTTPS request to the Gmail v1 REST API.

Surface (on the user's ``me`` mailbox):

* :meth:`GmailClient.list_messages` — ``GET .../messages?q=`` returns the parsed
  ``messages`` list (read-only, idempotent).
* :meth:`GmailClient.summarize_inbox` — lists the inbox and runs ONE *non-fatal*
  LLM pass to produce a short natural-language summary; any LLM error (or an
  empty inbox) degrades to a deterministic, count-bearing fallback and never
  raises (the list step still raises on a real REST failure).
* :meth:`GmailClient.draft_reply` — creates a **DRAFT** reply (the dedicated
  ``.../drafts`` endpoint) on the original message's thread and returns the
  created-draft JSON. This NEVER auto-sends — there is no call to the Gmail
  ``send`` endpoint anywhere on this path; sending stays a separate, explicit,
  human-confirmed action.

SECURITY: the OAuth bearer token originates from a :class:`~pydantic.SecretStr`
in config; here it is held as a plain ``str | None`` but is ONLY ever placed in
the ``Authorization: Bearer`` request header and is never logged (no ``logger``
call includes it, and error messages carry the status/endpoint — never the
token). A missing token raises a clear :class:`EmailError` BEFORE any network
I/O.
"""

from __future__ import annotations

import base64
import logging
from email.message import EmailMessage
from typing import Any

import httpx

from friday.errors import FridayError
from friday.providers.llm import LLMProvider, Message

logger = logging.getLogger("friday.integrations.email")

#: Base of the Gmail v1 REST API for the authenticated user's mailbox.
_BASE_URL = "https://gmail.googleapis.com/gmail/v1/users/me"
#: The messages collection (list + per-id get).
_MESSAGES_URL = f"{_BASE_URL}/messages"
#: The drafts collection (create-draft is the ONLY mutation FRIDAY performs).
_DRAFTS_URL = f"{_BASE_URL}/drafts"
#: Default per-request wall-clock budget (seconds) for every Gmail REST call.
_DEFAULT_TIMEOUT = 15.0
#: How many inbox messages the summary prompt references (bounds prompt size).
_SUMMARY_LIMIT = 25


class EmailError(FridayError):
    """A Gmail REST call failed (missing token, transport, or non-2xx)."""


class GmailClient:
    """Async ``httpx`` client for the subset of Gmail v1 FRIDAY uses.

    Args:
        token: The Gmail OAuth bearer token, or ``None`` when unset. Sent as
            ``Authorization: Bearer`` on every call; every method raises a clear
            :class:`EmailError` when it is ``None`` (before any network I/O).
        http: An injected ``httpx.AsyncClient`` the caller owns (so the client is
            trivially testable with ``respx`` and shares connection pooling).
        timeout: Per-request wall-clock budget in seconds.
    """

    def __init__(
        self,
        token: str | None,
        *,
        http: httpx.AsyncClient,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        self._token = token
        self._http = http
        self._timeout = timeout

    @property
    def has_token(self) -> bool:
        """Whether a bearer token is configured (without exposing it)."""
        return bool(self._token)

    def __repr__(self) -> str:
        """A token-free repr (the secret never leaks into repr/str/logs)."""
        return f"GmailClient(has_token={self.has_token})"

    async def aclose(self) -> None:
        """Close the underlying ``httpx.AsyncClient`` (when the client owns it)."""
        await self._http.aclose()

    def _auth_headers(self) -> dict[str, str]:
        """The authenticated-request headers; raises when no token is set.

        The token is placed ONLY here, in the ``Authorization`` header — never in
        a log line or an error message.
        """
        if not self._token:
            raise EmailError(
                "gmail oauth token not set — configure FRIDAY_GMAIL_OAUTH_TOKEN "
                "to use email"
            )
        return {"Authorization": f"Bearer {self._token}"}

    async def list_messages(self, query: str) -> list[dict[str, Any]]:
        """``GET .../messages?q=``; return the parsed ``messages`` list (read-only).

        Raises :class:`EmailError` when no token is configured (before any network
        I/O), on a transport error, or on a non-2xx response. A response with no
        ``messages`` yields an empty list (never raises).
        """
        headers = self._auth_headers()
        params = {"q": query}
        try:
            response = await self._http.get(
                _MESSAGES_URL, params=params, headers=headers, timeout=self._timeout
            )
        except httpx.HTTPError as exc:
            logger.warning("email list_messages transport error: %s", exc)
            raise EmailError(f"gmail list request failed: {exc}") from exc

        body = _require_object(response, "list")
        messages = body.get("messages", [])
        return [m for m in messages if isinstance(m, dict)]

    async def get_message(self, message_id: str) -> dict[str, Any]:
        """``GET .../messages/{id}``; return the full message resource.

        Used by :meth:`draft_reply` to read the original thread/headers so the
        reply stays on-thread. Raises :class:`EmailError` like the other reads.
        """
        headers = self._auth_headers()
        try:
            response = await self._http.get(
                f"{_MESSAGES_URL}/{message_id}",
                headers=headers,
                timeout=self._timeout,
            )
        except httpx.HTTPError as exc:
            logger.warning("email get_message transport error: %s", exc)
            raise EmailError(f"gmail get request failed: {exc}") from exc

        return _require_object(response, "get")

    async def summarize_inbox(self, llm: LLMProvider) -> str:
        """List the inbox and return a short, *non-fatal* LLM summary.

        The list step raises :class:`EmailError` on a real REST failure (so a
        broken token/endpoint is surfaced). The summary step is best-effort: an
        empty inbox skips the LLM entirely, and any LLM error degrades to a
        deterministic, count-bearing fallback (never raises). The token is never
        included in the prompt — only message ids/thread ids are.
        """
        messages = await self.list_messages("in:inbox")
        count = len(messages)
        if count == 0:
            return "Your inbox is empty — no messages."

        ids = ", ".join(str(m.get("id", "?")) for m in messages[:_SUMMARY_LIMIT])
        prompt = (
            f"You are summarizing an email inbox. There are {count} message(s). "
            f"Their ids are: {ids}. Write one short, friendly sentence telling the "
            f"user how many messages are waiting."
        )
        try:
            response = await llm.complete([Message(role="user", content=prompt)])
            text = (response.text or "").strip()
            if text:
                return text
        except Exception:  # noqa: BLE001 - inbox summary is optional + non-fatal
            logger.warning("email summarize_inbox LLM pass failed; using fallback")
        return f"You have {count} message(s) in your inbox."

    async def draft_reply(self, message_id: str, body: str) -> dict[str, Any]:
        """Create a **DRAFT** reply on the message's thread; return the draft JSON.

        This is the ONLY mutating call and it creates a draft ONLY — it NEVER
        auto-sends (there is no request to the Gmail ``send`` endpoint anywhere on
        this path). The original message is fetched to recover its thread id and
        recipient/subject so the reply threads correctly; the raw RFC 822 message
        is base64url-encoded into the Gmail draft resource.

        Raises :class:`EmailError` when no token is configured (before any network
        I/O), on a transport error, or on a non-2xx response.
        """
        headers = self._auth_headers()
        original = await self.get_message(message_id)
        thread_id = original.get("threadId")
        reply_headers = _extract_headers(original)

        raw = _build_reply_mime(reply_headers, body)
        draft: dict[str, Any] = {"message": {"raw": raw}}
        if isinstance(thread_id, str) and thread_id:
            draft["message"]["threadId"] = thread_id

        try:
            response = await self._http.post(
                _DRAFTS_URL, json=draft, headers=headers, timeout=self._timeout
            )
        except httpx.HTTPError as exc:
            logger.warning("email draft_reply transport error: %s", exc)
            raise EmailError(f"gmail draft request failed: {exc}") from exc

        return _require_object(response, "draft")


def _require_object(response: httpx.Response, action: str) -> dict[str, Any]:
    """Validate a Gmail response is a 2xx JSON object; raise :class:`EmailError` else.

    The error message carries the status and a short, token-free body snippet
    (the bearer token only ever rides in the request header) so a failure is
    diagnosable without leaking the secret.
    """
    if not response.is_success:
        logger.warning("email %s returned HTTP %d", action, response.status_code)
        raise EmailError(
            f"gmail {action} returned HTTP {response.status_code}: "
            f"{_safe_body(response)}"
        )
    try:
        parsed = response.json()
    except ValueError as exc:
        raise EmailError(f"gmail {action} returned a non-JSON body: {exc}") from exc
    if not isinstance(parsed, dict):
        raise EmailError(
            f"gmail {action} returned an unexpected (non-object) body"
        )
    return parsed


def _extract_headers(message: dict[str, Any]) -> dict[str, str]:
    """Pull the ``From``/``Subject``/``Message-ID`` headers from a message resource.

    Returns a lowercase-keyed dict (missing headers simply absent). A malformed
    payload yields an empty dict rather than raising — the draft still builds.
    """
    payload = message.get("payload")
    if not isinstance(payload, dict):
        return {}
    headers = payload.get("headers")
    if not isinstance(headers, list):
        return {}
    out: dict[str, str] = {}
    for header in headers:
        if not isinstance(header, dict):
            continue
        name = header.get("name")
        value = header.get("value")
        if isinstance(name, str) and isinstance(value, str):
            out[name.lower()] = value
    return out


def _build_reply_mime(headers: dict[str, str], body: str) -> str:
    """Build a base64url-encoded RFC 822 reply from the original headers + body.

    The reply ``To`` is the original ``From``; the ``Subject`` gains a ``Re:``
    prefix (idempotently); ``In-Reply-To``/``References`` thread it when the
    original carried a ``Message-ID``. The result is the ``raw`` field Gmail's
    draft API expects (URL-safe base64, no padding stripped — Gmail accepts it).
    """
    msg = EmailMessage()
    to_addr = headers.get("from")
    if to_addr:
        msg["To"] = to_addr
    subject = headers.get("subject", "")
    if subject and not subject.lower().startswith("re:"):
        subject = f"Re: {subject}"
    if subject:
        msg["Subject"] = subject
    original_id = headers.get("message-id")
    if original_id:
        msg["In-Reply-To"] = original_id
        msg["References"] = original_id
    msg.set_content(body)
    return base64.urlsafe_b64encode(msg.as_bytes()).decode("ascii")


def _safe_body(response: httpx.Response) -> str:
    """A short, token-free snippet of a response body for an error message.

    Truncated so a large/HTML error page does not bloat the raised error. Carries
    no FRIDAY secret (the bearer token is only ever in the request header).
    """
    text = response.text or ""
    snippet = text.strip().replace("\n", " ")
    return snippet[:200]
