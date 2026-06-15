"""A thin ``httpx`` adapter over the Twilio Messages REST API (SMS + WhatsApp).

:class:`TwilioComms` mirrors the keyless/keyed-tool style of
:mod:`friday.integrations.calendar` and :mod:`friday.n8n.client`: a small async
surface built on an injected ``httpx.AsyncClient``, with typed failures
(:class:`CommsError`) rather than bare exceptions leaking out. It uses NO Twilio
SDK — every call is a plain HTTPS ``POST`` to the Twilio Messages REST API
(``/2010-04-01/Accounts/{Sid}/Messages.json``) authenticated with HTTP basic
auth (the account SID + auth token).

Surface:

* :meth:`TwilioComms.send_sms` — form-encodes ``To`` / ``From`` / ``Body`` and
  POSTs to the account's Messages endpoint; returns the created-message JSON.
* :meth:`TwilioComms.send_whatsapp` — the same call, but both the sender and
  recipient are prefixed with ``whatsapp:`` (Twilio's WhatsApp addressing).

Two ``Tool`` adapters expose the surface to the registry:

* :class:`SendSmsTool` / :class:`SendWhatsappTool` — both ``side_effecting=True``
  and ``idempotent=False`` so the registry's confirm-step (build-spec §12) gates
  them before execution. A missing-credentials or transport failure is surfaced
  as ``ToolResult(ok=False, error=ToolError(code="comms_failed"))`` rather than a
  raised exception.

SECURITY: the account SID and auth token originate from
:class:`~pydantic.SecretStr` fields in config; here they are held as plain
``str`` values but are ONLY ever used to build the HTTP basic-auth credentials
and are never logged (no ``logger`` call includes them, error messages carry the
status — never the token) and never appear in ``repr``/``str``. Missing
credentials raise a clear :class:`CommsError` BEFORE any network I/O.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx
from pydantic import BaseModel, Field

from friday.errors import FridayError
from friday.tools.base import ToolError, ToolResult

logger = logging.getLogger("friday.integrations.comms")

#: Base of the Twilio REST API (the account id + ``/Messages.json`` is appended).
_TWILIO_BASE = "https://api.twilio.com/2010-04-01/Accounts"
#: Default per-request wall-clock budget (seconds) for every Twilio REST call.
_DEFAULT_TIMEOUT = 15.0
#: Twilio's WhatsApp channel address prefix for both ``From`` and ``To``.
_WHATSAPP_PREFIX = "whatsapp:"


class CommsError(FridayError):
    """A Twilio Messages call failed (missing creds, transport, or non-2xx)."""


class TwilioComms:
    """Async ``httpx`` client for the Twilio Messages API (SMS + WhatsApp).

    Args:
        account_sid: The Twilio account SID, or an empty string when unset. Used
            as the basic-auth username and in the request path; a missing value
            raises a clear :class:`CommsError` BEFORE any network I/O.
        auth_token: The Twilio auth token, or an empty string when unset. Used as
            the basic-auth password; a missing value raises before any I/O.
        from_number: The Twilio sender phone number (the ``From`` field), or an
            empty string when unset; a missing value raises before any I/O.
        http: An injected ``httpx.AsyncClient`` the caller owns (so the client is
            trivially testable with ``respx`` and shares connection pooling).
        timeout: Per-request wall-clock budget in seconds.
    """

    def __init__(
        self,
        account_sid: str,
        auth_token: str,
        from_number: str,
        *,
        http: httpx.AsyncClient,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        self._account_sid = account_sid
        self._auth_token = auth_token
        self._from_number = from_number
        self._http = http
        self._timeout = timeout

    @property
    def is_configured(self) -> bool:
        """Whether all three credentials are present (without exposing them)."""
        return bool(self._account_sid and self._auth_token and self._from_number)

    def __repr__(self) -> str:
        """A secret-free repr (the SID/token never leak into repr/str/logs)."""
        return f"TwilioComms(is_configured={self.is_configured})"

    async def aclose(self) -> None:
        """Close the underlying ``httpx.AsyncClient`` (when the client owns it)."""
        await self._http.aclose()

    def _require_creds(self) -> None:
        """Raise a clear :class:`CommsError` when any credential is missing.

        The token is never named here — the message points at the env vars, not
        the values.
        """
        if not self.is_configured:
            raise CommsError(
                "twilio credentials not set — configure FRIDAY_TWILIO_ACCOUNT_SID, "
                "FRIDAY_TWILIO_AUTH_TOKEN, and FRIDAY_TWILIO_FROM_NUMBER to send "
                "messages"
            )

    def _messages_url(self) -> str:
        """The account-scoped Messages endpoint for the configured SID."""
        return f"{_TWILIO_BASE}/{self._account_sid}/Messages.json"

    async def _send(self, to: str, body: str, *, channel: str) -> dict[str, Any]:
        """POST a message to Twilio and return the parsed message JSON.

        ``channel`` is ``"sms"`` or ``"whatsapp"``; for WhatsApp both the sender
        and recipient are prefixed with ``whatsapp:``. Raises :class:`CommsError`
        when credentials are missing (before any I/O), on a transport error, or
        on a non-2xx response.
        """
        self._require_creds()

        if channel == "whatsapp":
            from_value = f"{_WHATSAPP_PREFIX}{self._from_number}"
            to_value = f"{_WHATSAPP_PREFIX}{to}"
        else:
            from_value = self._from_number
            to_value = to

        form = {"To": to_value, "From": from_value, "Body": body}
        # The SID/token ride ONLY in the basic-auth credentials, never logged.
        auth = httpx.BasicAuth(self._account_sid, self._auth_token)
        try:
            response = await self._http.post(
                self._messages_url(),
                data=form,
                auth=auth,
                timeout=self._timeout,
            )
        except httpx.HTTPError as exc:
            logger.warning("twilio %s send transport error: %s", channel, exc)
            raise CommsError(f"twilio {channel} send failed: {exc}") from exc

        if not response.is_success:
            logger.warning(
                "twilio %s send returned HTTP %d", channel, response.status_code
            )
            raise CommsError(
                f"twilio {channel} send returned HTTP {response.status_code}: "
                f"{_safe_body(response)}"
            )

        try:
            parsed = response.json()
        except ValueError as exc:
            raise CommsError(
                f"twilio {channel} send returned a non-JSON body: {exc}"
            ) from exc
        if not isinstance(parsed, dict):
            raise CommsError(
                f"twilio {channel} send returned an unexpected (non-object) body"
            )
        return parsed

    async def send_sms(self, to: str, body: str) -> dict[str, Any]:
        """Send an SMS to ``to`` with text ``body``; return the message JSON."""
        return await self._send(to, body, channel="sms")

    async def send_whatsapp(self, to: str, body: str) -> dict[str, Any]:
        """Send a WhatsApp message to ``to`` with text ``body`` (``whatsapp:`` prefixed)."""
        return await self._send(to, body, channel="whatsapp")


class SendMessageArgs(BaseModel):
    """Arguments for :class:`SendSmsTool` / :class:`SendWhatsappTool`."""

    to: str = Field(min_length=1, description="Recipient phone number (E.164).")
    body: str = Field(min_length=1, max_length=1600, description="Message text.")


async def _run_tool(
    comms: TwilioComms, args: SendMessageArgs, *, channel: str
) -> ToolResult:
    """Drive a Twilio send for a tool, mapping failures to a typed result.

    A :class:`CommsError` (missing creds / transport / non-2xx) becomes
    ``ToolResult(ok=False, error=ToolError(code="comms_failed"))`` rather than a
    raised exception — the error message never carries a secret.
    """
    try:
        if channel == "whatsapp":
            message = await comms.send_whatsapp(args.to, args.body)
        else:
            message = await comms.send_sms(args.to, args.body)
    except CommsError as exc:
        return ToolResult(
            ok=False,
            error=ToolError(code="comms_failed", message=str(exc), retriable=False),
        )
    return ToolResult(ok=True, data=dict(message), error=None)


class SendSmsTool:
    """Send an SMS via Twilio (side-effecting, confirm-step gated)."""

    name = "send_sms"
    description = "Send an SMS text message to a phone number via Twilio."
    args_model = SendMessageArgs
    required_permission = "comms"
    idempotent = False
    side_effecting = True

    async def __call__(self, args: Any, *, comms: TwilioComms) -> ToolResult:
        """Send the SMS via the injected client; failures become a typed result."""
        if not isinstance(args, SendMessageArgs):
            args = SendMessageArgs.model_validate(args)
        return await _run_tool(comms, args, channel="sms")


class SendWhatsappTool:
    """Send a WhatsApp message via Twilio (side-effecting, confirm-step gated)."""

    name = "send_whatsapp"
    description = "Send a WhatsApp message to a phone number via Twilio."
    args_model = SendMessageArgs
    required_permission = "comms"
    idempotent = False
    side_effecting = True

    async def __call__(self, args: Any, *, comms: TwilioComms) -> ToolResult:
        """Send the WhatsApp message via the injected client (typed result)."""
        if not isinstance(args, SendMessageArgs):
            args = SendMessageArgs.model_validate(args)
        return await _run_tool(comms, args, channel="whatsapp")


def _safe_body(response: httpx.Response) -> str:
    """A short, token-free snippet of a response body for an error message.

    Truncated so a large/HTML error page does not bloat the raised error. Carries
    no FRIDAY secret (the SID/token are only ever in the basic-auth header).
    """
    text = response.text or ""
    snippet = text.strip().replace("\n", " ")
    return snippet[:200]
