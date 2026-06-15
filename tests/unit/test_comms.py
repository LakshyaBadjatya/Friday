"""Unit tests for the Twilio comms slice (Tier 3; default off, OFFLINE).

Every HTTP call is mocked with ``respx`` against the Twilio Messages REST
endpoint — no live network, no real Twilio credentials. The client is a thin
``httpx`` adapter (mirroring :mod:`friday.integrations.calendar` /
:mod:`friday.n8n.client`): the account SID + auth token are held as plain
``str`` values (sourced from :class:`~pydantic.SecretStr` in config so they never
log) and sent ONLY as the Twilio HTTP basic-auth credentials.

Covered:
* :meth:`TwilioComms.send_sms` issues a basic-authed ``POST`` to the account's
  Messages endpoint, form-encoding ``To`` / ``From`` / ``Body``, and returns the
  parsed message JSON.
* :meth:`TwilioComms.send_whatsapp` does the same but prefixes both the sender
  and recipient with ``whatsapp:``.
* Missing credentials (SID / token / from-number) raise a clear
  :class:`CommsError` BEFORE any network I/O — for both SMS and WhatsApp.
* A non-2xx Twilio response surfaces as a clean :class:`CommsError` (not a raw
  ``httpx`` status), and the auth token never appears in the error.
* The credentials never appear in the client's ``repr`` / ``str``.
* :class:`SendSmsTool` / :class:`SendWhatsappTool` are side-effecting and
  non-idempotent (so the registry confirm-step gates them).
"""

from __future__ import annotations

import base64

import httpx
import pytest
import respx

from friday.integrations.comms import (
    CommsError,
    SendSmsTool,
    SendWhatsappTool,
    TwilioComms,
)

_SID = "AC0123456789abcdef"
_TOKEN = "tok-secret-abcdef"
_FROM = "+15005550006"
_MESSAGES_URL = f"https://api.twilio.com/2010-04-01/Accounts/{_SID}/Messages.json"

_SMS_BODY = {
    "sid": "SMxxxx",
    "status": "queued",
    "to": "+15551230000",
    "from": _FROM,
    "body": "hello there",
}


def _expected_basic_auth() -> str:
    raw = f"{_SID}:{_TOKEN}".encode()
    return "Basic " + base64.b64encode(raw).decode()


def _client(http: httpx.AsyncClient) -> TwilioComms:
    return TwilioComms(_SID, _TOKEN, _FROM, http=http)


@respx.mock
async def test_send_sms_posts_and_parses() -> None:
    """``send_sms`` issues a basic-authed POST and returns the parsed JSON."""
    route = respx.post(_MESSAGES_URL).mock(
        return_value=httpx.Response(201, json=_SMS_BODY)
    )

    async with httpx.AsyncClient() as http:
        result = await _client(http).send_sms("+15551230000", "hello there")

    assert result["sid"] == "SMxxxx"
    assert result["status"] == "queued"
    sent = route.calls.last.request
    # Credentials ride ONLY in the basic-auth header.
    assert sent.headers["Authorization"] == _expected_basic_auth()
    # The form body carries To / From / Body (plain numbers for SMS).
    body = sent.content.decode()
    assert "To=%2B15551230000" in body
    assert f"From=%2B{_FROM[1:]}" in body
    assert "Body=hello+there" in body
    assert "whatsapp" not in body


@respx.mock
async def test_send_whatsapp_prefixes_numbers() -> None:
    """``send_whatsapp`` prefixes both From and To with ``whatsapp:``."""
    route = respx.post(_MESSAGES_URL).mock(
        return_value=httpx.Response(201, json=_SMS_BODY)
    )

    async with httpx.AsyncClient() as http:
        result = await _client(http).send_whatsapp("+15551230000", "hi over wa")

    assert result["sid"] == "SMxxxx"
    sent = route.calls.last.request
    assert sent.headers["Authorization"] == _expected_basic_auth()
    body = sent.content.decode()
    # The ``whatsapp:`` prefix is url-encoded as ``whatsapp%3A``.
    assert "To=whatsapp%3A%2B15551230000" in body
    assert f"From=whatsapp%3A%2B{_FROM[1:]}" in body


async def test_send_sms_missing_sid_raises_before_network() -> None:
    """A missing account SID raises a clear CommsError before any I/O."""
    async with httpx.AsyncClient() as http:
        client = TwilioComms("", _TOKEN, _FROM, http=http)
        with pytest.raises(CommsError) as exc:
            await client.send_sms("+15551230000", "hello")
    assert "TWILIO" in str(exc.value)


async def test_send_sms_missing_token_raises_before_network() -> None:
    """A missing auth token raises a clear CommsError before any I/O."""
    async with httpx.AsyncClient() as http:
        client = TwilioComms(_SID, "", _FROM, http=http)
        with pytest.raises(CommsError):
            await client.send_sms("+15551230000", "hello")


async def test_send_sms_missing_from_raises_before_network() -> None:
    """A missing sender number raises a clear CommsError before any I/O."""
    async with httpx.AsyncClient() as http:
        client = TwilioComms(_SID, _TOKEN, "", http=http)
        with pytest.raises(CommsError):
            await client.send_sms("+15551230000", "hello")


async def test_send_whatsapp_missing_creds_raises() -> None:
    """WhatsApp also raises clearly when credentials are missing."""
    async with httpx.AsyncClient() as http:
        client = TwilioComms("", "", "", http=http)
        with pytest.raises(CommsError):
            await client.send_whatsapp("+15551230000", "hi")


@respx.mock
async def test_send_sms_non_2xx_raises_clean_error() -> None:
    """A non-2xx Twilio response surfaces as a CommsError (token not leaked)."""
    respx.post(_MESSAGES_URL).mock(
        return_value=httpx.Response(
            401, json={"code": 20003, "message": "Authenticate"}
        )
    )

    async with httpx.AsyncClient() as http:
        with pytest.raises(CommsError) as exc:
            await _client(http).send_sms("+15551230000", "hello")

    assert "401" in str(exc.value)
    assert _TOKEN not in str(exc.value)


@respx.mock
async def test_send_sms_transport_error_raises_clean_error() -> None:
    """A transport error is wrapped as a CommsError rather than leaking."""
    respx.post(_MESSAGES_URL).mock(side_effect=httpx.ConnectError("boom"))

    async with httpx.AsyncClient() as http:
        with pytest.raises(CommsError):
            await _client(http).send_sms("+15551230000", "hello")


def test_repr_hides_credentials() -> None:
    """The SID/token never appear in repr/str (secret-free repr)."""
    client = TwilioComms(_SID, _TOKEN, _FROM, http=httpx.AsyncClient())
    text = repr(client)
    assert _TOKEN not in text
    assert _SID not in text


def test_sms_tool_is_side_effecting_non_idempotent() -> None:
    """``SendSmsTool`` is side-effecting + non-idempotent (confirm-step gated)."""
    tool = SendSmsTool()
    assert tool.side_effecting is True
    assert tool.idempotent is False
    assert tool.name == "send_sms"


def test_whatsapp_tool_is_side_effecting_non_idempotent() -> None:
    """``SendWhatsappTool`` is side-effecting + non-idempotent (confirm-step gated)."""
    tool = SendWhatsappTool()
    assert tool.side_effecting is True
    assert tool.idempotent is False
    assert tool.name == "send_whatsapp"


@respx.mock
async def test_sms_tool_sends_via_client() -> None:
    """``SendSmsTool`` builds a client and returns ``ok`` with the message sid."""
    respx.post(_MESSAGES_URL).mock(return_value=httpx.Response(201, json=_SMS_BODY))

    tool = SendSmsTool()
    async with httpx.AsyncClient() as http:
        result = await tool(
            {"to": "+15551230000", "body": "hello there"},
            comms=_client(http),
        )

    assert result.ok is True
    assert result.data["sid"] == "SMxxxx"
    assert result.error is None


async def test_sms_tool_missing_creds_returns_error_result() -> None:
    """A missing-creds failure becomes ``ToolResult(ok=False)``, not a raise."""
    tool = SendSmsTool()
    async with httpx.AsyncClient() as http:
        result = await tool(
            {"to": "+15551230000", "body": "hi"},
            comms=TwilioComms("", "", "", http=http),
        )

    assert result.ok is False
    assert result.error is not None
    assert result.error.code == "comms_failed"
