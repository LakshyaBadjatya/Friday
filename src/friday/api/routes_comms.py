"""``/comms`` — the flagged Twilio messaging surface (Tier 3; default off).

Two surfaces, both gated behind ``FRIDAY_ENABLE_COMMS`` (read lazily off
:func:`~friday.config.get_settings` so the router works mounted on a bare
``FastAPI()`` app, before ``app.py`` wiring exists); when the flag is off both
are ``404`` so the feature simply does not exist for callers (mirroring
``/calendar`` / ``/maps`` / ``/studio`` / ``/reminders``):

* ``POST /comms/sms`` ``{to, body}`` -> ``{message}`` — sends an SMS via Twilio
  and returns the created-message JSON.
* ``POST /comms/whatsapp`` ``{to, body}`` -> ``{message}`` — the same, over
  WhatsApp (``whatsapp:``-prefixed addressing).

The Twilio account SID + auth token are :class:`~pydantic.SecretStr` fields on
:class:`~friday.config.Settings`; they are read via ``get_secret_value()`` ONLY
to build the per-request :class:`~friday.integrations.comms.TwilioComms` basic-
auth credentials and are never logged or echoed in a response. Missing
credentials (or any Twilio REST failure) surface as a clean ``400`` JSON error
rather than a leaked ``500``.
"""

from __future__ import annotations

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError

from friday.config import get_settings
from friday.errors import FridayError
from friday.integrations.comms import TwilioComms
from friday.logging import get_logger

logger = get_logger("friday.api.routes_comms")

router = APIRouter()


class SendMessageRequest(BaseModel):
    """JSON body for ``POST /comms/sms`` and ``POST /comms/whatsapp``."""

    to: str = Field(min_length=1, max_length=64)
    body: str = Field(min_length=1, max_length=1600)


def _comms_enabled() -> bool:
    """Whether the comms surface is enabled, read lazily from settings."""
    return bool(getattr(get_settings(), "enable_comms", False))


def _disabled() -> JSONResponse:
    """The canonical ``comms disabled`` 404 response."""
    return JSONResponse(status_code=404, content={"detail": "comms disabled"})


def _build_client() -> TwilioComms:
    """Build a Twilio client from settings (creds read lazily, never logged).

    The SID/token are pulled from their :class:`~pydantic.SecretStr` fields only
    to seed the client; a missing credential does not raise here (the client
    raises a clear :class:`~friday.integrations.comms.CommsError` on first use,
    which the route maps to a clean 400).
    """
    settings = get_settings()
    sid_secret = getattr(settings, "twilio_account_sid", None)
    token_secret = getattr(settings, "twilio_auth_token", None)
    sid = sid_secret.get_secret_value() if sid_secret is not None else ""
    token = token_secret.get_secret_value() if token_secret is not None else ""
    from_number = getattr(settings, "twilio_from_number", "") or ""
    # A fresh AsyncClient per request, owned by the route (closed in the handler).
    return TwilioComms(sid, token, from_number, http=httpx.AsyncClient())


async def _parse_body(request: Request) -> SendMessageRequest | JSONResponse:
    """Validate the JSON body, returning a 422 ``JSONResponse`` on failure."""
    try:
        raw = await request.json()
    except (ValueError, UnicodeDecodeError):
        return JSONResponse(status_code=422, content={"detail": "expected a JSON body"})
    try:
        return SendMessageRequest.model_validate(raw)
    except ValidationError as exc:
        return JSONResponse(status_code=422, content={"detail": str(exc)})


@router.post("/comms/sms", response_model=None)
async def send_sms(request: Request) -> JSONResponse:
    """Send an SMS; 404 when disabled, 422 on bad body, 400 on a creds/API error."""
    if not _comms_enabled():
        return _disabled()

    parsed = await _parse_body(request)
    if isinstance(parsed, JSONResponse):
        return parsed

    client = _build_client()
    try:
        message = await client.send_sms(parsed.to, parsed.body)
    except FridayError as exc:
        logger.warning("comms sms send failed", extra={"error_type": type(exc).__name__})
        return JSONResponse(status_code=400, content={"detail": str(exc)})
    finally:
        await client.aclose()

    return JSONResponse(status_code=200, content={"message": message})


@router.post("/comms/whatsapp", response_model=None)
async def send_whatsapp(request: Request) -> JSONResponse:
    """Send a WhatsApp message; 404 when disabled, 422 bad body, 400 creds/API error."""
    if not _comms_enabled():
        return _disabled()

    parsed = await _parse_body(request)
    if isinstance(parsed, JSONResponse):
        return parsed

    client = _build_client()
    try:
        message = await client.send_whatsapp(parsed.to, parsed.body)
    except FridayError as exc:
        logger.warning(
            "comms whatsapp send failed", extra={"error_type": type(exc).__name__}
        )
        return JSONResponse(status_code=400, content={"detail": str(exc)})
    finally:
        await client.aclose()

    return JSONResponse(status_code=200, content={"message": message})
