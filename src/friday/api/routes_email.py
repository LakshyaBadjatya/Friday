"""``/email`` — the flagged Gmail surface (Tier 3; default off).

Two surfaces, both gated behind ``FRIDAY_ENABLE_EMAIL`` (read lazily off
:func:`~friday.config.get_settings` so the router works mounted on a bare
``FastAPI()`` app, before ``app.py`` wiring exists); when the flag is off both
are ``404`` so the feature simply does not exist for callers (mirroring
``/calendar`` / ``/maps`` / ``/studio``):

* ``GET  /email/inbox`` -> ``{messages, count, summary}`` — the inbox messages on
  the user's ``me`` mailbox plus a short, *non-fatal* LLM summary. Read-only.
* ``POST /email/draft`` ``{message_id, body}`` -> ``{draft}`` — creates a **DRAFT**
  reply (never auto-sends) on the original message's thread and returns the
  created-draft JSON. Sending stays a separate, explicit, human-confirmed action;
  there is NO send endpoint on this surface.

The Gmail OAuth bearer token is a :class:`~pydantic.SecretStr` on
:class:`~friday.config.Settings`; it is read via ``get_secret_value()`` ONLY to
build the per-request :class:`~friday.integrations.email.GmailClient` header and
is never logged or echoed in a response. A missing token (or any Gmail REST
failure) surfaces as a clean ``400`` JSON error rather than a leaked ``500``.

The integration agent wires this slice by including ``routes_email.router``
(exported as ``friday.api.routes_email.router``).
"""

from __future__ import annotations

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError

from friday.config import Settings, get_settings
from friday.errors import FridayError
from friday.integrations.email import GmailClient
from friday.logging import get_logger
from friday.providers.llm import (
    FakeLLM,
    FallbackLLM,
    GeminiProvider,
    LLMProvider,
    NvidiaNIMProvider,
)

logger = get_logger("friday.api.routes_email")

router = APIRouter()


class DraftReplyRequest(BaseModel):
    """JSON body for ``POST /email/draft``."""

    message_id: str = Field(min_length=1, max_length=512)
    body: str = Field(min_length=1, max_length=50_000)


def _email_enabled() -> bool:
    """Whether the email surface is enabled, read lazily from settings."""
    return bool(getattr(get_settings(), "enable_email", False))


def _disabled() -> JSONResponse:
    """The canonical ``email disabled`` 404 response."""
    return JSONResponse(status_code=404, content={"detail": "email disabled"})


def _build_client() -> GmailClient:
    """Build a Gmail client from settings (token read lazily, never logged).

    The bearer token is pulled from the :class:`~pydantic.SecretStr` only to seed
    the client; a missing token does not raise here (the client raises a clear
    :class:`~friday.integrations.email.EmailError` on first use, which the route
    maps to a clean 400).
    """
    secret = getattr(get_settings(), "gmail_oauth_token", None)
    token = secret.get_secret_value() if secret is not None else None
    # A fresh AsyncClient per request, owned by the route (closed in the handler).
    return GmailClient(token, http=httpx.AsyncClient())


def build_llm(settings: Settings) -> LLMProvider:
    """Select the LLM provider for the inbox summary from settings.

    Mirrors :func:`friday.app._build_llm` but lives in this slice so the route is
    self-contained (and monkeypatchable in tests) before ``app.py`` wiring exists.
    NVIDIA NIM when explicitly configured *and* a key is present (optionally
    wrapped in a Gemini :class:`FallbackLLM`); otherwise an empty-script
    :class:`FakeLLM`. The summary is non-fatal, so an empty FakeLLM simply trips
    the deterministic fallback inside
    :meth:`~friday.integrations.email.GmailClient.summarize_inbox`.
    """
    if settings.llm_provider == "nvidia" and settings.nvidia_api_key is not None:
        primary: LLMProvider = NvidiaNIMProvider(
            api_key=settings.nvidia_api_key.get_secret_value(),
            base_url=settings.nvidia_base_url,
            model=settings.nvidia_model,
            timeout=settings.llm_timeout_seconds,
        )
        if (
            settings.llm_fallback_provider == "gemini"
            and settings.gemini_api_key is not None
        ):
            secondary = GeminiProvider(
                api_key=settings.gemini_api_key.get_secret_value(),
                base_url=settings.gemini_base_url,
                model=settings.gemini_model,
                timeout=settings.llm_timeout_seconds,
            )
            return FallbackLLM(primary=primary, secondary=secondary)
        return primary
    return FakeLLM(responses=[])


@router.get("/email/inbox", response_model=None)
async def list_inbox(request: Request) -> JSONResponse:
    """List the inbox and summarize it; 404 when disabled, 400 on token/API error.

    The result is ``{messages, count, summary}`` (the parsed message refs, their
    count, and a short non-fatal LLM summary). A missing token or any Gmail REST
    failure surfaces as a clean 400 rather than a leaked 500.
    """
    if not _email_enabled():
        return _disabled()

    client = _build_client()
    try:
        messages = await client.list_messages("in:inbox")
        summary = await client.summarize_inbox(build_llm(get_settings()))
    except FridayError as exc:
        logger.warning("email inbox failed", extra={"error_type": type(exc).__name__})
        return JSONResponse(status_code=400, content={"detail": str(exc)})
    finally:
        await client.aclose()

    return JSONResponse(
        status_code=200,
        content={"messages": messages, "count": len(messages), "summary": summary},
    )


@router.post("/email/draft", response_model=None)
async def draft_reply(request: Request) -> JSONResponse:
    """Create a DRAFT reply; 404 when disabled, 422 on bad body, 400 on token/API error.

    The body is ``{"message_id": "...", "body": "..."}``. On success the
    created-draft JSON is returned as ``{"draft": {...}}``. This NEVER auto-sends —
    it only ever creates a draft (confirm-and-send stays a separate human action).
    """
    if not _email_enabled():
        return _disabled()

    try:
        raw = await request.json()
    except (ValueError, UnicodeDecodeError):
        return JSONResponse(status_code=422, content={"detail": "expected a JSON body"})
    try:
        body = DraftReplyRequest.model_validate(raw)
    except ValidationError as exc:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    client = _build_client()
    try:
        draft = await client.draft_reply(body.message_id, body.body)
    except FridayError as exc:
        logger.warning("email draft failed", extra={"error_type": type(exc).__name__})
        return JSONResponse(status_code=400, content={"detail": str(exc)})
    finally:
        await client.aclose()

    return JSONResponse(status_code=200, content={"draft": draft})
