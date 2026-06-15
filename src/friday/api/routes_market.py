"""``/market`` — the flagged market-data REST API (Tier 3; default off).

One surface, gated behind ``FRIDAY_ENABLE_MARKET_DATA`` (read lazily off
:func:`~friday.config.get_settings` so the router works mounted on a bare
``FastAPI()`` app, before ``app.py`` wiring exists); when the flag is off it is
``404`` so the feature simply does not exist for callers (mirroring ``/maps`` and
``/studio``):

* ``GET /market/quote?symbol=`` -> ``{"symbol": ..., "quote": {...}}`` — the live
  quote for one ``"EXCHANGE_SEGMENT:SECURITY_ID"`` instrument (e.g.
  ``NSE_EQ:11536``) fetched from the Dhan broker REST API.

The :class:`~friday.market.dhan.DhanClient` is built LAZILY inside the handler
from settings (the two Dhan credentials are :class:`~pydantic.SecretStr` and are
read via ``get_secret_value()`` ONLY to populate the request headers — never
logged). A :class:`~friday.market.dhan.MarketDataError` (missing creds, transport
error, or a Dhan non-2xx) becomes a clean ``502`` JSON error rather than a leaked
500 — and the route NEVER fabricates a quote on failure.
"""

from __future__ import annotations

import httpx
from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from friday.config import get_settings
from friday.logging import get_logger
from friday.market.dhan import DhanClient, MarketDataError

logger = get_logger("friday.api.routes_market")

router = APIRouter()


def _market_enabled() -> bool:
    """Whether the market-data surface is enabled, read lazily from settings."""
    return bool(getattr(get_settings(), "enable_market_data", False))


def _disabled() -> JSONResponse:
    """The canonical ``market data disabled`` 404 response."""
    return JSONResponse(status_code=404, content={"detail": "market data disabled"})


def _build_client(http: httpx.AsyncClient) -> DhanClient:
    """Build a :class:`DhanClient` lazily from settings (secrets via headers only).

    The two credentials are :class:`~pydantic.SecretStr`; ``get_secret_value()`` is
    called ONLY to hand them to the client, which in turn places them only in the
    Dhan request headers. The transport (``http``) is owned by the caller so its
    lifecycle is managed in one place.
    """
    settings = get_settings()
    client_id = settings.dhan_client_id
    access_token = settings.dhan_access_token
    return DhanClient(
        client_id.get_secret_value() if client_id is not None else None,
        access_token.get_secret_value() if access_token is not None else None,
        http=http,
    )


@router.get("/market/quote", response_model=None)
async def market_quote(symbol: str = Query(min_length=1)) -> JSONResponse:
    """Return the live quote for ``symbol``; 404 when disabled, 502 on a Dhan error.

    ``symbol`` is ``"EXCHANGE_SEGMENT:SECURITY_ID"`` (e.g. ``NSE_EQ:11536``); a
    missing/empty value is a ``422`` (FastAPI validation). A
    :class:`~friday.market.dhan.MarketDataError` — including missing credentials —
    is surfaced as a clean ``502`` with an honest message and NO fabricated quote.
    A fresh :class:`httpx.AsyncClient` is created and closed per request.
    """
    if not _market_enabled():
        return _disabled()

    async with httpx.AsyncClient(timeout=15.0) as http:
        client = _build_client(http)
        try:
            quote = await client.quote(symbol)
        except MarketDataError as exc:
            logger.warning(
                "market quote failed", extra={"error_type": type(exc).__name__}
            )
            return JSONResponse(
                status_code=502,
                content={"error": str(exc), "type": type(exc).__name__},
            )

    return JSONResponse(status_code=200, content={"symbol": symbol, "quote": quote})
