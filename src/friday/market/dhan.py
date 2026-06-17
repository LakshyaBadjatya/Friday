"""A thin ``httpx`` adapter over the Dhan market-data REST API (Tier 3).

:class:`DhanClient` mirrors the keyless-tool style of
:mod:`friday.tools.web_search` and :mod:`friday.n8n.client`: a small async
surface built on an *injected* :class:`httpx.AsyncClient`, with typed failures
(:class:`MarketDataError`) rather than bare exceptions leaking out.

Surface:

* :meth:`DhanClient.quote` — ``POST {base}/v2/marketfeed/quote`` for a single
  ``"EXCHANGE_SEGMENT:SECURITY_ID"`` symbol (e.g. ``"NSE_EQ:11536"``). Returns
  the per-instrument quote object that Dhan nests under
  ``data[segment][security_id]``. If that object is absent (an empty / unexpected
  envelope) it raises rather than fabricating a quote.
* :meth:`DhanClient.historical` — ``POST {base}/v2/charts/historical`` for daily
  OHLCV candles. Returns the parsed candle arrays (``open``/``high``/``low``/
  ``close``/``volume``/``timestamp``).

Both calls send the documented Dhan auth headers (``access-token`` and
``client-id``). When either credential is missing they raise a clear
"configure Dhan" :class:`MarketDataError` BEFORE any network I/O — the same
anti-fabrication contract the Analysis agent enforces: the client returns real
API data or an honest error, never an invented number.

SECURITY: the two credentials are held as plain ``str | None`` here but originate
from :class:`~pydantic.SecretStr` fields in config so they never log. They are
placed ONLY in the request headers; the client's ``repr`` deliberately hides
them, and no log line or error message includes them.

:class:`MarketDataTool` wraps :meth:`DhanClient.quote` as a read-only
(``side_effecting=False``), idempotent FRIDAY tool: success carries the real
quote, any failure becomes ``ToolResult(ok=False, error=...)`` (never a fake
result).
"""

from __future__ import annotations

import logging
from typing import Any

import httpx
from pydantic import BaseModel, Field

from friday.errors import FridayError
from friday.tools.base import ToolError, ToolResult

logger = logging.getLogger("friday.market.dhan")

#: Default base URL of the Dhan REST API.
DEFAULT_BASE_URL = "https://api.dhan.co"

#: HTTP statuses we treat as transient (worth surfacing as ``retriable=True``).
_TRANSIENT_STATUSES = frozenset({429, 500, 502, 503, 504})

#: The clear, secret-free message used whenever Dhan credentials are absent.
_CONFIGURE_MESSAGE = (
    "configure Dhan — set FRIDAY_DHAN_CLIENT_ID and FRIDAY_DHAN_ACCESS_TOKEN to "
    "use market data"
)


class MarketDataError(FridayError):
    """A Dhan market-data call failed (missing creds, transport, or non-2xx).

    ``status_code`` carries the HTTP status for a non-2xx response (``None`` for
    transport/credential/parse failures) so retriability is decided structurally
    rather than by scraping digits out of the error message.
    """

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def _parse_symbol(symbol: str) -> tuple[str, int]:
    """Split ``"EXCHANGE_SEGMENT:SECURITY_ID"`` into ``(segment, security_id)``.

    Dhan addresses an instrument by an exchange segment (e.g. ``NSE_EQ``) plus a
    numeric security id. A symbol that is not in that shape — or whose id is not
    an integer — is a deterministic caller error (never a fabricated lookup), so
    it raises :class:`MarketDataError`.
    """
    segment, sep, raw_id = symbol.partition(":")
    if not sep or not segment or not raw_id:
        raise MarketDataError(
            f"invalid symbol {symbol!r}: expected 'EXCHANGE_SEGMENT:SECURITY_ID' "
            "(e.g. 'NSE_EQ:11536')"
        )
    try:
        security_id = int(raw_id)
    except ValueError as exc:
        raise MarketDataError(
            f"invalid symbol {symbol!r}: security id must be an integer"
        ) from exc
    return segment, security_id


class DhanClient:
    """Async ``httpx`` client for the subset of the Dhan REST API FRIDAY uses.

    Args:
        client_id: The Dhan client id, or ``None``/empty when unset. Sent as the
            ``client-id`` header.
        access_token: The Dhan access token (JWT), or ``None``/empty when unset.
            Sent as the ``access-token`` header.
        http: An injected :class:`httpx.AsyncClient` (so tests can mock the
            transport with ``respx``); its ``base_url`` should target Dhan.
        base_url: Base URL used when the injected client has no ``base_url`` set;
            a trailing slash is stripped so path joining is unambiguous.

    Both credentials must be present for any call; otherwise the method raises a
    clear "configure Dhan" :class:`MarketDataError` before touching the network.
    """

    def __init__(
        self,
        client_id: str | None,
        access_token: str | None,
        *,
        http: httpx.AsyncClient,
        base_url: str = DEFAULT_BASE_URL,
    ) -> None:
        self._client_id = client_id or ""
        self._access_token = access_token or ""
        self._http = http
        self._base_url = base_url.rstrip("/")

    def __repr__(self) -> str:
        """A secret-free repr (never exposes the client id or access token)."""
        configured = "yes" if self.has_credentials else "no"
        return f"DhanClient(configured={configured})"

    @property
    def has_credentials(self) -> bool:
        """Whether both Dhan credentials are configured (without exposing them)."""
        return bool(self._client_id and self._access_token)

    def _auth_headers(self) -> dict[str, str]:
        """The authenticated-request headers; raises when a credential is missing.

        The secrets are placed ONLY here, in the documented Dhan headers — never
        in a log line or an error message.
        """
        if not self.has_credentials:
            raise MarketDataError(_CONFIGURE_MESSAGE)
        return {
            "access-token": self._access_token,
            "client-id": self._client_id,
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    def _url(self, path: str) -> str:
        """Join ``path`` onto the base URL (the injected client may also set one)."""
        return f"{self._base_url}{path}"

    async def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        """``POST`` ``payload`` to ``path``; return the parsed JSON object.

        Raises :class:`MarketDataError` on a transport error, a non-2xx response,
        or a non-object JSON body. Never fabricates a result on failure.
        """
        headers = self._auth_headers()
        try:
            response = await self._http.post(
                self._url(path), json=payload, headers=headers
            )
        except httpx.HTTPError as exc:
            logger.warning("dhan POST %s transport error: %s", path, exc)
            raise MarketDataError(f"Dhan request failed: {exc}") from exc

        if not response.is_success:
            logger.warning("dhan POST %s returned HTTP %d", path, response.status_code)
            raise MarketDataError(
                f"Dhan returned HTTP {response.status_code}: {_safe_body(response)}",
                status_code=response.status_code,
            )

        try:
            body = response.json()
        except ValueError as exc:
            raise MarketDataError(f"Dhan returned a non-JSON body: {exc}") from exc
        if not isinstance(body, dict):
            raise MarketDataError("Dhan returned an unexpected (non-object) body")
        return body

    async def quote(self, symbol: str) -> dict[str, Any]:
        """Fetch the full quote for one ``"SEGMENT:SECURITY_ID"`` symbol.

        ``POST {base}/v2/marketfeed/quote`` with ``{segment: [security_id]}`` and
        the Dhan auth headers. Returns the per-instrument quote object Dhan nests
        under ``data[segment][security_id]`` (LTP, OHLC, volume, depth, ...).

        Raises :class:`MarketDataError` when credentials are missing, the symbol
        is malformed, the call fails, or the instrument is absent from the
        envelope — it never invents a quote.
        """
        segment, security_id = _parse_symbol(symbol)
        body = await self._post(
            "/v2/marketfeed/quote", {segment: [security_id]}
        )
        data = body.get("data")
        if not isinstance(data, dict):
            raise MarketDataError("Dhan quote response had no 'data' object")
        segment_data = data.get(segment)
        if not isinstance(segment_data, dict):
            raise MarketDataError(
                f"Dhan quote response had no data for segment {segment!r}"
            )
        quote = segment_data.get(str(security_id))
        if not isinstance(quote, dict):
            raise MarketDataError(
                f"Dhan returned no quote for {symbol!r} (instrument not found)"
            )
        return quote

    async def historical(
        self,
        symbol: str,
        *,
        instrument: str,
        from_date: str,
        to_date: str,
        expiry_code: int = 0,
        oi: bool = False,
    ) -> dict[str, Any]:
        """Fetch daily OHLCV candles for one ``"SEGMENT:SECURITY_ID"`` symbol.

        ``POST {base}/v2/charts/historical``; returns the parsed candle arrays
        (``open``/``high``/``low``/``close``/``volume``/``timestamp``). ``from_date``
        / ``to_date`` are ``YYYY-MM-DD`` strings; ``to_date`` is non-inclusive per
        the Dhan API. Raises :class:`MarketDataError` on missing creds / malformed
        symbol / transport / non-2xx — never fabricates candles.
        """
        segment, security_id = _parse_symbol(symbol)
        payload: dict[str, Any] = {
            "securityId": str(security_id),
            "exchangeSegment": segment,
            "instrument": instrument,
            "expiryCode": expiry_code,
            "oi": oi,
            "fromDate": from_date,
            "toDate": to_date,
        }
        return await self._post("/v2/charts/historical", payload)


def _safe_body(response: httpx.Response) -> str:
    """A short, secret-free snippet of a response body for an error message.

    Truncated so a large/HTML error page does not bloat the raised error. Carries
    no FRIDAY secret (the credentials are only ever in the request headers).
    """
    text = response.text or ""
    snippet = text.strip().replace("\n", " ")
    return snippet[:200]


class QuoteArgs(BaseModel):
    """Arguments for :class:`MarketDataTool`."""

    symbol: str = Field(min_length=1, description="'EXCHANGE_SEGMENT:SECURITY_ID'")


class MarketDataTool:
    """Read-only FRIDAY tool wrapping :meth:`DhanClient.quote`.

    Side-effect-free and idempotent: it only reads a live quote. On success the
    result carries the REAL Dhan quote; on any failure it returns
    ``ToolResult(ok=False, error=...)`` — the same anti-fabrication contract the
    Analysis agent enforces (real API data or an honest error, never a fake
    number). Transient HTTP statuses are marked ``retriable``; a missing-creds or
    malformed-symbol failure is not.
    """

    name = "market_quote"
    description = (
        "Fetch a live market quote (LTP/OHLC/volume) for an instrument by its "
        "'EXCHANGE_SEGMENT:SECURITY_ID' symbol via the Dhan broker API."
    )
    args_model = QuoteArgs
    required_permission = "market_data"
    idempotent = True
    side_effecting = False

    def __init__(self, client: DhanClient) -> None:
        self._client = client

    async def __call__(self, args: Any) -> ToolResult:
        """Fetch a quote and return a normalized :class:`ToolResult`."""
        if not isinstance(args, QuoteArgs):
            args = QuoteArgs.model_validate(args)

        try:
            quote = await self._client.quote(args.symbol)
        except MarketDataError as exc:
            message = str(exc)
            # Decide retriability from the structured HTTP status, not by scanning
            # the message text (a body snippet echoing a transient code would
            # otherwise flip the flag). Transport/parse errors carry no status.
            retriable = exc.status_code in _TRANSIENT_STATUSES
            return ToolResult(
                ok=False,
                data={"symbol": args.symbol},
                error=ToolError(
                    code="market_data_failed",
                    message=message,
                    retriable=retriable,
                ),
            )

        return ToolResult(
            ok=True,
            data={"symbol": args.symbol, "quote": quote},
            error=None,
        )
