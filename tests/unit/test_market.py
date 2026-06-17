"""Unit tests for the market-data slice (Tier 3; default off).

Fully offline: every Dhan REST call is ``respx``-mocked against an injected
``httpx.AsyncClient`` — no live network. The tests prove that

* :class:`~friday.market.dhan.DhanClient.quote` parses the real Dhan
  ``/v2/marketfeed/quote`` envelope into the per-instrument quote dict, sending
  the documented ``access-token`` / ``client-id`` auth headers;
* :meth:`~friday.market.dhan.DhanClient.historical` parses the
  ``/v2/charts/historical`` OHLCV arrays;
* missing credentials raise a clear "configure Dhan" :class:`MarketDataError`
  BEFORE any network I/O (no fabricated numbers, ever);
* the secrets never leak into the client's ``repr``;
* :class:`~friday.market.dhan.MarketDataTool` is read-only and surfaces real
  API data on success and an honest ``ToolResult(ok=False, ...)`` on failure —
  exactly the Analysis-agent anti-fabrication contract.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
import respx

from friday.market.dhan import DhanClient, MarketDataError, MarketDataTool, QuoteArgs
from friday.tools.base import ToolResult

BASE_URL = "https://api.dhan.co"

# A trimmed but structurally faithful Dhan ``/v2/marketfeed/quote`` envelope for
# one NSE equity instrument (security id 11536 on the NSE_EQ segment).
_QUOTE_BODY = {
    "data": {
        "NSE_EQ": {
            "11536": {
                "last_price": 4525.55,
                "net_change": 17.7,
                "volume": 12345,
                "ohlc": {
                    "open": 4521.45,
                    "close": 4507.85,
                    "high": 4530,
                    "low": 4500,
                },
            }
        }
    },
    "status": "success",
}

# A trimmed Dhan ``/v2/charts/historical`` daily-candle response.
_HISTORICAL_BODY = {
    "open": [3978, 3856, 3925],
    "high": [3978, 3925, 3929],
    "low": [3861, 3856, 3836.55],
    "close": [3879.85, 3915.9, 3859.9],
    "volume": [3937092, 1906106, 3203744],
    "timestamp": [1326220200, 1326306600, 1326393000],
}


def _client(http: httpx.AsyncClient) -> DhanClient:
    return DhanClient("1000000001", "JWT-TOKEN", http=http)


# --------------------------------------------------------------------------- #
# QuoteArgs / tool metadata
# --------------------------------------------------------------------------- #


def test_market_tool_is_read_only() -> None:
    async def _build() -> MarketDataTool:
        async with httpx.AsyncClient() as http:
            return MarketDataTool(_client(http))

    tool = asyncio.run(_build())
    assert tool.name == "market_quote"
    assert tool.side_effecting is False
    assert tool.idempotent is True
    assert tool.args_model is QuoteArgs


def test_quote_args_require_symbol() -> None:
    with pytest.raises(ValueError):
        QuoteArgs(symbol="")


# --------------------------------------------------------------------------- #
# DhanClient.quote
# --------------------------------------------------------------------------- #


@respx.mock
async def test_quote_parses_envelope_and_sends_auth_headers() -> None:
    route = respx.post(f"{BASE_URL}/v2/marketfeed/quote").mock(
        return_value=httpx.Response(200, json=_QUOTE_BODY)
    )
    async with httpx.AsyncClient(base_url=BASE_URL) as http:
        quote = await _client(http).quote("NSE_EQ:11536")

    assert quote["last_price"] == 4525.55
    assert quote["ohlc"]["close"] == 4507.85

    # The documented Dhan auth headers were sent (and the body targets the right
    # exchange segment / security id).
    request = route.calls.last.request
    assert request.headers["access-token"] == "JWT-TOKEN"
    assert request.headers["client-id"] == "1000000001"
    assert json.loads(request.content) == {"NSE_EQ": [11536]}


@respx.mock
async def test_quote_missing_instrument_in_envelope_raises() -> None:
    # A 200 with the security id absent must NOT fabricate a quote.
    respx.post(f"{BASE_URL}/v2/marketfeed/quote").mock(
        return_value=httpx.Response(200, json={"data": {"NSE_EQ": {}}, "status": "success"})
    )
    async with httpx.AsyncClient(base_url=BASE_URL) as http:
        with pytest.raises(MarketDataError):
            await _client(http).quote("NSE_EQ:11536")


@respx.mock
async def test_quote_non_2xx_raises_with_status() -> None:
    respx.post(f"{BASE_URL}/v2/marketfeed/quote").mock(
        return_value=httpx.Response(401, text="unauthorized")
    )
    async with httpx.AsyncClient(base_url=BASE_URL) as http:
        with pytest.raises(MarketDataError) as exc_info:
            await _client(http).quote("NSE_EQ:11536")
    assert "401" in str(exc_info.value)


async def test_quote_bad_symbol_raises() -> None:
    # A symbol without the ``SEGMENT:SECURITY_ID`` shape is a deterministic error,
    # not a network call (it raises before any HTTP is attempted).
    async with httpx.AsyncClient(base_url=BASE_URL) as http:
        with pytest.raises(MarketDataError):
            await _client(http).quote("not-a-symbol")


# --------------------------------------------------------------------------- #
# Missing credentials -> clear "configure Dhan" error, no network
# --------------------------------------------------------------------------- #


async def test_quote_missing_credentials_is_configure_error() -> None:
    async with httpx.AsyncClient(base_url=BASE_URL) as http:
        client = DhanClient("", "", http=http)
        with pytest.raises(MarketDataError) as exc_info:
            await client.quote("NSE_EQ:11536")
    message = str(exc_info.value).lower()
    assert "configure dhan" in message


async def test_historical_missing_credentials_is_configure_error() -> None:
    async with httpx.AsyncClient(base_url=BASE_URL) as http:
        client = DhanClient(None, None, http=http)
        with pytest.raises(MarketDataError) as exc_info:
            await client.historical(
                "NSE_EQ:1333",
                instrument="EQUITY",
                from_date="2022-01-08",
                to_date="2022-02-08",
            )
    assert "configure dhan" in str(exc_info.value).lower()


# --------------------------------------------------------------------------- #
# DhanClient.historical
# --------------------------------------------------------------------------- #


@respx.mock
async def test_historical_parses_ohlcv_arrays() -> None:
    route = respx.post(f"{BASE_URL}/v2/charts/historical").mock(
        return_value=httpx.Response(200, json=_HISTORICAL_BODY)
    )
    async with httpx.AsyncClient(base_url=BASE_URL) as http:
        candles = await _client(http).historical(
            "NSE_EQ:1333",
            instrument="EQUITY",
            from_date="2022-01-08",
            to_date="2022-02-08",
        )

    assert candles["close"] == [3879.85, 3915.9, 3859.9]
    assert candles["volume"] == [3937092, 1906106, 3203744]

    import json

    sent = json.loads(route.calls.last.request.content)
    assert sent["securityId"] == "1333"
    assert sent["exchangeSegment"] == "NSE_EQ"
    assert sent["instrument"] == "EQUITY"
    assert sent["fromDate"] == "2022-01-08"
    assert sent["toDate"] == "2022-02-08"


# --------------------------------------------------------------------------- #
# Secrets never leak
# --------------------------------------------------------------------------- #


async def test_client_repr_hides_secrets() -> None:
    async with httpx.AsyncClient() as http:
        client = DhanClient("1000000001", "super-secret-token", http=http)
        text = repr(client)
    assert "super-secret-token" not in text
    assert "1000000001" not in text


# --------------------------------------------------------------------------- #
# MarketDataTool: real data on success, honest failure (never fabricates)
# --------------------------------------------------------------------------- #


@respx.mock
async def test_tool_returns_real_quote_on_success() -> None:
    respx.post(f"{BASE_URL}/v2/marketfeed/quote").mock(
        return_value=httpx.Response(200, json=_QUOTE_BODY)
    )
    async with httpx.AsyncClient(base_url=BASE_URL) as http:
        tool = MarketDataTool(_client(http))
        result = await tool(QuoteArgs(symbol="NSE_EQ:11536"))

    assert isinstance(result, ToolResult)
    assert result.ok is True
    assert result.error is None
    assert result.data["symbol"] == "NSE_EQ:11536"
    assert result.data["quote"]["last_price"] == 4525.55


@respx.mock
async def test_tool_surfaces_honest_failure_without_fabricating() -> None:
    respx.post(f"{BASE_URL}/v2/marketfeed/quote").mock(
        return_value=httpx.Response(500, text="boom")
    )
    async with httpx.AsyncClient(base_url=BASE_URL) as http:
        tool = MarketDataTool(_client(http))
        result = await tool(QuoteArgs(symbol="NSE_EQ:11536"))

    assert result.ok is False
    assert result.error is not None
    assert result.error.code == "market_data_failed"
    # No fabricated quote on failure.
    assert "quote" not in result.data


@respx.mock
async def test_tool_retriable_decided_by_status_not_body_digits() -> None:
    # A terminal 404 whose body merely echoes a transient code ("503") must NOT be
    # marked retriable — retriability is the HTTP status, not a substring of the body.
    respx.post(f"{BASE_URL}/v2/marketfeed/quote").mock(
        return_value=httpx.Response(404, json={"errorCode": "DH-404", "hint": "see 503 page"})
    )
    async with httpx.AsyncClient(base_url=BASE_URL) as http:
        tool = MarketDataTool(_client(http))
        result = await tool(QuoteArgs(symbol="NSE_EQ:11536"))
    assert result.ok is False and result.error is not None
    assert result.error.retriable is False


@respx.mock
async def test_tool_retriable_on_genuine_transient_status() -> None:
    # A real 503 (no transient digits in the body) IS retriable.
    respx.post(f"{BASE_URL}/v2/marketfeed/quote").mock(
        return_value=httpx.Response(503, text="service unavailable")
    )
    async with httpx.AsyncClient(base_url=BASE_URL) as http:
        tool = MarketDataTool(_client(http))
        result = await tool(QuoteArgs(symbol="NSE_EQ:11536"))
    assert result.ok is False and result.error is not None
    assert result.error.retriable is True


async def test_tool_missing_credentials_is_handled_failure() -> None:
    async with httpx.AsyncClient(base_url=BASE_URL) as http:
        tool = MarketDataTool(DhanClient("", "", http=http))
        result = await tool(QuoteArgs(symbol="NSE_EQ:11536"))
    assert result.ok is False
    assert result.error is not None
    assert "configure dhan" in result.error.message.lower()
    assert result.error.retriable is False


async def test_tool_coerces_raw_dict_args() -> None:
    # The registry passes validated args, but the tool must also accept a raw dict.
    async with httpx.AsyncClient(base_url=BASE_URL) as http:
        tool = MarketDataTool(DhanClient("", "", http=http))
        result = await tool({"symbol": "NSE_EQ:11536"})
    assert result.ok is False
