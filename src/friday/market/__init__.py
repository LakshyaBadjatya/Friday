"""Market data (Tier 3; default off behind ``FRIDAY_ENABLE_MARKET_DATA``).

This package owns FRIDAY's market-data feature — live quotes (and daily OHLCV
history) from the `Dhan <https://dhanhq.co>`_ broker REST API, talked to over an
injected :class:`httpx.AsyncClient` (NO Dhan SDK). It is off by default, so the
offline build wires no broker client and exposes no ``/market`` route (-> 404).

The two Dhan credentials are :class:`~pydantic.SecretStr` fields on
:class:`~friday.config.Settings` (``dhan_client_id`` / ``dhan_access_token``);
they never log and are sent only as the documented ``client-id`` / ``access-token``
request headers. Missing credentials raise a clear "configure Dhan" error before
any network I/O, and — exactly like the Analysis agent — the client returns real
API data or an honest error, never a fabricated number.

Public surface:

* :class:`~friday.market.dhan.DhanClient` — the thin httpx adapter
  (``quote`` / ``historical``).
* :class:`~friday.market.dhan.MarketDataError` — the typed failure.
* :class:`~friday.market.dhan.MarketDataTool` (+ :class:`~friday.market.dhan.QuoteArgs`)
  — the read-only FRIDAY tool wrapping ``quote``.
* the flagged ``/market`` :data:`router` (re-exported for the integration agent
  to wire — include ``friday.market.router``).

``router`` is resolved LAZILY (PEP 562 ``__getattr__``, mirroring
:mod:`friday.integrations`) so importing the :mod:`friday.market.dhan` submodule —
which the route module does — does NOT eagerly pull in
:mod:`friday.api.routes_market` and create an import cycle. The attribute still
works exactly like a normal export::

    from friday.market import router
"""

from __future__ import annotations

from typing import Any

from friday.market.dhan import (
    DhanClient,
    MarketDataError,
    MarketDataTool,
    QuoteArgs,
)

__all__ = [
    "DhanClient",
    "MarketDataError",
    "MarketDataTool",
    "QuoteArgs",
    "router",
]


def __getattr__(name: str) -> Any:
    """Lazily resolve :data:`router` to avoid an import cycle.

    The ``/market`` route module imports this package's :mod:`friday.market.dhan`
    submodule, which runs this ``__init__``; resolving the router eagerly here
    would import :mod:`friday.api.routes_market` mid-initialisation and raise a
    circular ``ImportError``. Deferring the import to attribute-access time keeps
    both import orders clean.
    """
    if name == "router":
        from friday.api.routes_market import router

        return router
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Include the lazily-resolved export in ``dir()`` for discoverability."""
    return sorted({*globals().keys(), *__all__})
