"""External-service integrations (Tier 3; each flagged, default off).

This package holds thin ``httpx`` adapters over third-party REST APIs (no heavy
vendor SDKs). Each integration is gated behind its own ``FRIDAY_ENABLE_*`` flag
so the offline default wires no client and exposes no route.

Currently:

* :mod:`friday.integrations.calendar` — a Google Calendar v3 adapter
  (:class:`~friday.integrations.calendar.GoogleCalendarClient`) plus its flagged
  ``/calendar`` REST surface. The OAuth bearer token is a
  :class:`~pydantic.SecretStr` in config (never logged) and is sent only as the
  ``Authorization: Bearer`` header.

The integration agent wires the calendar slice by including
:data:`friday.integrations.calendar_router`.

``calendar_router`` is resolved LAZILY (PEP 562 ``__getattr__``) so importing the
:mod:`friday.integrations.calendar` submodule — which the route module does — does
NOT eagerly pull in ``friday.api.routes_calendar`` and create an import cycle. The
attribute still works exactly like a normal export::

    from friday.integrations import calendar_router
"""

from __future__ import annotations

from typing import Any

__all__ = ["calendar_router"]


def __getattr__(name: str) -> Any:
    """Lazily resolve :data:`calendar_router` to avoid an import cycle.

    The ``/calendar`` route module imports this package's ``calendar`` submodule,
    which runs this ``__init__``; resolving the router eagerly here would import
    ``routes_calendar`` mid-initialisation and raise a circular ``ImportError``.
    Deferring the import to attribute-access time keeps both import orders clean.
    """
    if name == "calendar_router":
        from friday.api.routes_calendar import router

        return router
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Include the lazily-resolved export in ``dir()`` for discoverability."""
    return sorted({*globals().keys(), *__all__})
