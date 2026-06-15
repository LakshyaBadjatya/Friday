"""Structured logging for FRIDAY.

Provides JSON log output with a request-scoped ``correlation_id`` (carried in a
:class:`contextvars.ContextVar` so it propagates through async tasks) and
automatic redaction of any log field whose key looks like a secret
(``api_key``, ``token``, ``secret``, ``password``, ``authorization`` — matched
as a case-insensitive substring).
"""

from __future__ import annotations

import json
import logging
import sys
from contextvars import ContextVar
from typing import Any

REDACTED = "***REDACTED***"

# Substrings that mark a field as sensitive. Matched case-insensitively against
# the field name; any field containing one of these is redacted.
_SENSITIVE_SUBSTRINGS: tuple[str, ...] = (
    "api_key",
    "apikey",
    "token",
    "secret",
    "password",
    "authorization",
)

# Standard attributes present on every ``logging.LogRecord``; anything not in
# this set was supplied by the caller via ``extra=`` and is treated as a custom
# field to be serialized (and possibly redacted).
_RESERVED_RECORD_ATTRS: frozenset[str] = frozenset(
    {
        "name",
        "msg",
        "args",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "exc_info",
        "exc_text",
        "stack_info",
        "lineno",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "processName",
        "process",
        "taskName",
    }
)

_correlation_id: ContextVar[str | None] = ContextVar("correlation_id", default=None)


def bind_correlation_id(cid: str | None) -> None:
    """Bind a correlation id for the current context (async-safe)."""
    _correlation_id.set(cid)


def _is_sensitive(key: str) -> bool:
    lowered = key.lower()
    return any(token in lowered for token in _SENSITIVE_SUBSTRINGS)


def _redact(key: str, value: Any) -> Any:
    return REDACTED if _is_sensitive(key) else value


class JsonFormatter(logging.Formatter):
    """Format log records as a single line of JSON with redaction applied."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": _correlation_id.get(),
        }

        for key, value in record.__dict__.items():
            if key in _RESERVED_RECORD_ATTRS or key in payload:
                continue
            payload[key] = _redact(key, value)

        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


class RedactingTextFormatter(logging.Formatter):
    """Plain-text formatter that still prepends the correlation id."""

    def format(self, record: logging.LogRecord) -> str:
        cid = _correlation_id.get()
        base = super().format(record)
        return f"[cid={cid}] {base}"


def configure_logging(json_logs: bool = True, level: str = "INFO") -> None:
    """Configure the root logger to emit to stderr at ``level``.

    Replaces any existing handlers so repeated calls (e.g. in tests) are
    idempotent.
    """
    root = logging.getLogger()
    root.setLevel(level)

    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.StreamHandler(stream=sys.stderr)
    formatter: logging.Formatter
    if json_logs:
        formatter = JsonFormatter()
    else:
        formatter = RedactingTextFormatter(
            "%(asctime)s %(levelname)s %(name)s: %(message)s"
        )
    handler.setFormatter(formatter)
    root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger."""
    return logging.getLogger(name)
