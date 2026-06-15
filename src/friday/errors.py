"""Typed error hierarchy for FRIDAY.

All domain errors derive from :class:`FridayError` so callers can catch the
whole family with a single ``except`` while still being able to discriminate
specific failure modes.
"""


class FridayError(Exception):
    """Base for all FRIDAY errors."""


class ProviderError(FridayError):
    """A provider (LLM/STT/TTS) call failed."""


class ToolError(FridayError):
    """A tool execution failed."""


class PermissionError(FridayError):  # noqa: A001 - domain-specific
    """An action was denied by the permission system."""


class RoutingError(FridayError):
    """Intent routing failed."""
