# © Lakshya Badjatya — Author
"""Secret rotation reminders: flag credentials that are overdue for a refresh.

A vault-aware nudge: given when each secret was last rotated and a maximum
allowed age, this module reports which secrets are now overdue so FRIDAY can
remind the owner to rotate them. It is pure and deterministic — ages are computed
from an injected ``now`` (never the wall clock), it reads no configuration (the
policy window is injected), holds no secret *values* (only names + timestamps),
and performs no I/O.
"""

from __future__ import annotations

from collections.abc import Iterable

from pydantic import BaseModel


class SecretAge(BaseModel):
    """When a named secret was last rotated.

    Carries only the secret's name and its last-rotation timestamp — never the
    secret value — so a rotation report is safe to log or surface.
    """

    name: str
    last_rotated_ts: float


class RotationStatus(BaseModel):
    """A single secret's rotation standing at a given instant."""

    name: str
    age_seconds: float
    due: bool


class RotationPolicy:
    """Decides whether a secret is overdue, given a maximum allowed age.

    Args:
        max_age_seconds: The longest a secret may go un-rotated before it is
            considered due (must be positive).
    """

    def __init__(self, max_age_seconds: float) -> None:
        if max_age_seconds <= 0:
            raise ValueError("max_age_seconds must be positive")
        self._max_age = max_age_seconds

    def status(self, secret: SecretAge, *, now: float) -> RotationStatus:
        """The :class:`RotationStatus` for ``secret`` at ``now``.

        Age is clamped to ``>= 0`` (a future ``last_rotated_ts`` reads as age 0),
        and a secret is ``due`` once its age reaches the policy's max age.
        """
        age = max(0.0, now - secret.last_rotated_ts)
        return RotationStatus(name=secret.name, age_seconds=age, due=age >= self._max_age)

    def due(self, secrets: Iterable[SecretAge], *, now: float) -> list[str]:
        """The names of the secrets overdue for rotation at ``now``, in input order."""
        return [s.name for s in secrets if self.status(s, now=now).due]
