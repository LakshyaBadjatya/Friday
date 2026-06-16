# © Lakshya Badjatya — Author
"""Egress allow-list: a fail-closed outbound-host firewall, as code.

FRIDAY's promise is that nothing phones home unless you point it somewhere on
purpose. This module turns that convention into an enforceable policy: an
:class:`EgressPolicy` holds an explicit allow-list of hosts, and every outbound
target (a URL or a bare host) is checked against it. The default is **fail
closed** — an empty allow-list denies everything — so a misconfiguration can
only ever block traffic, never silently permit it.

Matching is by host, case-insensitively, with subdomain support: an allow-list
entry ``"example.com"`` permits ``example.com`` and any subdomain
(``api.example.com``) but not a lookalike (``notexample.com``). The policy is
pure and deterministic — it parses the host, consults the frozen allow-list, and
returns a verdict; it performs no network I/O and reads no configuration (the
allow-list is injected by ``app.py`` from settings).
"""

from __future__ import annotations

from collections.abc import Iterable
from urllib.parse import urlparse

from pydantic import BaseModel


class EgressDecision(BaseModel):
    """The verdict on one outbound target.

    Attributes:
        allowed: Whether the target's host is on the allow-list.
        host: The host parsed from the target (empty when it could not be parsed).
        reason: A short, human-readable explanation of the verdict.
    """

    allowed: bool
    host: str
    reason: str


class EgressPolicy:
    """A fail-closed outbound-host allow-list.

    Args:
        allowed_hosts: The hosts traffic may reach. Each is normalized
            (lowercased, surrounding dots/space stripped); blank entries are
            dropped. An empty resulting set denies all egress (fail closed).
    """

    def __init__(self, allowed_hosts: Iterable[str]) -> None:
        self._allowed: frozenset[str] = frozenset(
            h.strip().lower().strip(".") for h in allowed_hosts if h.strip()
        )

    @property
    def allowed_hosts(self) -> frozenset[str]:
        """The normalized allow-list (read-only)."""
        return self._allowed

    @staticmethod
    def _host_of(target: str) -> str:
        """Extract the lowercase host from a URL or bare host:port, else ``""``.

        Adds a ``//`` prefix when the target has no scheme so a bare
        ``host:port`` is parsed as a netloc rather than a path.
        """
        candidate = target.strip()
        if not candidate:
            return ""
        if "//" not in candidate:
            candidate = "//" + candidate
        host = urlparse(candidate).hostname
        return host.lower() if host else ""

    def _is_allowed(self, host: str) -> bool:
        """Whether ``host`` matches the allow-list (exact or as a subdomain)."""
        if not host:
            return False
        return any(
            host == rule or host.endswith("." + rule) for rule in self._allowed
        )

    def check(self, target: str) -> EgressDecision:
        """Return the :class:`EgressDecision` for an outbound ``target``."""
        host = self._host_of(target)
        if not host:
            return EgressDecision(
                allowed=False, host="", reason="could not parse a host from target"
            )
        if self._is_allowed(host):
            return EgressDecision(
                allowed=True, host=host, reason=f"{host} is on the egress allow-list"
            )
        return EgressDecision(
            allowed=False,
            host=host,
            reason=f"{host} is not on the egress allow-list (fail-closed)",
        )

    def allows(self, target: str) -> bool:
        """Shorthand for ``self.check(target).allowed``."""
        return self.check(target).allowed
