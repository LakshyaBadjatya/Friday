# © Lakshya Badjatya — Author
"""Unit tests for the fail-closed egress allow-list firewall."""

from __future__ import annotations

from friday.security.egress import EgressPolicy


def test_empty_allowlist_denies_all_fail_closed() -> None:
    policy = EgressPolicy([])
    decision = policy.check("https://example.com/x")
    assert decision.allowed is False
    assert "fail-closed" in decision.reason
    assert policy.allows("https://anything.test") is False


def test_exact_and_subdomain_allowed() -> None:
    policy = EgressPolicy(["example.com"])
    assert policy.allows("https://example.com/path?q=1") is True
    assert policy.allows("https://api.example.com") is True  # subdomain
    assert policy.check("https://example.com").host == "example.com"


def test_lookalike_hosts_denied() -> None:
    policy = EgressPolicy(["example.com"])
    assert policy.allows("https://notexample.com") is False
    assert policy.allows("https://example.com.evil.com") is False  # suffix trick


def test_bare_host_and_host_port() -> None:
    policy = EgressPolicy(["example.com"])
    assert policy.allows("example.com") is True
    assert policy.check("example.com:8443").host == "example.com"
    assert policy.allows("example.com:8443") is True


def test_normalization_case_and_dots() -> None:
    policy = EgressPolicy(["  Example.COM. ", "", "  "])
    assert policy.allowed_hosts == frozenset({"example.com"})
    assert policy.allows("https://EXAMPLE.com") is True


def test_unparseable_target_denied() -> None:
    policy = EgressPolicy(["example.com"])
    decision = policy.check("")
    assert decision.allowed is False
    assert decision.host == ""
