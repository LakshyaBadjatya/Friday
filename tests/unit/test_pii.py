# © Lakshya Badjatya — Author
"""Unit tests for PII redaction (email / card / ip / phone)."""

from __future__ import annotations

from friday.security.pii import PIIRedactor


def test_redacts_each_kind() -> None:
    r = PIIRedactor()
    assert r.scrub("write a.b+x@example.co.uk") == "write [EMAIL]"
    assert r.scrub("card 4111 1111 1111 1111 ok") == "card [CARD] ok"
    assert r.scrub("host 192.168.0.1 down") == "host [IP] down"
    assert r.scrub("call (555) 123-4567") == "call [PHONE]"


def test_card_before_phone_ordering() -> None:
    # A 16-digit card must redact as [CARD], not be nibbled by the phone rule.
    assert PIIRedactor().scrub("4111111111111111") == "[CARD]"


def test_counts_and_mixed_text() -> None:
    text = "mail a@b.com, card 4111111111111111, ip 10.0.0.1, ph 555-123-4567"
    result = PIIRedactor().redact(text)
    assert result.counts == {"email": 1, "card": 1, "ip": 1, "phone": 1}
    assert "@" not in result.text
    assert "4111" not in result.text
    assert "10.0.0.1" not in result.text


def test_contains_pii() -> None:
    r = PIIRedactor()
    assert r.contains_pii("reach me at x@y.com") is True
    assert r.contains_pii("nothing sensitive here") is False


def test_clean_text_unchanged() -> None:
    r = PIIRedactor()
    out = r.redact("a perfectly ordinary sentence with no secrets")
    assert out.text == "a perfectly ordinary sentence with no secrets"
    assert out.counts == {}
