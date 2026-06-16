# © Lakshya Badjatya — Author
"""Unit tests for external audit anchoring (tamper detection via pinned head)."""

from __future__ import annotations

import pytest

from friday.security.anchor import make_anchor, verify_anchor


def test_make_anchor_stamps_fields() -> None:
    anchor = make_anchor("deadbeef", now=123.0, note="printed")
    assert anchor.head_hash == "deadbeef"
    assert anchor.ts == 123.0
    assert anchor.note == "printed"


def test_blank_hash_rejected() -> None:
    with pytest.raises(ValueError, match="blank"):
        make_anchor("   ", now=0.0)


def test_verify_true_when_hash_still_in_chain() -> None:
    anchor = make_anchor("h2", now=0.0)
    assert verify_anchor(anchor, ["h0", "h1", "h2", "h3"]) is True


def test_verify_false_when_history_rewritten() -> None:
    anchor = make_anchor("h2", now=0.0)
    # The anchored hash is gone -> the chain up to that point was rewritten.
    assert verify_anchor(anchor, ["x0", "x1", "x2"]) is False
