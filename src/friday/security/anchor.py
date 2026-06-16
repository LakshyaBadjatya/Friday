# © Lakshya Badjatya — Author
"""External audit anchoring: pin the ledger head out-of-band to detect rewrites.

The broker's audit ledger is a hash-chained, tamper-evident log — but a
sufficiently determined tamperer who can rewrite the whole file could re-chain it
cleanly. *Anchoring* defends against that by periodically recording the ledger's
current head hash somewhere out-of-band (a printout, an email-to-self, a file on
another machine). Later, the anchored hash is checked against the live chain: if
it is no longer present, history up to that point was rewritten.

This module is the pure core of that: :func:`make_anchor` stamps a head hash with
an injected timestamp, and :func:`verify_anchor` reports whether the anchored hash
still appears in the ledger's chain. It imports no LLM SDK, reads no
configuration, and performs no I/O — the out-of-band *delivery* of an anchor is
the caller's job.
"""

from __future__ import annotations

from collections.abc import Iterable

from pydantic import BaseModel


class AuditAnchor(BaseModel):
    """A pinned ledger head hash plus when (and optionally why) it was anchored.

    Attributes:
        head_hash: The ledger's chain head hash at the moment of anchoring.
        ts: The injected timestamp when the anchor was taken.
        note: An optional human note (e.g. where it was pinned).
    """

    head_hash: str
    ts: float
    note: str = ""


def make_anchor(head_hash: str, *, now: float, note: str = "") -> AuditAnchor:
    """Stamp ``head_hash`` as an anchor at ``now``; raise on a blank hash."""
    cleaned = head_hash.strip()
    if not cleaned:
        raise ValueError("head_hash must not be blank")
    return AuditAnchor(head_hash=cleaned, ts=now, note=note)


def verify_anchor(anchor: AuditAnchor, chain_hashes: Iterable[str]) -> bool:
    """Whether the anchored head hash still appears in the ledger's chain.

    ``chain_hashes`` is the set of record hashes currently in the ledger. If the
    anchored hash is absent, the history up to that point no longer exists as it
    was — evidence the ledger was rewritten since the anchor was taken.
    """
    return anchor.head_hash in set(chain_hashes)
