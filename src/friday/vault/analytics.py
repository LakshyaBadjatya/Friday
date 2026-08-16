"""Where the marks are going.

Every solved problem already carries a subject, a chapter, and an independent
verification verdict. Rolled up per chapter, that is a real answer to "what
should I revise" — built from what actually happened rather than from
self-report.

**What "mastery" means now.** :class:`~friday.vault.models.Verification` has
three states — ``VERIFIED``, ``REFUTED``, ``NOT_VERIFIABLE`` — since
:mod:`friday.vault.solver` learned to ask the drafting operator for the
decisive equation of its own solution and check that, rather than the raw OCR
statement. Where a chapter has verifiable solves at all, mastery is
``verified / (verified + refuted)`` — ``NOT_VERIFIABLE`` solves are excluded
from the denominator entirely, not counted as failures. That is the fix for
the bug this module used to document as a caveat: a chapter of entirely
correct chemistry (or any problem shape SymPy can't touch) no longer rolls up
to ``mastery == 0.0`` just because nothing in it happened to be checkable.

**Where a chapter has no verifiable solves at all**, mastery falls back to
ensemble agreement as the confidence signal instead of reporting a number with
zero solves behind it: the fraction of that chapter's solves where the panel
was unanimous, parsed from :attr:`~friday.vault.models.Consensus.agreement`
(the ``"<votes>/<drafts>"`` string). ``basis`` on :class:`ChapterMastery`
says which signal produced the number — ``"sympy"`` or ``"agreement"`` — so a
caller never has to guess which kind of "mastery" it is looking at. Even
``basis="agreement"`` is a genuinely weaker signal than an independent check —
three operators agreeing is not the same as SymPy re-deriving the answer — so
a caller surfacing this should still not present ``mastery`` as "percent
correct" without knowing which basis produced it.

**Locked items are excluded.** :meth:`~friday.vault.index.VaultIndex.list_items`
defaults to hiding ``Privacy.LOCKED`` items, and this rollup keeps that
default rather than overriding it. :mod:`friday.vault.search` sets the
precedent — its docstring treats "locked items never appear, full stop" as a
security guarantee, not just a model-provider guardrail — and the same
reasoning applies here: a mastery rollup over a locked chapter would still
leak that chapter's existence, its classification, and whether the student
got it right, through an aggregate rather than raw content, which is exactly
what marking something LOCKED is meant to prevent. The alternative — folding
locked items in because mastery stats are meant to reflect what actually
happened — is a real argument, but it is in tension with a privacy signal the
student set deliberately, and this module defers to that signal.

**Known scaling limits, flagged rather than fixed here** (the ``VaultIndex``
protocol has no pagination cursor or batch solve lookup, and changing it is
out of scope for this module):

* ``list_items(..., limit=_MAX_ITEMS)`` truncates silently past ``_MAX_ITEMS``
  items — a vault larger than that undercounts without any error.
* Every item with a ``solve_id`` costs one ``get_solve`` call — N+1 reads. On
  the Firestore backend that is N+1 network round trips per rollup, not one
  bulk read.
* :meth:`~friday.vault.index.VaultIndex.get_solve` can raise on the Firestore
  backend when the service is unreachable (see
  :class:`friday.vault.firestore_index.FirestoreIndexError`, whose docstring
  is explicit that the caller must treat that as a failure, not an empty
  result). This module does not catch it — an exception propagates out of
  :meth:`Analytics.rollup` rather than being swallowed into a quietly
  incomplete rollup. That matches this codebase's existing rule (see
  ``FirestoreIndexError``'s own docstring): it is fine to report short
  results only when the vault genuinely has fewer of them, never when a read
  failed partway through.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel

from friday.vault.index import VaultIndex
from friday.vault.models import VerificationStatus

_MAX_ITEMS = 10_000


class ChapterMastery(BaseModel):
    """One chapter's record."""

    subject: str
    chapter: str
    attempts: int
    verified: int
    verifiable_attempts: int
    """How many of this chapter's solves could actually be checked at all
    (``verified + refuted``) — the denominator ``mastery`` was computed over
    when ``basis == "sympy"``. Zero when ``basis == "agreement"``."""
    basis: str
    """``"sympy"`` when verifiable solves existed and ``mastery`` is
    ``verified / verifiable_attempts``; ``"agreement"`` when none did and
    ``mastery`` falls back to the fraction of unanimous panel votes instead."""
    mastery: float
    last_seen: str


def _is_unanimous(agreement: str) -> bool:
    """Whether ``"<votes>/<drafts>"`` records every draft agreeing.

    Malformed text or a zero-denominator ("0/0", from an empty panel) counts
    as non-unanimous rather than raising — this is untrusted, model-adjacent
    provenance by the time it reaches analytics, not a value this module
    controls the shape of.
    """
    parts = agreement.split("/")
    if len(parts) != 2:
        return False
    try:
        votes, drafts = int(parts[0]), int(parts[1])
    except ValueError:
        return False
    return drafts > 0 and votes == drafts


@dataclass
class _Bucket:
    """Mutable accumulator for one (subject, chapter) key while rolling up."""

    attempts: int = 0
    verified: int = 0
    refuted: int = 0
    unanimous: int = 0
    last_seen: str = ""


class Analytics:
    """Mastery rollups over the solved items in the vault."""

    def __init__(self, index: VaultIndex) -> None:
        self._index = index

    def rollup(self, owner_uid: str) -> list[ChapterMastery]:
        """Per-chapter mastery for one owner, weakest first.

        Only items that carry a ``solve_id`` resolving to a stored
        :class:`~friday.vault.models.Solve` count as attempts. An item
        without a solve was never attempted and contributes nothing; an item
        whose ``solve_id`` points at a solve that is missing (deleted, or
        malformed and skipped by the index's own "log and skip" row-parsing
        rule) is also skipped rather than counted, since there is no
        verification verdict to attach to it — counting it as an
        indeterminate attempt would bias mastery without any real signal
        either way.

        Ties (equal ``mastery`` and equal ``attempts``) sort deterministically
        by subject then chapter, rather than depending on dict insertion
        order.
        """
        buckets: dict[tuple[str, str], _Bucket] = {}
        for item in self._index.list_items(owner_uid, include_locked=False, limit=_MAX_ITEMS):
            if not item.solve_id:
                continue
            solve = self._index.get_solve(item.solve_id)
            if solve is None:
                continue
            key = (
                item.classification.subject or "unclassified",
                item.classification.chapter or "unfiled",
            )
            bucket = buckets.setdefault(key, _Bucket())
            bucket.attempts += 1
            status = solve.verification.status
            if status is VerificationStatus.VERIFIED:
                bucket.verified += 1
            elif status is VerificationStatus.REFUTED:
                bucket.refuted += 1
            if _is_unanimous(solve.consensus.agreement):
                bucket.unanimous += 1
            if item.created_at > bucket.last_seen:
                bucket.last_seen = item.created_at

        rows = []
        for (subject, chapter), bucket in buckets.items():
            verifiable = bucket.verified + bucket.refuted
            if verifiable > 0:
                basis = "sympy"
                mastery = bucket.verified / verifiable
            else:
                basis = "agreement"
                # Safe: a bucket only exists once `setdefault` has been
                # followed by an increment, so `attempts` is never zero here.
                mastery = bucket.unanimous / bucket.attempts
            rows.append(
                ChapterMastery(
                    subject=subject,
                    chapter=chapter,
                    attempts=bucket.attempts,
                    verified=bucket.verified,
                    verifiable_attempts=verifiable,
                    basis=basis,
                    mastery=mastery,
                    last_seen=bucket.last_seen,
                )
            )
        rows.sort(key=lambda r: (r.mastery, -r.attempts, r.subject, r.chapter))
        return rows
